# FlashDec 接下来工作计划

## 当前基线

- RTX 5070 Week 10 kernel correctness：`88 passed in 5.00s`。
- paged decode 支持 FP16/BF16、head_dim 64/128、MHA/GQA/MQA、变长 batch、block size 8/16/32。
- 当前冻结配置：token-major、`block_size=32`、`num_warps=2`、`num_stages=None`。
- block32 相对 block16 的 full-sweep p50 几何平均加速约 1.31x。
- token-major 在 layout full sweep 中赢得 25/28 个 p50；dim-major p50 几何平均约慢 31.4%。
- 最终 FP16 medium/large p50 为 `0.155520/0.884576 ms`。
- PagedKVCache v2 已支持 request lifecycle、block free/reuse、容量失败原子性、metrics 和 invariant validation。
- RTX 5070 runtime v2 focused：`90 passed in 4.47s`；完整回归：`170 passed in 4.66s`。
- R2-D commit `fa0f89a` 的 144 行正式 multi-layer 矩阵已通过严格校验；fused complete-token p50/p90/TPS 几何平均为 `1.2101x/1.3826x/1.2800x`。
- R2-D 24 个 dtype/case 组合中 20 个三轮 p50 稳定胜出、4 个跨过 1.0、0 个稳定回退；profiler 显示收益来自 append/launch，decode device time 基本不变。

## 总目标

将 FlashDec 从“正确、可测量的 paged decode kernel”推进为“单 GPU LLM decode 执行与 KV Cache 管理原型”。最终系统需要贯通 request lifecycle、Paged KV block allocator、RoPE/KV append、paged attention、dynamic active batch 和端到端 workload 指标。

详细系统边界与完成标准见 `docs/AI_INFRA_SCOPE.md`。

Week 12 之后的长期深化目标、阶段交付物与验收门槛见 `docs/ROADMAP.md`。后续优先顺序固定为：证据闭环 -> block-aware scheduler -> multi-layer KV transaction -> prefix/公开基线二选一。

## 阶段 1：最终 Kernel 基线

状态：已完成。

已交付：

- reference、Triton paged decode、public API。
- FP16/BF16、MHA/GQA/MQA、主要 shape correctness。
- `num_warps`、block size、KV layout 实验。
- 最终默认配置 profiling 和性能报告。

## 阶段 2：冻结 Kernel 配置

状态：已完成。

目标：完成最后一个有边界的参数实验，然后停止无止境调参。

任务：

1. 为 wrapper 增加可选 `num_stages`，保留当前隐式配置作为 baseline。
2. 增加 `default/1/2/3/4` sweep。
3. 固定 token-major、`block_size=32`、`num_warps=2`。
4. 只使用 medium、large、large-batch 做默认决策。
5. 只有 p50 几何平均稳定提升超过 5%，且主要 shape 无明显回退，才修改默认值。
6. 对 block table/mask/offset 只允许一个 time-boxed 实验；无稳定收益就记录负结果并结束 kernel 调优。

最终结果：

- 已完成 wrapper 的可选 `num_stages`，`None` 继续表示 Triton implicit default。
- 已完成 `default/1/2/3/4` 专项 sweep、Profiler 参数/元数据和测试代码。
- RTX 5070 correctness：`88 passed in 5.00s`；full sweep 的 30 条记录均为 `validated=True`。
- 最佳候选 stage 2 的六场景 p50 几何平均仅快约 0.39%，未达到 5% 门槛。
- 保留 `num_stages=None`，不再为 staging 增加 shape dispatch 或额外 Profiler。
- kernel 配置冻结；除 correctness 或明确性能回归外，不再重复 sweep 已冻结参数。

完成标准：

- 最终 kernel config 被代码、测试、benchmark 和文档共同固定。
- 完整 correctness 与 quick regression 通过。
- 后续除非发现 correctness 或明显性能回归，不再重新 sweep 已冻结参数。

## 阶段 3：Paged KV Runtime v2

状态：已完成。

目标：实现真正的请求生命周期和 physical block 内存管理，这是项目从算子走向 AI Infra 的第一条主线。

任务：

1. 实现 `finish_request()` 和 `cancel_request()`。
2. 将请求持有的 physical blocks 归还 free list，并支持后续请求复用。
3. 增加 request state query 和 cache metrics：
   - used/free blocks。
   - block utilization。
   - internal fragmentation。
   - allocation/free/reuse 次数。
4. 保证批量 append 容量不足时不发生 partial mutation。
5. 固定验证单 layer runtime，并在 API/文档中明确多 layer execution 尚不在 `v0.1.0` 范围。

当前实现：

- active -> finished/cancelled 单向状态转换，终态 request 不能继续 append 或重新激活。
- 释放 block 优先进入 free list 前端供新请求复用；释放时不清零物理 K/V。
- 批量 append 在分配前统一计算所需 block，容量不足时不创建新 request、不增长已有 seq_len。
- `metrics()` 报告 block utilization、internal fragmentation、allocation/free/reuse 和 lifecycle 计数。
- `validate_invariants()` 检查 owned/free block 完整覆盖、无重复所有权和终态无 block 泄漏。
- legacy append/runtime v2 路径继续只支持 `num_layers=1`；R2-A 新增的多层写入只能通过 token transaction API。
- RTX 5070 focused correctness：`90 passed in 4.47s`。
- RTX 5070 完整回归：`170 passed in 4.66s`。

核心测试：

```text
add -> append -> finish -> reuse
add -> append -> cancel -> reuse
capacity failure -> no partial mutation
mixed active/finished requests -> metadata correct
request churn -> no leaked blocks
```

完成标准：

- 长时间 request churn 后 used + free 始终等于 max_blocks。
- block reuse 不改变有效 K/V correctness。
- capacity、重复 finish、unknown request 等错误路径明确。

## 阶段 4：RoPE + KV Append 数据路径

状态：三条 RoPE/KV append 路径均已通过 RTX 5070 correctness；full CUDA-event p50 几何平均显示 fused 为 `1.2226x` vs torch，独立 CUDA append 为 `0.9840x`，因此停止独立 append 微调。

目标：实现至少一条原生 CUDA 数据路径，并与 PagedKVCache v2 集成。

任务：

1. 保留 PyTorch RoPE + paged KV append reference 作为语义基线。
2. 保留公开 RoPE API 默认 `torch`，GPU Engine 显式使用 `fused_cuda`。
3. 进入 DecodeEngine 与动态 batch 的 complete-step runtime。

当前实现：

- `apply_rope()`：split-half rotary embedding，支持 partial `rotary_dim`，FP32 计算后回写原 dtype。
- `PagedKVCache.next_positions()`：返回 append 前 position，新 request 为 0，active request 为当前 seq_len。
- `rope_paged_kv_append_ref()`：旋转 Q/K、保持 V 不变、把 rotated K/raw V 写入 paged cache。
- `RopeAppendResult`：返回 rotated Q、pre-append positions、block tables 和 post-append seq_lens。
- focused tests 覆盖手算公式、partial rotary、FP16/BF16/FP32、block 边界、容量失败和 terminal request。
- RTX 5070 focused：`38 passed in 3.60s`；完整回归：`186 passed in 4.96s`。
- PyTorch 为 `2.11.0+cu128`，PyTorch CUDA 为 12.8；Toolkit 前置检查已通过：`nvcc 12.8.93`、`CUDA_HOME=/usr/local/cuda-12.8`、Ninja 1.13.0、GCC/G++ 13.3.0。
- 已实现 lazy JIT `cuda_kv_append()` 与 `PagedKVCache.append_cuda()`；RTX 5070 focused 为 `34 passed in 3.59s`，完整回归为 `198 passed in 5.13s`。
- 已实现并验证 `rope_paged_kv_append(..., append_backend="torch" | "cuda" | "fused_cuda")`；fused focused 为 `66 passed in 44.35s`（含首次 JIT build），完整回归为 `214 passed in 4.52s`。
- Week 11 full CUDA-event：fused 在 6/6 个 p50 case 胜出，几何平均 `1.2226x`（约 18.2% latency 降低）；独立 CUDA append 几何平均 `0.9840x`。详细表见 `benchmarks/results/week11_rope_kv_append_summary.md`。

完成标准：

- CUDA extension 可构建、导入、测试和 benchmark。
- CUDA 写入后的 block tables/seq_lens/K/V 与 reference 一致。
- 能说明 fusion 是否减少 launch 和中间数据访问。

## 阶段 5：DecodeEngine 与动态 Batch

状态：已完成。DecodeEngine v1 的 CPU/reference 与 RTX 5070 fused/Triton correctness 已验证通过。

目标：把 runtime 与 kernel 组织成可以多步运行的单 GPU decode execution engine。

任务：

1. 定义并验证 request 状态：waiting、active、finished、cancelled。
2. 验证 request admission 和 active batch builder。
3. 每一步接收 active requests 的 Q/K/V，执行：

```text
RoPE/KV append -> block_tables/seq_lens -> paged decode -> state update
```

4. 支持不同 context 的请求在不同 step 加入和退出。
5. 验证 cache capacity 不足时返回明确 backpressure/admission 结果。
6. 保证 batch row、request id、block table 和输出一一对应。

完成标准：

- 多步动态 batch 输出与逐请求 reference 对齐。
- 请求完成/取消后 block 被回收并能服务新请求。
- execution engine 不直接依赖 benchmark 脚本内部对象。

## 阶段 6：端到端 Workload 与系统指标

状态：正式 36 行 multi-trial 与 12-case profiler 已完成；所有 invariant/trajectory/range-count 校验通过。p50/p90/TPS 几何平均为 1.0668x/1.0317x/1.0811x，short-churn p50 与 p99 仍不稳定。

目标：证明系统在动态请求负载下的性能和内存行为，而不只报告单 kernel latency。

任务：

1. 建立可复现 synthetic workload：arrival step、initial context、decode length、deterministic cancel interval/probability。
2. 同时报告：
   - kernel latency。
   - complete decode-step p50/p90/p99。
   - tokens/s 和 active batch size。
   - block utilization 和 internal fragmentation。
   - allocation/free/reuse 与 admission failure。
3. 对比至少三种 workload：
   - short requests / high churn。
   - mixed context / steady state。
   - long context / memory pressure。
4. 分析 kernel 优化能否转化为 end-to-end 收益。

当前实现：

- `WorkloadConfig` 定义 deterministic arrival、最大 active batch、每请求 decode token budget、可选 prompt context/stagger，以及带 seed 的 cancel interval/probability 策略。
- `run_synthetic_workload()` 执行 request submit/admit、完整 `DecodeEngine.step()`、finish/cancel 与 backpressure recovery；每个 measured step 使用 CUDA 同步后的 wall-clock。
- prompt prefill 与随机 Q/K/V 生成明确排除在 decode-step latency 外；submit/admit、allocator、append、paged decode 与 post-step lifecycle 明确包含。
- `short_churn`、`mixed_steady`、`long_pressure` 三个 workload 分别验证高 churn、mixed context 稳态和可恢复的 physical-block 压力。
- CSV 同时记录 p50/p90/p99、tokens/s、active batch、block utilization/fragmentation、allocation/free/reuse、backpressure、seed、环境和冻结 kernel 配置。

完成标准：

- workload 结果绑定 commit、seed、环境和配置。
- 能解释 compute、memory、scheduler/runtime overhead 各占什么位置。
- 至少记录一个 kernel 更快但端到端收益有限的系统级负结果，或证明收益可以传递。

## 阶段 7：工程发布

目标：形成可安装、可测试、可复现的 `v0.1.0`。

任务：

1. 完善 README 架构、quick start、支持矩阵和限制。
2. 增加 kernel correctness、runtime state-machine、engine integration 和 quick workload 命令。
3. 在干净 WSL 环境复跑安装和核心验证。
4. 整理 `CHANGELOG.md`、`docs/reproducibility.md` 和 release tag。

完成标准：

- 新环境能运行一个 correctness suite 和一个动态 workload quick benchmark。
- 所有公开性能数字能追溯到命令、硬件、commit 和结果摘要。
- kernel、runtime、engine、workload 四层边界清楚。

## 当前立即执行

R1 已在 RTX 5070 完成 36 行正式策略矩阵并通过严格摘要校验。boundary-deadlock 中 lifetime FIFO + aging 完成 2/2 请求且无取消/死锁；cancel baseline 完成 1/2，greedy baseline 完成 0/2 并触发确定的零进展检测。普通 finite queue 中三种策略均完成 6/6，请勿把 R1 表述为无条件吞吐优化。

R2-D RTX workload 证据已完成。commit `fa0f89a` 的 `12 cases x 2 dtypes x 2 backends x 3 trials = 144 rows` 正式矩阵通过 shape、pair trajectory、transaction、block accounting、rollback、profiler、seed 与 backend-order 严格校验。整体 complete-token p50 降低约 17.4%，TPS 提高约 28.0%；层数从 1 增加到 2/4 时，p50 几何平均由 `1.1112x` 提高到 `1.2567x/1.2690x`。p99 必须继续连同范围报告，不能把 20-repeat 的单轮最大值写成稳定尾延迟结论。

Mac 文档闭环验证已完成：R2-D 10 个 dependency-free tests 通过，正式 summary 重新生成后逐字一致，`compileall`、`git diff --check` 和 `check_release.py --require-evidence` 通过。证据提交 `67bee15` 已在 RTX 5070 完成最终完整回归：`337 passed, 25 subtests passed in 5.82s`，无 skipped 或 failure。R2 功能、correctness、性能与文档证据至此闭环。

当前立即执行顺序：

1. 提交最终 RTX 回归记录。
2. 执行 clean WSL venv editable install、release quick workload 和 release checker。
3. release gate 全部通过后，才把版本从 `0.0.0` 更新为 `0.1.0` 并创建 tag；暂不并行启动 shared prefix。
