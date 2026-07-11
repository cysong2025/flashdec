# FlashDec 接下来工作计划

## 当前基线

- RTX 5070 最终配置 correctness：`76 passed in 4.49s`。
- paged decode 支持 FP16/BF16、head_dim 64/128、MHA/GQA/MQA、变长 batch、block size 8/16/32。
- 当前通用配置：token-major、`block_size=32`、`num_warps=2`。
- block32 相对 block16 的 full-sweep p50 几何平均加速约 1.31x。
- token-major 在 layout full sweep 中赢得 25/28 个 p50；dim-major p50 几何平均约慢 31.4%。
- 最终 FP16 medium/large p50 为 `0.155520/0.884576 ms`。
- PagedKVCache v1 支持 allocate-on-append、block table、seq_len 和容量检查，但不支持 request free/reuse。

## 总目标

将 FlashDec 从“正确、可测量的 paged decode kernel”推进为“单 GPU LLM decode 执行与 KV Cache 管理原型”。最终系统需要贯通 request lifecycle、Paged KV block allocator、RoPE/KV append、paged attention、dynamic active batch 和端到端 workload 指标。

详细系统边界与完成标准见 `docs/AI_INFRA_SCOPE.md`。

## 阶段 1：最终 Kernel 基线

状态：已完成。

已交付：

- reference、Triton paged decode、public API。
- FP16/BF16、MHA/GQA/MQA、主要 shape correctness。
- `num_warps`、block size、KV layout 实验。
- 最终默认配置 profiling 和性能报告。

## 阶段 2：冻结 Kernel 配置

目标：完成最后一个有边界的参数实验，然后停止无止境调参。

任务：

1. 为 wrapper 增加可选 `num_stages`，保留当前隐式配置作为 baseline。
2. 增加 `default/1/2/3/4` sweep。
3. 固定 token-major、`block_size=32`、`num_warps=2`。
4. 只使用 medium、large、large-batch 做默认决策。
5. 只有 p50 几何平均稳定提升超过 5%，且主要 shape 无明显回退，才修改默认值。
6. 对 block table/mask/offset 只允许一个 time-boxed 实验；无稳定收益就记录负结果并结束 kernel 调优。

当前实现状态：

- 已完成 wrapper 的可选 `num_stages`，`None` 继续表示 Triton implicit default。
- 已完成 `default/1/2/3/4` 专项 sweep、Profiler 参数/元数据和测试代码。
- 待 RTX 5070 完成 correctness、quick 和 full sweep；在结果分析前不修改默认值。

完成标准：

- 最终 kernel config 被代码、测试、benchmark 和文档共同固定。
- 完整 correctness 与 quick regression 通过。
- 后续除非发现 correctness 或明显性能回归，不再重新 sweep 已冻结参数。

## 阶段 3：Paged KV Runtime v2

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

目标：实现至少一条原生 CUDA 数据路径，并与 PagedKVCache v2 集成。

任务：

1. 检查并安装匹配 PyTorch CUDA 的 Toolkit/`nvcc`。
2. 先实现 PyTorch RoPE + paged KV append reference。
3. 实现独立 CUDA KV append；通过后再融合 RoPE。
4. 注册 Python op，并保留 PyTorch fallback。
5. 覆盖 FP16/BF16、block 边界、新 block 分配和多 KV head。
6. 对比分离 RoPE+append 与 fused op 的 latency、launch 数和内存访问。

完成标准：

- CUDA extension 可构建、导入、测试和 benchmark。
- CUDA 写入后的 block tables/seq_lens/K/V 与 reference 一致。
- 能说明 fusion 是否减少 launch 和中间数据访问。

## 阶段 5：DecodeEngine 与动态 Batch

目标：把 runtime 与 kernel 组织成可以多步运行的单 GPU decode execution engine。

任务：

1. 定义 request 状态：waiting、active、finished、cancelled。
2. 实现 request admission 和 active batch builder。
3. 每一步接收 active requests 的 Q/K/V，执行：

```text
RoPE/KV append -> block_tables/seq_lens -> paged decode -> state update
```

4. 支持不同 context 的请求在不同 step 加入和退出。
5. cache capacity 不足时返回明确 backpressure/admission 结果。
6. 保证 batch row、request id、block table 和输出一一对应。

完成标准：

- 多步动态 batch 输出与逐请求 reference 对齐。
- 请求完成/取消后 block 被回收并能服务新请求。
- execution engine 不直接依赖 benchmark 脚本内部对象。

## 阶段 6：端到端 Workload 与系统指标

目标：证明系统在动态请求负载下的性能和内存行为，而不只报告单 kernel latency。

任务：

1. 建立可复现 synthetic workload：arrival step、initial context、decode length、cancel probability。
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

阶段 2 只作为短周期收尾：在 RTX 5070 执行 `num_stages` correctness、quick 和 full sweep，依据 Week 10 决策规则冻结 kernel 配置。随后立即进入 PagedKVCache v2 的 `finish/cancel/free/reuse`，而不是继续增加更多孤立算子或参数实验。完整命令见 `docs/weekly/week_10_status.md`。
