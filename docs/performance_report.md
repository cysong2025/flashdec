# FlashDec 性能报告

本文记录 FlashDec paged decode attention、DecodeEngine、Scheduler、multi-layer transaction、Shared Prefix、trusted/integrated workload 与 FlashInfer 有限基线的性能结论和 profiling 证据。

## 当前结论

- Week 7 已完成真实 decode shape 覆盖：`head_dim=128`、FP16/BF16、GQA/MQA、batch/context sweep。
- Week 8 完整 `num_warps=2/4/8` sweep 显示：`num_warps=2` 在 28 个 dtype/case 组合中全部 p50 最优。
- block-size full sweep 显示 block32 在 24/28 个 p50 组合中最优，相对 block16 的 p50 几何平均加速约 1.31x。
- KV layout full sweep 显示 token-major 在 25/28 个 p50 和 25/28 个 p90 组合中最优；dim-major 的 p50 几何平均约慢 31.4%。
- 当前 paged decode 冻结配置为 `token-major + block_size=32 + num_warps=2 + num_stages=None`。
- Week 10 `num_stages` full sweep 的最佳候选 stage 2 仅取得约 1.0039x p50 几何平均加速，未达到 5% 门槛；最终冻结为 `num_stages=None`。
- Week 8 有效带宽估算显示：长 context 下 kernel 更接近 K/V 访存主导。
- Week 9 PyTorch profiler 和 Chrome trace 进一步显示：large context 下总估算流量几乎全部来自 K/V 读取，kernel 时间随 context 近似线性增长。
- 最终默认配置已完成 FP16/BF16 的 small、medium、large、large-batch profiling；correctness 为 `76 passed in 4.49s`。
- 最终 FP16 medium/large p50 分别为 `0.155520/0.884576 ms`，相对早期 block16 event baseline 加速 `1.305x/1.481x`。
- 当前 RTX 5070 WSL 环境缺少 `ncu` / `nsys`，Nsight 硬件计数暂未补充。
- Week 12 正式 36 行 multi-trial 显示 fused complete-step p50/p90/TPS 几何平均为 1.0668x/1.0317x/1.0811x；short-churn p50 跨 trial 穿过 1.0。
- Week 12 正式 12-case profiler 显示 fusion 将 CUDA event 数减少 21.8%-45.6%，paged decode device time 仅变化 -1.7%-+1.1%。
- R1 Scheduler 的稳定交付是容量安全和进展保证：boundary-deadlock 下 lifetime policy 100% completion，cancel/greedy 对照分别为 50%/0%；它不是无条件 latency 加速。
- R2 multi-layer fused complete-token p50/p90/TPS 几何平均为 `1.2101x/1.3826x/1.2800x`，24 个分组中 20 个 p50 三轮稳定胜出；收益主要来自 append/launch 路径。
- R3 75% shared-prefix hit 节省 `68.8%`/`5.5 MiB` context KV capacity，并把 bounded-pool admission 从 `9/16` 提高到 `16/16`；完整 step latency 为 near-neutral/no stable direction。
- R4-A trusted path 的 complete-token p50/TPS 为 `1.7307x/1.7131x`，16/16 p50 分组稳定胜出；R4-B 未过 16/16 keep gate 并已回滚，R4-C 完成组合 correctness。
- R5 的共同 paged-decode kernel-only 对比中，FlashInfer CUDA-core/tensor-core p50 ratio 几何平均为 `1.2003x/1.2284x`；p99 和端到端 serving 不作稳定胜负声明。

## Profiling 场景与命令

代表场景：

| 场景 | shape | 目的 |
| --- | --- | --- |
| small | `batch=1, context=128` | 观察固定开销和 launch overhead |
| medium | `batch=16, context=1024` | 观察常规 decode workload |
| large | `batch=16, context=8192` | 观察长 context 访存瓶颈 |

PyTorch profiler 命令：

```bash
python benchmarks/profile_paged_decode.py --case all --repeat 10 --output-dir benchmarks/profiles/week9_paged_decode
```

Chrome trace 命令：

```bash
python benchmarks/profile_paged_decode.py --case medium --repeat 10 --export-trace --output-dir benchmarks/profiles/week9_paged_decode_trace
```

## 早期 Block16 PyTorch Profiler 结果

RTX 5070 上已完成 PyTorch profiler 三场景 profiling：

```bash
python benchmarks/profile_paged_decode.py --case all --repeat 10 --output-dir benchmarks/profiles/week9_paged_decode
```

| 场景 | profiler CUDA total | profiler CUDA avg/call | 观察 |
| --- | ---: | ---: | --- |
| small_b1_ctx128 | 74.285 us | 7.428 us | kernel 本体极短，固定开销和 launch overhead 更重要 |
| medium_b16_ctx1024 | 1.585 ms | 158.493 us | 常规 decode workload，适合作为后续 profiler 重点 |
| large_b16_ctx8192 | 12.524 ms | 1.252 ms | 长 context 下 kernel 时间明显上升，符合 K/V 访存主导判断 |

注意：

- 这里记录的是 PyTorch profiler 表中的 CUDA kernel time。
- CUDA event 的 p50/p90 已由脚本写入 `benchmarks/profiles/...txt` 和 `benchmarks/results/week9_summary.md`，本次粘贴日志未展开这些文件内容。
- 当前环境缺少 `ncu` / `nsys`，暂时无法补 Nsight Compute / Nsight Systems 的硬件计数，例如 memory throughput、achieved occupancy、register 使用。

## 早期 Block16 Chrome Trace 与 CUDA Event 结果

RTX 5070 上已补充 medium / large Chrome trace：

| 场景 | event mean | event p50 | event p90 | effective_total_gbps_p50 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| medium_b16_ctx1024 | 0.203555 ms | 0.202880 ms | 0.208352 ms | 1059.3716 | 常规 decode workload，适合作为 Week 10 baseline |
| large_b16_ctx8192 | 1.314675 ms | 1.309984 ms | 1.328576 ms | 1236.1614 | 长 context 下 K/V 读取占主导 |

large trace 关键估算：

| metric | value |
| --- | ---: |
| `estimated_kv_read_bytes` | 1,618,067,456 |
| `estimated_total_bytes` | 1,619,351,552 |
| `profiler CUDA avg/call` | 1.230 ms |
| `cuLaunchKernelEx avg/call` | 8.602 us |

解释：

- large 场景中 `estimated_kv_read_bytes` 与 `estimated_total_bytes` 几乎相同，说明总数据量主要来自 K/V cache 读取。
- context 从 `1024` 增到 `8192` 后，kernel 时间也接近按比例增长，支持 memory-bound 判断。
- large 场景下 launch overhead 相比 kernel 本体很小，优先优化方向应放在 KV layout、block size 和索引/访存路径，而不是 Python wrapper。

## 最终默认配置 Profiling（2026-07-12）

固定配置：

- `kv_layout=token_major`
- `block_size=32`
- `num_warps=2`
- FP16/BF16
- warmup 5，repeat 10

correctness：

```text
76 passed in 4.49s
```

CUDA event 结果：

| case | FP16 p50 | BF16 p50 | FP16 effective GB/s | BF16 effective GB/s |
| --- | ---: | ---: | ---: | ---: |
| small_b1_ctx128 | 0.015328 ms | 0.038176 ms | 90.8894 | 36.4929 |
| medium_b16_ctx1024 | 0.155520 ms | 0.160864 ms | 1325.1028 | 1281.0822 |
| large_b16_ctx8192 | 0.884576 ms | 0.928064 ms | 1745.7439 | 1663.9404 |
| large_batch_b64_ctx4096 | 1.934560 ms | 1.961216 ms | 1605.6542 | 1583.8309 |

与早期 block16 event baseline 对比：

| case | block16 FP16 p50 | block32 FP16 p50 | speedup |
| --- | ---: | ---: | ---: |
| medium_b16_ctx1024 | 0.202880 ms | 0.155520 ms | 1.305x |
| large_b16_ctx8192 | 1.309984 ms | 0.884576 ms | 1.481x |

观察：

- medium、large、large-batch 中，BF16 相对 FP16 的 p50 差异仅约 3.4%、4.9%、1.4%，继续支持瓶颈主要来自访存而非低精度算力的判断。
- large 和 large-batch 的估算有效带宽达到约 1.58-1.75 TB/s，GPU 工作量增大后带宽利用率明显高于 small。
- small 的 p50/p90 差异明显且延迟低于 0.1 ms，固定开销和环境抖动占比高，不用于 dtype 或默认配置决策。
- FP16 medium/large 的 profiler kernel avg 分别约为 `98.834 us/828.547 us`，与 CUDA event 的优化方向一致。
- 部分 BF16/large-batch profiler 表只捕获 7-9 次 kernel event，而请求 repeat 为 10。延迟结论以完整的 CUDA-event p50/p90 为准，profiler 表只用于 kernel 归因和趋势判断。

原始精简结果见 `benchmarks/results/week9_final_default_summary.md`；详细 profiler 文本保存在本地 `benchmarks/profiles/week9_final_default/`。

## 已验证有效优化

### `num_warps=4 -> 2`

证据：

- Week 8 full sweep 中，`num_warps=2` 在 28 个 dtype/case 组合中全部 p50 最优。
- `num_warps=2` 相比 `num_warps=4` 平均 p50 加速约 2.10x。
- `num_warps=2` 相比 `num_warps=8` 平均 p50 加速约 3.75x。

解释：

- 当前 kernel 每个 program 处理一个 `(sequence, q_head)`。
- 对当前 `block_size=16`、`head_dim=128` 的实现，更多 warps 没有带来足够收益，反而可能增加调度、同步或 register pressure。
- 较少 warps 更适合当前工作粒度。

## 可选硬件计数假设（不阻塞交付）

- K/V cache global memory load 是否占主导。
- block table 间接索引在长 context 下是否可见。
- `seq_len`、mask 和 logical block loop 的控制开销是否影响短 context。
- 当前 register 使用是否限制 occupancy。

## Profiling 最终判断

- small shape：最终 FP16 profiler kernel avg 约 4.8 us/call，固定开销与测量抖动更值得关注。
- medium shape：最终 FP16 profiler kernel avg 约 98.8 us/call，曾作为 staging/profiler 实验的主力代表场景；当前只在明确回归或独立研究启动时复用。
- large shape：最终 FP16 profiler kernel avg 约 828.5 us/call，随 context 变长明显增长，继续支持 memory-bound 判断。
- PyTorch profiler 已经能确认主要 CUDA 时间集中在 `_paged_decode_attention_kernel`；但它还不能替代 Nsight 的硬件计数。
- 当前环境没有 `ncu` / `nsys`；既有结论使用 CUDA event、PyTorch profiler 和逻辑带宽估算。Nsight 硬件计数不属于当前交付，只有明确启动独立研究时再补充。

## Week 12 Complete-step Multi-trial（2026-07-12）

正式矩阵固定 RTX 5070、commit `3708b87`、Triton decode、block size 32 和 2 warps。三轮使用 seed 431/432/433，并交替 torch/fused backend 顺序。

| dtype | workload | p50 median [min, max] | p90 median | p99 median [min, max] | TPS median | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| FP16 | short-churn | 1.0001x [0.9311, 1.0042] | 0.9917x | 3.1075x [1.8226, 3.5203] | 1.0928x | p50 不稳定 |
| FP16 | mixed-steady | 1.0927x [1.0837, 1.1109] | 1.1248x | 1.2761x [1.2699, 2.0583] | 1.1004x | 稳定提升 |
| FP16 | long-pressure | 1.0890x [1.0614, 1.1274] | 1.0900x | 1.0103x [0.4886, 1.0766] | 1.0899x | p50 稳定，p99 不稳定 |
| BF16 | short-churn | 1.0366x [0.9892, 1.0508] | 1.2267x | 0.6758x [0.2444, 5.0578] | 1.0735x | p50/p99 不稳定 |
| BF16 | mixed-steady | 1.0948x [1.0882, 1.2193] | 1.1779x | 1.1966x [1.0020, 1.3348] | 1.1651x | 稳定提升 |
| BF16 | long-pressure | 1.0744x [1.0741, 1.1054] | 1.0854x | 1.0688x [0.3759, 3.3289] | 1.0754x | p50 稳定，p99 不稳定 |

总体几何平均为 p50 1.0668x、p90 1.0317x、mean/TPS 1.0811x。p99 几何平均虽然为 1.2590x，但范围跨越极大，不能作为稳定收益。

## Week 12 Engine 阶段归因（2026-07-12）

| dtype | workload | instrumented wall p50 | CUDA event 减少 | Engine CPU total 减少 | decode device 变化 |
| --- | --- | ---: | ---: | ---: | ---: |
| FP16 | short-churn | 1.1679x | 21.8% | 27.0% | +0.7% |
| FP16 | mixed-steady | 1.2768x | 22.8% | 28.5% | +0.7% |
| FP16 | long-pressure | 1.1000x | 45.6% | -3.7% | +1.1% |
| BF16 | short-churn | 1.0653x | 21.8% | 11.8% | +0.7% |
| BF16 | mixed-steady | 1.3346x | 22.8% | 28.2% | -1.7% |
| BF16 | long-pressure | 1.1523x | 45.6% | 10.0% | -0.0% |

这里的 profiler totals 是 nested attribution，不能相加，也不能替代 non-instrumented benchmark。Attention device time 基本不变，而事件数显著下降，支持“fusion 优化 append/launch/runtime 路径”的判断。long-pressure FP16 的 CPU total 与 instrumented p99 仍有回退，是必须保留的负结果。

## R1 Block-aware Scheduler 正式矩阵（2026-07-13）

commit `16de9d4` 的 RTX 5070 正式矩阵为 2 cases × 2 dtypes × 3 policies × 3 trials，共 36 行。boundary-deadlock 下默认 lifetime FIFO + aging 达到 100% completion、0 cancel、0 deadlock；cancel-on-backpressure 只有 50% completion，greedy-step-only 为 0% completion，且每次 trial 都检测到 1 次确定 deadlock。finite queue 下三策略都完成 100%，所以 R1 的可交付结论是容量承诺、公平等待和进展保证，不是普通 workload 的无条件 speedup。完整数据见[R1 正式摘要](../benchmarks/results/r1_scheduler_workload_trials3_summary.md)。

## R2 Multi-layer Transaction 正式矩阵（2026-07-15）

commit `fa0f89a` 的正式矩阵为 12 cases × 2 dtypes × 2 backends × 3 trials，共 144 行。fused complete-token p50/p90/TPS 几何平均为 `1.2101x/1.3826x/1.2800x`；per-layer append device、decode device 与 CUDA event ratio 分别为 `1.6103x/1.0024x/1.9784x`，说明主要收益来自 append 与 launch 数减少，而不是 attention device kernel 改变。

24 个 dtype/case 分组中 20 个三轮 p50 稳定胜出，4 个范围跨 1，没有稳定 torch-faster case。每轮只有 20 repeats，nearest-rank p99 接近样本最大值；BF16 `l4_b4_c128` 还保留独立 profiler attribution anomaly，因此不声明生产级尾延迟收益。完整数据见[R2 正式摘要](../benchmarks/results/r2_multi_layer_engine_trials3_summary.md)。

## R3 Shared Prefix 正式矩阵（2026-07-17）

正式矩阵固定 RTX 5070、commit `1d5d8d0`、16 requests、128-token context、block size 32、fused CUDA append 与 Triton decode。0%/25%/50%/75% hit rate 覆盖 FP16/BF16 和 3 trials，共 24 行；case 顺序逐轮旋转。严格 validator 已检查 matrix、seed、capacity commitment、block/byte accounting、materialized context、immutable prefix、eviction 与最终 cleanup。

两种 dtype 的内存与接纳结果一致：

| hit rate | bounded-pool admission | context physical/logical blocks | context saved | peak blocks | saved MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 9/16 | 64/64 | 0.0% | 80 | 0.000 |
| 25% | 12/16 | 52/64 | 18.8% | 68 | 1.500 |
| 50% | 15/16 | 36/64 | 43.8% | 52 | 3.500 |
| 75% | 16/16 | 20/64 | 68.8% | 36 | 5.500 |

稳定结论：

- shared-prefix hit rate 提高时，physical context blocks 按 ownership 公式确定性下降；75% hit 避免 44 个重复 context blocks。
- 同一个 48-block bounded pool 中，第一次调度可接纳请求数从 9 提高到 16，证明节省的 physical KV 可以直接转化为 admission capacity。
- 每个 request 的 decode tail 始终私有，所以 75% 的 context saving 为 68.8%，完整 peak block reduction 为 55.0%，不能混为同一个百分比。
- 非零 hit-rate 的 prefix attach p50 为 `0.388-0.736 us`，相对毫秒级 complete step 很小。

这里的 saved MiB 是避免占用的 KV pool capacity，不是 fixed-full-batch probe 中直接观察到的进程显存下降。该 probe 为保持 tensor shape 相同，在每个 hit rate 下都预分配 80 blocks；实际用途是提高固定 pool admission，或在固定 request target 下将 pool right-size 到更小的 `max_blocks`。

限制与负结果：paired p50 显示 FP16 25% 为 `1.0672x [1.0076,1.1174]`，是唯一三轮稳定更快的非零 case；FP16/BF16 50% 与 BF16 25% 均跨过 1。75% hit 在 FP16/BF16 都三轮稳定更慢，p50 ratio 分别为 `0.9377x [0.9298,0.9870]` 与 `0.9054x [0.8602,0.9816]`，TPS ratio 分别为 `0.9569x [0.8416,0.9712]` 与 `0.9022x [0.8271,0.9795]`。

拆分 attribution 后，75% scheduler p50 ratio 在 FP16/BF16 分别为 `0.8716x` 与 `0.8958x`，均稳定回退；BF16 Engine p50 ratio 也稳定回退到 `0.9025x`，FP16 Engine ratio 则跨 1。该 3-trial 结果保留为 R3-D 优化前基线；submission-time metadata cache 针对 scheduler 重复 lookup，不能预先解释独立的 Engine 波动。

## R3-D 优化后 confirmation（2026-07-18）

commit `fe72e27` 在同一 RTX 5070、PyTorch `2.11.0+cu128`、CUDA 12.8 环境下完成 8 trials。seed `613-620`，四种 hit-rate 顺序各轮转两次，FP16/BF16 共 64 行并通过严格 validator。targeted hot-path test、focused 与完整 correctness 分别为 `1 passed`、`61 passed, 8 subtests passed` 和 `361 passed, 25 subtests passed`。

容量与 admission 轨迹再次严格复现：75% hit 的 context physical blocks 为 `20/64`，相对 0% 避免 44 blocks、`68.8%`/`5.5 MiB` KV-pool capacity；peak blocks 从 `80` 降至 `36`，固定 48-block pool 的 admission 从 `9/16` 提高到 `16/16`。

| dtype | hit rate | complete p50 median [min,max] | scheduler p50 | Engine p50 |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 25% | `1.0300x [0.9295,1.1273]` | `0.9747x [0.9484,1.0918]` | `1.0349x [0.9388,1.1352]` |
| FP16 | 50% | `1.0011x [0.8285,1.0998]` | `0.9817x [0.8319,1.1218]` | `1.0003x [0.8273,1.0990]` |
| FP16 | 75% | `1.0454x [0.7654,1.1811]` | `0.9961x [0.5970,1.1312]` | `1.0472x [0.7829,1.1933]` |
| BF16 | 25% | `1.0207x [0.3056,1.0427]` | `0.9867x [0.9518,1.0238]` | `1.0242x [0.2932,1.0461]` |
| BF16 | 50% | `1.0088x [0.9332,1.0534]` | `0.9853x [0.8832,1.0798]` | `1.0119x [0.9182,1.0556]` |
| BF16 | 75% | `1.0094x [0.9376,1.0556]` | `0.9850x [0.9590,1.0701]` | `1.0040x [0.9402,1.0546]` |

所有非零 case 的 complete、scheduler 和 Engine p50 range 都跨过 1，p90/p99/TPS range 也没有稳定方向。中位数大多接近 1，但 BF16 trial 1 的 25% case 出现 Engine 主导的整行慢点，FP16 trial 7 的 25% case 出现尾部尖峰；相同顺序第二轮均未复现。端点设备快照不能证明运行中不存在瞬时干扰，因此不猜测根因。

最终结论：R3-D 证明了热路径重复 lookup 已被移除，但当前数据不能量化跨 commit 的因果 speedup。R3 性能冻结为 near-neutral/no stable direction；稳定收益只声明 physical KV capacity、bounded-pool admission 与 ownership correctness。完整数据见[R3 8-trial 摘要](../benchmarks/results/r3_shared_prefix_workload_trials8_summary.md)。

## R4-A Trusted Transaction 正式结果（2026-07-19）

R2 profiler 表明 multi-layer complete-token 收益主要来自 append/launch 路径，而 decode device time 近似不变。代码审计进一步定位到 Cache-owned fused transaction 每层仍在公开 raw primitive 中执行 5 次 CUDA reduction + `.item()`：block id 上下界、block offset 上下界与 position 非负。allocator 已在 host 上生成并拥有这些位置，因此 R4-A 将这组证明前移到 `begin_token()`，Cache public transaction API 回查内部 state 后使用 private trusted raw launch；public raw primitive 的防御性检查保持不变。

commit `4018449` 在 RTX 5070、CUDA 12.8 上完成 `8 cases x 2 dtypes x 2 paths x 5 trials = 160 rows`、80 个 paired trials 的正式 A/B。全部 16 个 `dtype x case` 分组均为 `trusted_faster`，且每组五轮 p50 最小值都大于 1；预设的 overall p50 `>=1.05x` 且所有分组不穿过 1 的 gate 通过。

| 指标 | trusted vs checked |
| --- | ---: |
| complete-token p50 | `1.7307x` |
| complete-token p90 | `1.6751x` |
| complete-token p99 | `1.6944x` |
| decode TPS | `1.7131x` |
| profiler append CPU/layer | `2.3612x` |

CPU-only profiler 证明 trusted 路径删除了每层 5 次 item/local-scalar 同步；这项归因不代表 kernel math 或 device execution 本身更快。16 个分组中有 7 个 p99 `[min,max]` 穿过 1，因此只能声明稳定的 complete-token p50 改善，不能把 overall p99 几何平均写成稳定尾延迟收益。focused `73 passed, 23 subtests passed` 和完整 `410 passed, 48 subtests passed` 均通过。R4-A 已冻结；完整数据见[R4-A 五轮正式摘要](../benchmarks/results/r4_fused_transaction_fast_path_trials5_summary.md)。

## R4-B Persistent Metadata 正式负结果（2026-07-19）

R4-B commit `8047a9c` 在同一 trusted CUDA/Triton math 下比较 materialized 与 persistent metadata lifetime。focused `101 passed`、完整回归 `434 passed, 48 subtests passed`；正式矩阵为 FP16/BF16、8 cases、2 paths、5 trials，共 160 rows/80 pairs。exact parity、block/transaction/Engine trajectory、rollback、metadata build/reuse/release、CPU ranges 与 terminal zero-resident 均通过严格校验。

persistent 将 Cache transaction views 从 l2/l4 的 `6/10` 降为 1，并分别复用 2/4 层。overall complete-token p50/TPS 与 append CPU/layer 为 `1.2493x/1.2392x/3.0366x`，但只有 13/16 个分组的五轮 p50 最小值严格大于 1；BF16 `l2_b4_c1024`、FP16 `l2_b16_c128` 与 FP16 `l4_b16_c128` 跨过 1。预注册 keep 门要求 16/16，因此正式状态为 fail，主线恢复 R4-A/materialized 默认。该结果说明 metadata reuse 稳定减少 host append 开销，但不能证明所有完整 token 场景都稳定受益；完整数据见[R4-B 五轮正式负结果](../benchmarks/results/r4_persistent_transaction_metadata_trials5_summary.md)。

## R4-C Integrated Scheduled Multi-layer 正式结果（2026-07-21）

commit `6912894` 在 RTX 5070、PyTorch `2.11.0+cu128`、CUDA 12.8 上完成 focused `60 passed, 17 subtests passed in 3.09s`、完整回归 `425 passed, 57 subtests passed in 6.52s`，并通过 FP16 `l2_c32` quick 与预注册的 24-row/3-trial FP16/BF16 正式矩阵。

| layers / context | BF16 p50 ms / TPS | FP16 p50 ms / TPS |
| --- | ---: | ---: |
| 2 / 64 | `1.389153 / 122.698` | `1.371000 / 126.641` |
| 2 / 128 | `1.360588 / 66.123` | `1.446410 / 65.859` |
| 4 / 64 | `2.130567 / 83.670` | `2.188326 / 81.959` |
| 4 / 128 | `2.146924 / 44.887` | `2.371724 / 43.070` |

所有行的 reference digest、dynamic admission/defer/completion/cancellation trajectory、layer failure rollback、transaction/prefix 计数、prefix lifetime、released-block reuse 与 final zero-used cleanup 均通过 strict validator。因此 R4-C 的组合 correctness/lifecycle gate 通过，R4 阶段完成。

这是一条只有 10 个 logical steps 的功能轨迹。p90/p99 受 private miss 的 caller-supplied multi-layer context-write admission steps 主导，不能解释为 steady-state decode tail；矩阵也不比较 R4-B candidate、不证明 shared prefix latency speedup。随机构建、prefix registration 与 terminal eviction 均在计时区间外。完整数据见[R4-C 正式摘要](../benchmarks/results/r4_integrated_scheduled_multi_layer_trials3_summary.md)。

## R5 FlashInfer 有限公开基线（2026-07-26）

commit `d7d4feb` 在 RTX 5070、Python 3.12.3、PyTorch `2.11.0+cu128`、Triton `3.6.0`、CUDA Toolkit 12.8.1、NVCC 12.8.93 与 `flashinfer-python==0.6.15.post1` 上完成预注册的 `4 cases x 2 dtypes x 3 backends x 3 trials = 72 rows`。post-schema focused 为 `93 passed, 37 subtests passed`，完整回归为 `453 passed, 94 subtests passed`；strict summary 和 clean-tree release check 均通过。

三个 backend 共用 Q/K/V、logical pages、page table、sequence lengths、softmax scale、dtype 和 seed。FlashDec token-major physical layout 对应 FlashInfer `HND`；CUDA event 只包围 `run`/kernel dispatch，排除输入构造、reference validation、FlashInfer plan/JIT、workspace 与 metadata 构建。因此下面只回答共同 paged-decode shape 的 kernel-only 表现，不能与 FlashDec scheduler/transaction/runtime 或完整 serving 直接相除。

| FlashInfer 执行选项 | 8 组 p50 ratio 几何平均 | 三轮 range 全部高于 1 | 最小 range 下界 |
| --- | ---: | ---: | ---: |
| FA2 CUDA-core | `1.2003x` | `8/8` | `1.0231x` |
| FA2 tensor-core | `1.2284x` | `8/8` | `1.1197x` |

ratio 定义为 `FlashDec p50 / FlashInfer p50`，大于 1 表示对应 FlashInfer 路径延迟更低；TPS ratio 与其同方向。两条执行选项合计为 16/16 个 backend/dtype/case 三轮 p50 range 严格高于 1。它们来自同一个固定 FlashInfer 安装，不把合计 16 项当作独立系统样本，也不在看到结果后只选择较快的路径。

稳定性限制同样属于结果：FP16 small 的 CUDA-core/tensor-core 上界分别达到 `2.8514x/4.1659x`，呈现与两条比较共用的 FlashDec baseline 扰动一致的特征，不能作为典型收益。绝对 p90 中位数在 16/16 组低于 FlashDec，但 2/16 三轮范围重叠；绝对 p99 中位数在 14/16 组低于 FlashDec，7/16 范围重叠，另外两组 tensor-core 中位数方向反转。每 row 只有 50 repeats，p99 接近样本最大值，因此不声明稳定或生产级尾延迟优势。完整逐组数字见[R5 正式摘要](../benchmarks/results/r5_flashinfer_paged_decode_trials3_summary.md)。

## 后续工作边界

1. kernel 配置已经冻结，不再重复 `num_warps`、block size、layout 或 `num_stages` sweep。
2. Block-aware Scheduler、Multi-layer KV Token Transaction、R4-A trusted validation 与 R4-C integrated workload 均已完成；R4-B 已完成评估并按稳定性门回滚。
3. 项目整理已完成当前状态、结果索引和证据边界统一；不把文档整理解释为新的 GPU 性能运行。
4. public-source gate 只把已验证证据整理为 `0.0.0` research preview，没有产生新的 GPU 结果；clean-install、新环境复现、`v0.1.0` 版本与 tag 留在未来独立稳定发布 gate。
