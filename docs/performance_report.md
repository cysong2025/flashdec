# FlashDec 性能报告

本文记录 FlashDec paged decode attention、DecodeEngine 和 multi-layer transaction 的性能结论与 profiling 证据。

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

## 需要验证的瓶颈

- K/V cache global memory load 是否占主导。
- block table 间接索引在长 context 下是否可见。
- `seq_len`、mask 和 logical block loop 的控制开销是否影响短 context。
- 当前 register 使用是否限制 occupancy。

## Profiling 最终判断

- small shape：最终 FP16 profiler kernel avg 约 4.8 us/call，固定开销与测量抖动更值得关注。
- medium shape：最终 FP16 profiler kernel avg 约 98.8 us/call，可作为后续参数实验的主力代表场景。
- large shape：最终 FP16 profiler kernel avg 约 828.5 us/call，随 context 变长明显增长，继续支持 memory-bound 判断。
- PyTorch profiler 已经能确认主要 CUDA 时间集中在 `_paged_decode_attention_kernel`；但它还不能替代 Nsight 的硬件计数。
- 因当前环境没有 `ncu` / `nsys`，Week 10 先使用 CUDA event、PyTorch profiler 和逻辑带宽估算推进优化实验；后续环境具备时再补硬件计数。

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

## 后续工作边界

1. kernel 配置已经冻结，不再重复 `num_warps`、block size、layout 或 `num_stages` sweep。
2. Block-aware Scheduler 与 Multi-layer KV Token Transaction 已完成，后续优化必须基于新的 correctness 或系统证据。
3. clean-install、版本与 tag 留在最终 release gate。
4. Shared Prefix R3 已完成；当前不继续围绕同一数据调参。固定版本公开基线与 release 均暂停，未来若执行仍必须统一功能与计时边界。
