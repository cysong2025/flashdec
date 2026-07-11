# Week 10 num_stages Sweep Summary

## 环境与配置

- 日期：2026-07-12
- GPU：NVIDIA GeForce RTX 5070
- PyTorch：2.11.0+cu128
- CUDA：12.8
- layout：token-major
- block size：32
- num warps：2
- warmup：5
- repeat：30
- correctness：`88 passed in 5.00s`
- benchmark：3 cases × 2 dtypes × 5 staging configs，共 30 条记录，全部 `validated=True`

## CUDA event p50

| dtype | case | default | stage 1 | stage 2 | stage 3 | stage 4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | medium_b16_ctx1024 | 0.154304 | 0.156000 | 0.153792 | 0.153184 | 0.153536 |
| FP16 | large_b16_ctx8192 | 0.933152 | 0.916448 | 0.914240 | 0.914432 | 0.913216 |
| FP16 | large_batch_b64_ctx4096 | 1.973856 | 1.973120 | 1.977280 | 1.969600 | 1.969696 |
| BF16 | medium_b16_ctx1024 | 0.157728 | 0.158944 | 0.157888 | 0.157920 | 0.158080 |
| BF16 | large_b16_ctx8192 | 0.958048 | 0.958432 | 0.955936 | 0.962208 | 0.963328 |
| BF16 | large_batch_b64_ctx4096 | 2.007680 | 2.004160 | 2.007232 | 2.009920 | 2.007008 |

单位为 ms。加速比使用 `default p50 / candidate p50`，并对六个组合取几何平均。

## 汇总

| num_stages | 六场景几何平均加速 | p50 胜场 | 最大 p50 回退 | FP16 几何平均 | BF16 几何平均 | 最大 p90 回退 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0002x | 3/6 | 1.10% | 1.0025x | 0.9979x | 1.48% |
| 2 | 1.0039x | 4/6 | 0.17% | 1.0074x | 1.0005x | 无回退 |
| 3 | 1.0038x | 3/6 | 0.43% | 1.0100x | 0.9978x | 13.38% |
| 4 | 1.0036x | 4/6 | 0.55% | 1.0096x | 0.9975x | 18.93% |

## 结论

stage 2 是总体最优候选，但 p50 几何平均只提升约 0.39%，远低于预先设定的 5% 门槛。stage 3/4 的 FP16 收益没有传递到 BF16，并伴随 BF16 medium p90 回退。

因此保留 `num_stages=None`，让 Triton 使用 implicit default；不增加显式 staging 默认值、shape dispatch 或 dtype dispatch。最终通用 kernel 配置冻结为 token-major、`block_size=32`、`num_warps=2`、implicit staging。

原始 CSV/log 保存在本地并由 `.gitignore` 排除；本摘要保存足够的环境、命令配置、核心数据和决策依据用于公开复现。
