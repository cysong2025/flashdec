# FlashDec 接下来工作计划

## 当前基线

- RTX 5070 correctness：Paged KV layout 扩展回归为 `73 passed in 9.40s`。
- 支持 FP16/BF16、head_dim 64/128、MHA/GQA/MQA、变长 batch、block size 8/16/32。
- 当前通用配置：token-major、`block_size=32`、`num_warps=2`。
- block32 相对 block16 的 full-sweep p50 几何平均加速约 1.31x。
- token-major 在 layout full sweep 中赢得 25/28 个 p50；dim-major p50 几何平均约慢 31.4%。
- PyTorch profiler 与 Chrome trace 已建立，但现有主要结果来自 block16 阶段，需要用最终配置更新。

## 总目标

把当前正确、可 benchmark 的 Triton paged decode 原型推进为一个包含最终 profiling 证据、原生 CUDA 数据写入路径和可复现发布流程的完整工程。

## 阶段 1：重建最终性能基线

目标：保证性能报告测量的是当前真实默认配置，而不是早期 block16 配置。

任务：

1. 在 profiler 输出中显式记录 `kv_layout=token_major`。（代码已完成，待 RTX 5070 验证。）
2. 用 `block_size=32, num_warps=2` 重跑 FP16/BF16：
   - small：batch=1, context=128。
   - medium：batch=16, context=1024。
   - large：batch=16, context=8192。
   - large-batch：batch=64, context=4096。
3. 更新 `benchmarks/results/week9_summary.md` 和 `docs/performance_report.md`。
4. 复核 CUDA event、PyTorch profiler CUDA time 与有效带宽三套指标是否一致。

完成标准：

- 4 个场景全部通过 reference validation。
- 性能报告不再混用 block16 与 block32 数据。
- 每条结果包含 dtype、layout、block size、num warps、shape 和环境版本。

## 阶段 2：最后一轮 Triton 优化

目标：用受控实验判断 `num_stages` 和索引路径是否还有稳定收益。

任务：

1. 增加 `num_stages=1/2/3/4` sweep，固定其他参数。
2. 优先观察 medium、large、large-batch，避免用单个微小 shape 决策。
3. 对 block table load、mask、offset 计算各做一个最小改动实验。
4. 每次实验都保留 reference correctness、p50/p90 和回滚条件。

完成标准：

- 只有跨主要 shape 稳定超过 5% 的方案才进入默认配置。
- 无收益的实验也记录原因，避免重复尝试。
- 默认配置变化后复跑完整 correctness 和 quick benchmark。

## 阶段 3：CUDA extension

目标：实现 fused RoPE + paged KV append；若范围过大，则先完成独立 KV append CUDA kernel。

任务：

1. 检查并安装与 PyTorch CUDA 版本匹配的 CUDA Toolkit / `nvcc`。
2. 建立 `flashdec/csrc/` 和 Python 注册入口。
3. 先写 PyTorch reference，再实现 CUDA kernel。
4. 覆盖 FP16/BF16、GQA/MQA 使用的 KV head shape、block 边界与新 block 分配。
5. benchmark 单独 append 与 fused RoPE + append 的 latency 和内存流量。

完成标准：

- extension 能构建、导入并通过 correctness。
- block 内偏移和 block table 写入路径有边界测试。
- benchmark 能说明 fusion 是否减少 launch 和中间内存访问。

## 阶段 4：工程发布

目标：形成可安装、可测试、可复现的 `v0.1.0`。

任务：

1. 补齐 README quick start、支持矩阵、默认配置和已知限制。
2. 增加一键 correctness 与 quick benchmark 命令。
3. 在干净 WSL 环境复跑安装和核心验证。
4. 整理 `CHANGELOG.md`、`docs/reproducibility.md` 和 release tag。

完成标准：

- 新环境按文档可运行一个 correctness test 和一个 quick benchmark。
- 所有公开性能数字能追溯到命令、硬件、commit 和结果摘要。
- 仓库只包含源码、测试、benchmark、设计/实验文档及必要配置。

## 当前立即执行

`profile_paged_decode.py` 已完成以下扩展：

- `--dtype both` 一次运行 FP16/BF16。
- `--kv-layout token_major|dim_major` 显式控制并记录 layout。
- `--case all` 覆盖 small、medium、large、large-batch。
- summary 每行记录 shape、dtype、layout、block size、num warps、GPU、PyTorch、CUDA 和 profile 路径。
- 输出文件名包含 layout、block size 和 num warps，避免不同配置互相覆盖。

下一步在 RTX 5070 完成 correctness、smoke 和 full profiling。阶段 1 验证完成前不开始新的 kernel 优化，避免在过期基线上做判断。

```bash
python -m pytest -vv \
  tests/test_profile_paged_decode.py \
  tests/test_paged_cache.py \
  tests/test_paged_decode.py \
  tests/test_public_api.py

python benchmarks/profile_paged_decode.py \
  --case small \
  --dtype both \
  --kv-layout token_major \
  --block-size 32 \
  --num-warps 2 \
  --repeat 3 \
  --output-dir benchmarks/profiles/week9_final_default_smoke \
  --summary-output benchmarks/results/week9_final_default_smoke.md

python benchmarks/profile_paged_decode.py \
  --case all \
  --dtype both \
  --kv-layout token_major \
  --block-size 32 \
  --num-warps 2 \
  --repeat 10 \
  --output-dir benchmarks/profiles/week9_final_default \
  --summary-output benchmarks/results/week9_final_default_summary.md
```
