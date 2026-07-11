# FlashDec 接下来工作计划

## 当前基线

- RTX 5070 最终配置 correctness：`76 passed in 4.49s`。
- 支持 FP16/BF16、head_dim 64/128、MHA/GQA/MQA、变长 batch、block size 8/16/32。
- 当前通用配置：token-major、`block_size=32`、`num_warps=2`。
- block32 相对 block16 的 full-sweep p50 几何平均加速约 1.31x。
- token-major 在 layout full sweep 中赢得 25/28 个 p50；dim-major p50 几何平均约慢 31.4%。
- 最终默认配置的 FP16/BF16 四场景 profiler 已完成；FP16 medium/large p50 为 `0.155520/0.884576 ms`。

## 总目标

把当前正确、可 benchmark 的 Triton paged decode 原型推进为一个包含最终 profiling 证据、原生 CUDA 数据写入路径和可复现发布流程的完整工程。

## 阶段 1：重建最终性能基线

状态：已完成。

目标：保证性能报告测量的是当前真实默认配置，而不是早期 block16 配置。

任务：

1. 在 profiler 输出中显式记录 `kv_layout=token_major`。（已完成。）
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

阶段 1 已闭环：代码、correctness、8 组最终 profiling 和性能报告均已完成。下一步进入阶段 2，先实现独立的 `num_stages=1/2/3/4` sweep，固定 `token-major + block_size=32 + num_warps=2`，优先比较 medium、large、large-batch，只有跨主要 shape 稳定超过 5% 的方案才调整默认值。
