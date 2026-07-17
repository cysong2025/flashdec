# FlashDec 当前状态与后续目标

本文只记录尚未完成的工程目标。已经结束的阶段、实验数据和设计取舍分别归档在[项目演进](PROJECT_PLAN.md)、[性能报告](performance_report.md)和[路线图](ROADMAP.md)。

## 当前基线

FlashDec 已完成从 paged decode kernel 到单 GPU decode runtime 的主链路：

```text
Block-aware Scheduler
        -> DecodeEngine
        -> Multi-layer KV Token Transaction
        -> Fused RoPE/KV Append
        -> Triton Paged Decode Attention
```

已闭合的里程碑：

- kernel 配置冻结：token-major、`block_size=32`、`num_warps=2`、`num_stages=None`。
- Paged KV request lifecycle、block allocate/free/reuse、容量预检和 invariant validation。
- fused CUDA append、动态 DecodeEngine workload 和阶段 profiler 归因。
- R1 block-aware scheduler：容量安全、FIFO + aging、公平 runnable subset 和 stale decision 拒绝。
- R2 multi-layer token transaction：共享位置、单次 seq_len commit 和异常 rollback。
- RTX 5070 最终回归：`337 passed, 25 subtests passed`，无 skipped 或 failure。
- R2 正式矩阵：144 行全部通过严格校验，complete-token p50/p90/TPS 几何平均为 `1.2101x/1.3826x/1.2800x`。

当前版本仍为 `0.0.0`。这表示 release gate 尚未全部关闭，不表示 R1/R2 功能未完成。

## 当前目标：公开仓库整理

状态：进行中。

目标是让仓库本身可以独立说明问题、架构、实现边界和证据，而不依赖开发过程中的上下文。

完成条件：

- README 以系统能力和可复现实验为主，不再充当阶段日记。
- 设计、性能、兼容性、复现和历史记录有稳定导航。
- 当前状态与历史记录分离，已完成项目不再被写成未来计划。
- 本地 Markdown 链接、Python 语法和 release artifacts 有自动检查。
- GitHub CI 能在无 GPU 环境执行 dependency-free 质量门禁。

## 最终目标：v0.1.0 Release Gate

状态：待作品集整理完成后执行。

执行顺序：

1. 在全新 WSL virtualenv 中执行 editable install。
2. 运行 dependency-free 检查、CPU/reference suite、完整 RTX suite 和 release quick workload。
3. 保存环境、commit、命令与输出，确认公开数字可追溯。
4. 将 `pyproject.toml` 与 `flashdec.__version__` 同步升级为 `0.1.0`。
5. 更新 Changelog，创建并验证 `v0.1.0` tag。

版本升级和 tag 只能发生在 clean-install 证据通过之后。

## Release 后的选择性扩展

R3 与 R4 都不影响当前核心项目完成度，release 后只选择一条：

- **R3 Shared Prefix Blocks**：研究 immutable full-block 共享、refcount、回收和显存节省。
- **R4 公开基线**：选择固定版本的 FlashInfer 或 vLLM，只对齐共同支持的 shape、layout 和计时边界。

不在范围内：HTTP 服务、tokenizer、sampling、完整模型 forward、swap/offload、TP/PP 和多机执行。

## 日常质量检查

```bash
python scripts/check_docs.py
python scripts/check_release.py --require-evidence
python -m compileall -q flashdec tests benchmarks scripts
python -m pytest -q -ra
git diff --check
```

Clean install、正式 workload 和 tag 命令见[复现指南](reproducibility.md)。
