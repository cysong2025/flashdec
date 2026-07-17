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

## 当前目标：R3 Shared Prefix Blocks

状态：R3 设计与实现进行中；仓库按所有者要求继续保持 private。

R3 研究重复 system prompt / 固定上下文的 immutable full-block 共享，目标是减少重复 KV physical blocks，同时保持 request lifecycle、transaction rollback 和容量预检的正确性。

分阶段目标：

- R3-A：完成 prefix 注册、挂载、引用计数、inactive LRU、回收和 CPU 不变量测试。
- R3-B：让 DecodeEngine 与 block-aware scheduler 正确区分 shared residency 和 request-private commitment。
- R3-C：完成 hit-rate benchmark、RTX correctness、显存节省证据和结果归档。

明确边界：

- 只共享调用方已经构建的 immutable full blocks；tail block 保持 request-private。
- 不实现 tokenizer、模型 prefill、sampling、HTTP server 或分布式执行。
- R3-A 不绕过 scheduler 的 lifetime commitment；scheduler-managed prefix 在 R3-B 接入。

所有权与验收细节见[Shared Prefix Blocks 设计](design_shared_prefix_blocks.md)。

## 最终目标：v0.1.0 Release Gate

状态：待 R3 优化与证据闭合后执行。

执行顺序：

1. 在全新 WSL virtualenv 中执行 editable install。
2. 运行 dependency-free 检查、CPU/reference suite、完整 RTX suite 和 release quick workload。
3. 保存环境、commit、命令与输出，确认公开数字可追溯。
4. 将 `pyproject.toml` 与 `flashdec.__version__` 同步升级为 `0.1.0`。
5. 更新 Changelog，创建并验证 `v0.1.0` tag。

版本升级和 tag 只能发生在 clean-install 证据通过之后。

## 公开发布设置

仓库当前继续保持 private。可见性和许可证不再阻塞 R3；只有准备公开时才重新确认：

- 是否将 GitHub repository 改为 public；
- 采用 MIT、Apache-2.0，或继续保留无开源授权状态；
- GitHub `quality` workflow 和公开链接是否正常。

R4 FlashInfer/vLLM 有限公开对比仍是选择性扩展，不阻塞 R3 或 v0.1.0。

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
