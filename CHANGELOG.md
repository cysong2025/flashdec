# Changelog

FlashDec `0.0.0` 是研究原型版本。本文件只记录版本级能力变化；实验配置、精确数字、负结果与适用边界分别由[性能报告](docs/performance_report.md)和[结果索引](benchmarks/results/README.md)维护。

## [Unreleased]

### Runtime 与 kernel

- 增加 PyTorch dense/paged decode reference，以及支持 FP16/BF16、MHA/GQA/MQA、变长 context 和非连续 physical blocks 的 Triton paged-decode kernel。
- 实现 request-scoped `PagedKVCache`：block allocate/free/reuse、capacity atomicity、committed `seq_len`、fragmentation/utilization metrics 和 invariant validation。
- 实现 PyTorch、独立 CUDA 与 fused CUDA 的 RoPE + paged KV append 路径；GPU Engine 默认使用 fused append，reference API 保持 PyTorch 语义基线。
- 实现 `DecodeEngine` 的 waiting/active/finished/cancelled lifecycle、动态 active batch、显式 backpressure，以及 caller-provided Q/K/V 的 single-token step。
- 实现 block-aware scheduler：lifetime block commitment、FIFO + aging、bounded runnable subset，以及携带原始 snapshot/config 的 decision；Engine 在 lifecycle/cache mutation 前重建权威 snapshot、重跑 canonical policy 并原子拒绝 stale、错配或伪造结果。`apply_scheduler_decision()` 现在要求显式传入生成 decision 的 `scheduler` 与 `snapshot`。
- 实现 multi-layer token transaction：跨 layer 共享写入位置、顺序执行、单次 `seq_len` commit、batch abort 和失败回滚；commit/abort 立即释放 Cache 内部完整 transaction state，只以容量 256 的轻量终态 tombstone 保留近期重复操作诊断。
- 实现 immutable full-block shared prefix：request refcount、private tail、inactive residency/LRU、terminal cleanup，以及 scheduler 的 shared/private capacity accounting。
- 实现 Cache-owned trusted transaction provenance，避免已验证位置在每个 layer 重复执行 device reduction 与 host scalar extraction。
- 实现 integrated scheduled multi-layer workload，覆盖 dynamic arrival、mixed prefix、caller-supplied context、rollback、block reuse 和 terminal zero-used cleanup。

### 实验与证据

- 增加 paged-decode shape、warp、block size、KV layout 和 staging 的受控实验；通用配置选择为 token-major、`block_size=32`、`num_warps=2`、implicit `num_stages`。
- 增加 complete-step dynamic workload 与 profiler attribution，区分 append-only、kernel-only、完整 Engine wall-clock 和 instrumented profiling。
- 增加 scheduler progress、multi-layer transaction、shared-prefix capacity、trusted transaction 和 integrated lifecycle 的严格 multi-trial validators。
- 保留 persistent-metadata candidate 的预注册负结果；该实现未替换 materialized metadata 默认路径。
- 增加固定 `flashinfer-python==0.6.15.post1` 的 FlashInfer paged-decode kernel baseline，并用 [`constraints/flashinfer-cu128.txt`](constraints/flashinfer-cu128.txt) 固定已验证 cu128/SM120a 环境。
- Canonical evidence 采用审核后的 Markdown summaries；原始 CSV、log、quick 和 profiler 产物留在 Git 之外。

### Repository surface

- 增加公开 API 文档、研究问题导航、兼容性矩阵、复现指南、贡献指南、安全政策、行为准则、支持说明和引用元数据。
- 增加 GitHub Actions 的 dependency-free repository checks、Python 3.10 compatibility checks、Markdown/HTML resource validation，以及 repository/evidence integrity checker。
- 增加 `scripts/run_validation.py`，用于分层 CUDA/tracked-clean 预检、dry-run、产物校验和仓库外结果导出。
- 增加由 canonical summaries 派生的 light/dark 研究结果图；生成器验证 commit、matrix、trial、ratio direction 和负结果后再确定性输出 SVG。
- 文档结构改为“研究问题 → 系统设计 → 实验方法 → 可复现证据”，公开树只保留可长期维护的设计、实验与复现资料。
