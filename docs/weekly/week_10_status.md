# Week 10 状态记录

## 本周主题

冻结 paged decode kernel 配置，并把项目重心转向 Paged KV runtime 生命周期。

## 已完成的代码

- `paged_decode_attention()` 新增可选 `num_stages`。
- `num_stages=None` 不向 Triton launch 显式传参，保留当前 implicit default 作为真实 baseline。
- 新增 `benchmarks/run_num_stages_sweep.py`，只对 `default/1/2/3/4` 做有边界的 sweep。
- 固定默认决策条件：token-major、`block_size=32`、`num_warps=2`，场景为 medium、large、large-batch。
- Profiler 记录 `num_stages`，文件名和 summary 可区分 implicit default 与显式候选。
- 新增参数解析、非法输入和显式 staging correctness 测试。

## 当前状态

代码已完成，尚未在 RTX 5070 上验证。当前不能宣称某个显式 `num_stages` 更快，也不能修改默认值。

本地 macOS 只做静态验证；CUDA correctness 和性能实验必须在 RTX 5070 WSL 环境完成。

## RTX 5070 执行顺序

### 1. 拉取并进入环境

```bash
cd ~/work/flashdec
git pull origin main
source .venv/bin/activate
```

### 2. 正确性回归

```bash
python -m pytest -vv \
  tests/test_num_stages_sweep.py \
  tests/test_profile_paged_decode.py \
  tests/test_paged_decode.py \
  tests/test_public_api.py
```

### 3. Quick sweep

```bash
python benchmarks/run_num_stages_sweep.py \
  --cases medium \
  --dtype both \
  --num-stages default 1 2 3 4 \
  --kv-layout token_major \
  --block-size 32 \
  --num-warps 2 \
  --warmup 3 \
  --repeat 10 \
  --output benchmarks/results/week10_num_stages_quick.csv \
  | tee benchmarks/results/week10_num_stages_quick.log
```

### 4. Full sweep

```bash
python benchmarks/run_num_stages_sweep.py \
  --cases medium large large_batch \
  --dtype both \
  --num-stages default 1 2 3 4 \
  --kv-layout token_major \
  --block-size 32 \
  --num-warps 2 \
  --warmup 5 \
  --repeat 30 \
  --output benchmarks/results/week10_num_stages.csv \
  | tee benchmarks/results/week10_num_stages.log
```

不要加 `--skip-validate`。先完成 quick，再运行 full；Profiler 只在 full 结果选出候选后用于 baseline/候选解释，不用于搜索参数。

## 默认配置决策规则

以 CUDA event `p50` 为主、`p90` 为辅。只有候选同时满足以下条件才替换 implicit default：

1. 所有行均完成 reference validation。
2. 相对 default 的 p50 几何平均提升超过 5%。
3. medium、large、large-batch 中任一主要 shape 的 p50 回退不超过 5%。
4. FP16 与 BF16 的方向基本一致。

若没有候选满足条件，记录负结果并保留 `num_stages=None`。无论结果如何，本轮后冻结 kernel 配置并进入 PagedKVCache v2。

## 结果同步

WSL 先复制到 Windows 用户目录：

```bash
mkdir -p /mnt/c/Users/user/flashdec_results
cp \
  benchmarks/results/week10_num_stages_quick.csv \
  benchmarks/results/week10_num_stages_quick.log \
  benchmarks/results/week10_num_stages.csv \
  benchmarks/results/week10_num_stages.log \
  /mnt/c/Users/user/flashdec_results/
```

然后在 Mac 执行：

```bash
cd /Users/songchuangye/Documents/ai_infra
scp \
  user@192.168.71.95:flashdec_results/week10_num_stages_quick.csv \
  user@192.168.71.95:flashdec_results/week10_num_stages_quick.log \
  user@192.168.71.95:flashdec_results/week10_num_stages.csv \
  user@192.168.71.95:flashdec_results/week10_num_stages.log \
  benchmarks/results/
```

Windows OpenSSH 环境不依赖 `rsync`，继续沿用 `cp + scp` 流程。

## 下一阶段入口

结果分析并冻结配置后，开始 PagedKVCache v2：优先实现 `finish_request()`、`cancel_request()`、physical block free/reuse、无 partial mutation 的容量失败语义和 request churn 测试。这是从单 kernel 走向 decode runtime 的主线。
