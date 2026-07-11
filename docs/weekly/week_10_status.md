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

阶段二已完成。RTX 5070 correctness、quick 和 full sweep 均已执行；没有显式 `num_stages` 达到默认值替换门槛，因此保留 Triton implicit default 并冻结 kernel 配置。

correctness：

```text
88 passed in 5.00s
```

full sweep 共 30 条记录，对应 3 个代表场景 × 2 种 dtype × 5 个 staging 配置，全部为 `validated=True`。

## Full sweep 结果

| num_stages | 六场景 p50 几何平均加速 | p50 胜场 | 最大 p50 回退 | FP16 几何平均 | BF16 几何平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0002x | 3/6 | 1.10% | 1.0025x | 0.9979x |
| 2 | 1.0039x | 4/6 | 0.17% | 1.0074x | 1.0005x |
| 3 | 1.0038x | 3/6 | 0.43% | 1.0100x | 0.9978x |
| 4 | 1.0036x | 4/6 | 0.55% | 1.0096x | 0.9975x |

stage 2 是六场景几何平均最优候选，但仅约快 0.39%，远低于 5% 门槛。stage 3/4 在 FP16 上约快 1%，BF16 却整体略慢；它们在 BF16 medium 上的最大 p90 回退分别达到约 13.38% 和 18.93%。

FP16 large 的 implicit default p90 为 `1.178848 ms`，而显式候选约为 `0.921-0.926 ms`；但 default p50 仅比候选慢约 1.8%-2.1%，说明该差异主要来自尾部抖动，不能据此修改默认配置。

## 最终决策

- 保留 token-major、`block_size=32`、`num_warps=2`、`num_stages=None`。
- 不选择显式 stage 2，因为 0.39% 的总体收益不足以覆盖测量噪声、维护成本和未来环境变化。
- 不再为 `num_stages` 做候选 Profiler，也不增加 dtype/shape dispatch。
- 这是一项有价值的负结果：显式 staging 没有给当前 memory-bound paged decode kernel 带来可观、跨 dtype 的稳定收益。
- kernel 参数调优到此冻结，开始 PagedKVCache v2。

详细摘要见 `benchmarks/results/week10_num_stages_summary.md`。

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

实际结果没有候选满足条件，因此已记录负结果并保留 `num_stages=None`。本轮后冻结 kernel 配置并进入 PagedKVCache v2。

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

配置已经冻结。PagedKVCache v2 第一批代码已经实现：

- `finish_request()` / `cancel_request()`。
- physical block release 和 reuse-priority free list。
- active/finished/cancelled request state query。
- batch append 容量 preflight，无 partial request mutation。
- block utilization、internal fragmentation、allocation/free/reuse/lifecycle metrics。
- allocator invariant validator 和 request churn 测试。
- 显式限制单 layer runtime；多 layer execution 不在当前 `v0.1.0` 范围。

当前 Codex macOS 环境没有 torch/pytest/CUDA，因此只完成了 compileall 和静态检查，不能写入 GPU 通过结论。

RTX 5070 focused 验证：

```bash
python -m pytest -vv \
  tests/test_paged_cache.py \
  tests/test_paged_decode.py \
  tests/test_public_api.py
```

focused 通过后执行完整回归：

```bash
python -m pytest -vv
```

验证通过并记录结果后，进入 RoPE + KV append 数据路径。这一步把项目从单 kernel 推进为具有 request lifecycle 和 physical memory ownership 的 decode runtime。
