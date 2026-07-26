# Trusted Transaction Validation Summary

## Validation

- Input: `benchmarks/results/r4_fused_transaction_fast_path_trials5.csv`.
- Rows: 160; paired trials: 80.
- Device: NVIDIA GeForce RTX 5070.
- PyTorch/CUDA: 2.11.0+cu128 / 12.8.
- Git commit: `4018449`.
- Checked and trusted paths used identical fused CUDA/Triton math; only the Cache-owned validation boundary differed.
- Matrix, seed/order, pure-wall timing scope, block accounting, Engine/transaction trajectory, CPU profiler ranges, bounded capture attempts, and invariants were validated.
- Profiler capture attempts: 160 total; extra retries: 0; maximum per row: 1.
- Complete-token latency is pure synchronized wall time with no CUDA events in its interval; the separate profiler is CPU-only and excludes device attribution.

Ratios above 1 favor trusted dispatch. Latency and append-CPU ratios are checked/trusted; throughput is trusted/checked.

## Cross-trial Cases

| dtype | case | p50 median [min,max] | p90 | p99 [min,max] | TPS | append CPU | direction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| bfloat16 | l2_b16_c1024 | 1.5178x [1.4484,1.6020] | 1.5455x | 1.5711x [1.3447,7.8006] | 1.5367x | 2.3101x | trusted_faster |
| bfloat16 | l2_b16_c128 | 1.6066x [1.5383,1.7392] | 1.5771x | 1.5633x [1.4390,2.4653] | 1.5947x | 2.4667x | trusted_faster |
| bfloat16 | l2_b4_c1024 | 1.7300x [1.6777,1.8181] | 1.7657x | 1.5740x [1.0901,1.8330] | 1.7388x | 2.5084x | trusted_faster |
| bfloat16 | l2_b4_c128 | 1.7622x [1.3229,1.9183] | 1.6979x | 1.6004x [0.2727,2.1356] | 1.7773x | 2.4903x | trusted_faster |
| bfloat16 | l4_b16_c1024 | 1.6694x [1.6088,1.6992] | 1.6280x | 1.6431x [1.4995,3.1883] | 1.6789x | 2.2066x | trusted_faster |
| bfloat16 | l4_b16_c128 | 1.8866x [1.8130,1.9061] | 1.8834x | 1.8446x [1.6282,5.1466] | 1.8822x | 2.4384x | trusted_faster |
| bfloat16 | l4_b4_c1024 | 1.8927x [1.6041,1.9661] | 1.9398x | 1.8525x [0.6340,8.3030] | 1.9056x | 2.2584x | trusted_faster |
| bfloat16 | l4_b4_c128 | 2.0666x [1.9811,2.1298] | 2.0646x | 1.9644x [1.9204,2.1349] | 2.1017x | 2.3985x | trusted_faster |
| float16 | l2_b16_c1024 | 1.4956x [1.4283,1.6091] | 1.4090x | 1.7032x [0.9113,2.1481] | 1.4462x | 2.1877x | trusted_faster |
| float16 | l2_b16_c128 | 1.5865x [1.2567,1.6543] | 1.5373x | 1.5123x [0.4335,1.6638] | 1.5728x | 2.4846x | trusted_faster |
| float16 | l2_b4_c1024 | 1.7974x [1.6549,1.9965] | 1.7746x | 1.6186x [1.1271,2.6283] | 1.7704x | 2.3786x | trusted_faster |
| float16 | l2_b4_c128 | 1.8888x [1.5061,1.9246] | 1.6889x | 1.6444x [0.6000,5.3844] | 1.8533x | 2.4465x | trusted_faster |
| float16 | l4_b16_c1024 | 1.5774x [1.4338,1.6424] | 1.5897x | 1.4836x [0.5450,1.5087] | 1.6069x | 2.1038x | trusted_faster |
| float16 | l4_b16_c128 | 1.8011x [1.7738,1.9794] | 1.7602x | 1.8013x [0.5485,6.9533] | 1.8223x | 2.3403x | trusted_faster |
| float16 | l4_b4_c1024 | 1.8509x [1.7976,1.9381] | 1.8734x | 1.8447x [1.1875,4.8721] | 1.8720x | 2.4517x | trusted_faster |
| float16 | l4_b4_c128 | 1.9826x [1.6910,2.1110] | 1.9338x | 1.9773x [1.6837,2.0951] | 1.9886x | 2.5933x | trusted_faster |

## Absolute Attribution Medians

| dtype | case | path | token p50 ms | begin host ms | commit host ms | append CPU ms/layer | item | local scalar | capture attempts |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bfloat16 | l2_b16_c1024 | checked | 3.159817 | 0.849849 | 0.321930 | 0.989742 | 20 | 20 | 1.0 |
| bfloat16 | l2_b16_c1024 | trusted | 2.081836 | 0.830096 | 0.319230 | 0.426868 | 0 | 0 | 1.0 |
| bfloat16 | l2_b16_c128 | checked | 2.890844 | 0.847396 | 0.189844 | 0.886363 | 20 | 20 | 1.0 |
| bfloat16 | l2_b16_c128 | trusted | 1.786299 | 0.833969 | 0.195654 | 0.359331 | 0 | 0 | 1.0 |
| bfloat16 | l2_b4_c1024 | checked | 2.440341 | 0.372518 | 0.225957 | 0.885117 | 20 | 20 | 1.0 |
| bfloat16 | l2_b4_c1024 | trusted | 1.409553 | 0.354110 | 0.233267 | 0.361111 | 0 | 0 | 1.0 |
| bfloat16 | l2_b4_c128 | checked | 2.398613 | 0.351531 | 0.180874 | 0.891985 | 20 | 20 | 1.0 |
| bfloat16 | l2_b4_c128 | trusted | 1.353858 | 0.350301 | 0.199576 | 0.362531 | 0 | 0 | 1.0 |
| bfloat16 | l4_b16_c1024 | checked | 5.260043 | 0.844294 | 0.317149 | 0.981132 | 40 | 40 | 1.0 |
| bfloat16 | l4_b16_c1024 | trusted | 3.144444 | 0.833984 | 0.320959 | 0.442254 | 0 | 0 | 1.0 |
| bfloat16 | l4_b16_c128 | checked | 4.704140 | 0.818544 | 0.189635 | 0.957081 | 40 | 40 | 1.0 |
| bfloat16 | l4_b16_c128 | trusted | 2.530855 | 0.816030 | 0.187716 | 0.367747 | 0 | 0 | 1.0 |
| bfloat16 | l4_b4_c1024 | checked | 4.514875 | 0.367348 | 0.276638 | 0.934266 | 40 | 40 | 1.0 |
| bfloat16 | l4_b4_c1024 | trusted | 2.424932 | 0.368771 | 0.285598 | 0.409743 | 0 | 0 | 1.0 |
| bfloat16 | l4_b4_c128 | checked | 4.206451 | 0.359261 | 0.183082 | 0.916610 | 40 | 40 | 1.0 |
| bfloat16 | l4_b4_c128 | trusted | 2.045852 | 0.353921 | 0.194233 | 0.359548 | 0 | 0 | 1.0 |
| float16 | l2_b16_c1024 | checked | 3.194351 | 0.843754 | 0.311229 | 0.928978 | 20 | 20 | 1.0 |
| float16 | l2_b16_c1024 | trusted | 2.135790 | 0.858804 | 0.315179 | 0.425331 | 0 | 0 | 1.0 |
| float16 | l2_b16_c128 | checked | 2.807799 | 0.815593 | 0.186466 | 0.912068 | 20 | 20 | 1.0 |
| float16 | l2_b16_c128 | trusted | 1.894243 | 0.850634 | 0.201045 | 0.370538 | 0 | 0 | 1.0 |
| float16 | l2_b4_c1024 | checked | 2.459779 | 0.354200 | 0.214366 | 0.871265 | 20 | 20 | 1.0 |
| float16 | l2_b4_c1024 | trusted | 1.341818 | 0.355050 | 0.223136 | 0.370378 | 0 | 0 | 1.0 |
| float16 | l2_b4_c128 | checked | 2.371966 | 0.354740 | 0.177035 | 0.924843 | 20 | 20 | 1.0 |
| float16 | l2_b4_c128 | trusted | 1.270756 | 0.336889 | 0.189055 | 0.387163 | 0 | 0 | 1.0 |
| float16 | l4_b16_c1024 | checked | 5.115669 | 0.817236 | 0.306997 | 0.948097 | 40 | 40 | 1.0 |
| float16 | l4_b16_c1024 | trusted | 3.243132 | 0.862286 | 0.315139 | 0.432777 | 0 | 0 | 1.0 |
| float16 | l4_b16_c128 | checked | 4.655231 | 0.836174 | 0.185905 | 0.893908 | 40 | 40 | 1.0 |
| float16 | l4_b16_c128 | trusted | 2.584643 | 0.831404 | 0.201736 | 0.383415 | 0 | 0 | 1.0 |
| float16 | l4_b4_c1024 | checked | 4.345483 | 0.349460 | 0.248347 | 0.921560 | 40 | 40 | 1.0 |
| float16 | l4_b4_c1024 | trusted | 2.332275 | 0.359350 | 0.261227 | 0.411458 | 0 | 0 | 1.0 |
| float16 | l4_b4_c128 | checked | 4.141658 | 0.340630 | 0.177535 | 0.892039 | 40 | 40 | 1.0 |
| float16 | l4_b4_c128 | trusted | 2.047147 | 0.351250 | 0.187795 | 0.343207 | 0 | 0 | 1.0 |

## Overall Geometric Mean

| metric | trusted vs checked |
| --- | ---: |
| complete-token p50 | 1.7307x |
| complete-token p90 | 1.6751x |
| complete-token p99 | 1.6944x |
| complete-token mean | 1.7131x |
| decode tokens/s | 1.7131x |
| begin host p50 | 0.9978x |
| commit host p50 | 0.9618x |
| profiler append CPU/layer | 2.3612x |

## Interpretation

- `unstable_crosses_1` means the complete-token p50 direction changes across trials; do not claim a stable win.
- The trusted path is accepted only for Cache-owned transaction metadata; public raw CUDA calls retain checked validation.
- Profiler inclusive CPU totals, scalar-extraction counts, and bounded recapture attempts explain the removed synchronization boundary but are not release latency; device attribution is intentionally excluded.
- p99 must be reported with its full range, and a ratio below 1 remains a negative result.
