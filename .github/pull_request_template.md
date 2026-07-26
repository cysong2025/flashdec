## 🎯 Summary

<!-- What changed, why it changed, and the user/developer impact. -->

## 🧭 Scope and boundaries

- Affected area:
- Explicitly unchanged:
- Related issue/design/evidence:

## 🔒 Correctness and state semantics

<!-- Describe affected invariants, ownership/lifecycle, failure atomicity, rollback, and API compatibility. Write N/A with a reason when not applicable. -->

## ✅ Validation

- [ ] `python scripts/check_docs.py`
- [ ] `python scripts/check_release.py --require-evidence`
- [ ] `python -m compileall -q flashdec tests benchmarks scripts`
- [ ] Relevant targeted tests
- [ ] Full pytest, or a documented reason it is not applicable/available
- [ ] `git diff --check`

Exact commands and results:

```text
<commands and concise output>
```

## 📊 GPU / performance evidence

- [ ] Not applicable; this change makes no GPU/performance claim.
- [ ] Applicable; the evidence below records commit/clean tree, GPU, Python/PyTorch/Triton/CUDA, shape/dtype, seed, warmup/repeat/trials, timing boundary, absolute latency, ratios/ranges, and negative results.

```text
<evidence or N/A reason>
```

## 📚 Documentation and evidence impact

- [ ] Relevant design/API/compatibility/reproduction docs are updated, or N/A is explained.
- [ ] Canonical Markdown summary is updated when evidence changes; raw CSV/log/trace files remain local.
- [ ] `CHANGELOG.md` is updated, or N/A is explained.

## ⚠️ Risk and rollback

<!-- State the main regression risk, detection signal, and safe rollback/default path. -->

## Final checklist

- [ ] The diff is focused and contains no unrelated files.
- [ ] Reference/parity and applicable error paths are covered.
- [ ] Performance and release claims stay within the measured scope.
- [ ] No version, tag, visibility, license, or release change is included without separate owner authorization.
