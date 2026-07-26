# Support

FlashDec is a research prototype rather than a production serving framework. Support is provided on a best-effort basis and is limited to the documented project scope.

## Before opening an issue

1. Read the [README](README.md), [documentation index](docs/INDEX.md), and [supported scope](docs/DELIVERY_STATUS.md).
2. Check the [compatibility notes](docs/compatibility.md) and [reproduction guide](docs/reproducibility.md).
3. Search existing issues for the same symptom or question.

## Where to ask

- Correctness, crashes, or invariant violations: use the **Correctness / regression** issue form.
- Usage or environment questions: use the **Usage / environment question** issue form.
- Scoped implementation or evidence proposals: use the **Scoped change / evidence proposal** form.
- Security vulnerabilities: follow [SECURITY.md](SECURITY.md) and do not disclose details publicly.

Performance reports must include the commit, device, software versions, shape and dtype, seed, warmup/repeat/trial counts, timing boundary, absolute measurements, and comparison direction. Results from different hardware, shapes, layouts, or timing scopes are not assumed to be comparable.

The project does not currently promise support for complete model serving, multi-GPU execution, HTTP/RPC APIs, tokenizer/sampling stacks, or production deployment.
