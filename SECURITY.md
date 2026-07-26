# Security Policy

FlashDec is a research prototype that combines Python, Triton, and native CUDA code. Security reports are welcome, especially when they involve memory safety, malformed tensor metadata, state corruption, unsafe native-extension behavior, or dependency vulnerabilities.

## Supported versions

There is no stable release line yet. Only the latest commit on `main` is considered for security fixes; historical commits and benchmark evidence commits are retained for reproducibility but are not supported release branches.

## Reporting a vulnerability

Do **not** open a public issue with exploit details, credentials, private paths, or sensitive logs.

Use GitHub's private vulnerability reporting flow from the repository **Security** tab. If that option is temporarily unavailable, open a minimal public issue asking the maintainer to enable a confidential channel; do not include vulnerability details, logs, or a proof of concept in that issue.

Please include, when available:

- the affected commit and execution path;
- Python, PyTorch, Triton, CUDA, compiler, and GPU versions;
- a minimal trigger or proof of concept;
- the expected and observed behavior;
- the impact and any known workaround.

The maintainer will acknowledge and assess reports on a best-effort basis, coordinate a fix and advisory when appropriate, and ask the reporter to delay public disclosure until users have a reasonable opportunity to update.

## Non-security reports

Correctness regressions, performance questions, unsupported environments, and documentation problems should use the normal [issue forms](https://github.com/cysong2025/flashdec/issues/new/choose). See [SUPPORT.md](SUPPORT.md) for routing guidance.
