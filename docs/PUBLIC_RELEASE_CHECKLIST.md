# Public Repository Readiness

This checklist separates **making the source repository public** from publishing a stable `v0.1.0` software release. FlashDec may be public as an explicitly pre-release research prototype while fresh-environment certification, versioning, and a release tag remain incomplete.

## Visibility blockers

| Gate | Required evidence | Current state |
| --- | --- | --- |
| License and provenance | Owner-selected license; package metadata aligned; no incompatible third-party code | Apache-2.0 selected; root text, package metadata, citation metadata, and README aligned; source provenance scan found no embedded third-party copyright block |
| Current-tree secret scan | No credentials, private keys, tokens, personal emails, or private machine details | Gitleaks 8.30.1 worktree scan passed; personal paths/LAN host normalized |
| History review | No obvious credentials; commit authors use public-safe addresses; no large hidden artifacts | Gitleaks scanned 109 commits with no leaks; 109/109 commits use GitHub noreply email; largest historical blob is about 68 KiB |
| Public-facing status | README/current docs consistently say public research preview | Public-preview wording complete; repository page, README, license, security policy, charts, data snapshot, and Actions badge respond anonymously |
| Governance | Security policy, conduct policy, support routing, contribution guide, issue/PR templates, ownership | Added; docs/release/YAML validation passed |
| Evidence integrity | Charts derive from canonical summaries and retain negative results and scope boundaries | Processed snapshot, source anchors, generator, two SVG themes and tests validated |
| GitHub metadata | Description, homepage, topics, community profile | About, homepage, 12 topics, merge settings, and 100% community profile verified |
| Visibility change | Explicit final owner confirmation after reviewing consequences | Public since 2026-07-26; switched only after clean commit `16f655f` passed both repository-check jobs |

## Public research-preview gates

- [x] Root license and `pyproject.toml` license metadata agree.
- [x] `README.md`, current design/status docs, issue forms, and badges use public-preview wording.
- [x] `SECURITY.md`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`, `CITATION.cff`, `CONTRIBUTING.md`, CODEOWNERS, and issue/PR templates pass repository checks.
- [x] Current tracked files and full history pass the documented secret-pattern review.
- [x] No raw logs, profiles, credentials, local backups, native binaries, or large generated artifacts are tracked.
- [x] Experiment overview data identify their canonical summary, GPU, commit, trial scope, ratio direction, and negative-result boundary.
- [x] Repository checks pass on a clean commit.
- [x] Existing Actions history/logs have been reviewed because they become public with the repository.
- [x] About description, homepage, topics, and merge settings are configured.

## Post-visibility GitHub settings

Verified on 2026-07-26:

- [x] `main` requires the `repository-checks` and `python-310-compat` statuses, an up-to-date branch, linear history, and resolved conversations.
- [x] Force pushes and branch deletion are blocked; administrators retain an explicit solo-maintainer bypass.
- [x] Dependency graph, Dependabot alerts/security updates, secret scanning, push protection, and private vulnerability reporting are enabled.
- [x] Actions are limited to GitHub-owned actions with full-SHA pinning; CodeQL default setup passed for Actions, C/C++, and Python.
- [x] Head branches are deleted automatically after merge.
- [x] The issue chooser route, citation panel, 100% community profile, license, code of conduct, and security-policy surfaces were verified after publication.

## Stable `v0.1.0` release gates

The following are deliberately separate from public source visibility:

- fresh clone and isolated environment installation;
- dependency-resolution and CUDA-extension build validation;
- full CPU/GPU correctness and evidence replay on the supported environment;
- `0.0.0 -> 0.1.0` version update;
- signed/annotated tag and GitHub Release;
- release artifact checksums and, if published, sanitized canonical CSV assets.

Until those gates pass, public documentation must call FlashDec a pre-release research prototype and must not promise a stable API, PyPI package, or clean installation on arbitrary systems.
