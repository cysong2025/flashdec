# Public Repository Readiness

This checklist separates **making the source repository public** from publishing a stable `v0.1.0` software release. FlashDec may be public as an explicitly pre-release research prototype while fresh-environment certification, versioning, and a release tag remain incomplete.

## Visibility blockers

| Gate | Required evidence | Current state |
| --- | --- | --- |
| License and provenance | Owner-selected license; package metadata aligned; no incompatible third-party code | Apache-2.0 selected; root text, package metadata, citation metadata, and README aligned; source provenance scan found no embedded third-party copyright block |
| Current-tree secret scan | No credentials, private keys, tokens, personal emails, or private machine details | Gitleaks 8.30.1 worktree scan passed; personal paths/LAN host normalized |
| History review | No obvious credentials; commit authors use public-safe addresses; no large hidden artifacts | Gitleaks scanned 108 commits with no leaks; 108/108 commits use GitHub noreply email; largest historical blob is about 68 KiB |
| Public-facing status | README/current docs consistently say public research preview | Public-preview wording complete; final unauthenticated verification pending |
| Governance | Security policy, conduct policy, support routing, contribution guide, issue/PR templates, ownership | Added; docs/release/YAML validation passed |
| Evidence integrity | Charts derive from canonical summaries and retain negative results and scope boundaries | Processed snapshot, source anchors, generator, two SVG themes and tests validated |
| GitHub metadata | Description, homepage, topics, community profile | About, topics, and merge settings configured and verified; community profile pending final content push |
| Visibility change | Explicit final owner confirmation after reviewing consequences | Confirmed 2026-07-26; switch waits for a green clean-commit CI run |

## Public research-preview gates

- [x] Root license and `pyproject.toml` license metadata agree.
- [ ] `README.md`, current design/status docs, issue forms, and badges use public-preview wording.
- [x] `SECURITY.md`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`, `CITATION.cff`, `CONTRIBUTING.md`, CODEOWNERS, and issue/PR templates pass repository checks.
- [x] Current tracked files and full history pass the documented secret-pattern review.
- [x] No raw logs, profiles, credentials, local backups, native binaries, or large generated artifacts are tracked.
- [x] Experiment overview data identify their canonical summary, GPU, commit, trial scope, ratio direction, and negative-result boundary.
- [ ] Repository checks pass on a clean commit.
- [x] Existing Actions history/logs have been reviewed because they become public with the repository.
- [x] About description, homepage, topics, and merge settings are configured.

## Post-visibility GitHub settings

Immediately after the visibility change, verify and record the following settings:

1. require the `repository-checks` and `python-310-compat` job statuses on `main` (the workflow display name is `repository checks`);
2. block force pushes and branch deletion;
3. confirm Dependabot alerts/security updates remain enabled (configured before visibility), then enable dependency graph, secret scanning, push protection, and private vulnerability reporting when available;
4. enable automatic head-branch deletion after merge;
5. verify the issue chooser, citation panel, community profile, and security policy links as an unauthenticated visitor.

## Stable `v0.1.0` release gates

The following are deliberately separate from public source visibility:

- fresh clone and isolated environment installation;
- dependency-resolution and CUDA-extension build validation;
- full CPU/GPU correctness and evidence replay on the supported environment;
- `0.0.0 -> 0.1.0` version update;
- signed/annotated tag and GitHub Release;
- release artifact checksums and, if published, sanitized canonical CSV assets.

Until those gates pass, public documentation must call FlashDec a pre-release research prototype and must not promise a stable API, PyPI package, or clean installation on arbitrary systems.
