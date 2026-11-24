# Iteration 3 Engineering Report

## Status: Partially implemented; acceptance criteria remain open

Baseline: `b4a8c35e1392341356e4b692e26681803c34b7cc` (working tree was clean at inspection). The repository remains on that commit with uncommitted changes; there is no final commit SHA yet.

## Implemented in this pass

- Removed the legacy static `API_KEY` authentication fallback. It can no longer create an identity or grant roles.
- Made unauthenticated access opt-in (`DEV_AUTH_ENABLED`, default false), fixed to `DEV_TENANT_ID`, and limited it to reader permissions. Staging and production reject this mode.
- Added credential-provisioning checks that reject tenant-admin cross-tenant requests, unknown roles, and permissions beyond the caller's permission set. The `platform_admin` role is explicitly distinct from tenant `admin`.
- Production/staging startup and readiness now require a registered credential.
- Wired `recover_and_cleanup()` into FastAPI startup.
- Removed the fabricated structural-to-semantic entailment score. Structural checks report `semantic_entailment_score: null` and `semantic_entailment_status: not_evaluated` unless a real verifier supplies a score.
- Renamed the offline overlap scorer to `LexicalOverlapReranker` and updated the E5 description so it is not represented as a trained cross-encoder.

## Verification evidence

- `./.venv-linux/bin/python -m compileall -q src tests`: passed.
- `./.venv-linux/bin/python -m pytest tests/unit/test_experiments.py tests/unit/test_citations_context.py -q`: **25 passed**.
- `./.venv-linux/bin/python -m pytest tests/unit/test_security_api.py -k 'authentication or authorization_helpers or global_api_key or development_bypass' -q`: **5 passed, 10 deselected**.
- `git diff --check`: passed after removing the line-ending-sensitive `.env.example` edit.
- The full baseline command could not start as `pytest` and `python` are absent from PATH. It was retried as `./.venv-linux/bin/python -m pytest -q`, but did not complete within the observed run.
- Focused FastAPI test-client requests did not complete in this environment. HTTP authorization behavior is therefore not marked verified.
- PostgreSQL was not provisioned or tested. No PostgreSQL support claim is made here.

## Acceptance status

| Area | Status | Evidence / remaining work |
|---|---|---|
| Legacy global key escalation | PARTIALLY VERIFIED | Fallback removed and unit attack test added; HTTP test did not complete. |
| Fixed tenant and dev reader access | PARTIALLY VERIFIED | Direct authentication test added; full HTTP checks pending. |
| Credential provisioning scope | PARTIALLY VERIFIED | Route checks added; HTTP attack test did not complete. |
| Durable credentials / shared revocation | NOT VERIFIED | Registry still uses in-memory dictionaries. |
| Real PostgreSQL document registry | NOT VERIFIED | `SQLDocumentRegistry` still opens SQLite connections even when passed `postgres_uri`. |
| Startup ingestion recovery | PARTIALLY VERIFIED | Lifespan calls recovery; failure-injection/startup tests remain. |
| Semantic grounding score integrity | VERIFIED | Structural validation no longer manufactures entailment values. |
| Real cross-encoder comparison | NOT VERIFIED | Only lexical baseline labeling changed; model adapter and paired measurements remain. |
| Evaluation provenance and leakage prevention | NOT VERIFIED | Dataset/adjudication changes remain. |
| Benchmarks and CI matrix | NOT VERIFIED | No updated benchmark artifacts or PostgreSQL CI service. |
| Full test suite | NOT VERIFIED | No completed full-suite result in this environment. |

## Remaining implementation required

Credential persistence and transactional multi-worker rotation, an actual PostgreSQL adapter and migration, verified failure-injection recovery tests, the real cross-encoder adapter and paired experiments, evaluation split/provenance enforcement, benchmark regeneration, CI PostgreSQL service coverage, telemetry concurrency fixes, and a complete test run remain necessary before Iteration 3 can be declared complete.
