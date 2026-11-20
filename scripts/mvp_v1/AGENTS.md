# MVP v1 Cloud Runner Rules

- **Python 3.12 Runtime Compatibility**: Cloud containers often run Python 3.12 (`cp312` PyTorch wheels) while local development targets Python 3.14. All imports under `scripts/mvp_v1/` must remain compatible via `py312_compat.py` (PEP 758 `except` parenthesization + `CO_FUTURE_ANNOTATIONS`), and remote deployments must clear stale `__pycache__` directories.
- **Pre-Flight Twin Gate**: Before booting or uploading to a cloud GPU instance, run `uv run python -m scripts.mvp_v1.cloud_preflight twin --target <TARGET>` locally to validate uncached source loading and smoke execution.
- **Single-Pass Connectivity Probe**: Before running remote `ssh`, run `uv run python -m scripts.mvp_v1.cloud_preflight probe <HOST> [--proxy ...]`. On `ssh_banner_blocked_http_open`, never retry port 22; switch immediately to a high SSH port or one-shot `nohup` execution via Web Console bootstrap.
- **Hard Evidence Gate**: Never write remote results into `data/outputs/` or report benchmark numbers to the user before verifying `BROWSER_HOST_FACTS` (for browser runs) and running `uv run python -m scripts.mvp_v1.cloud_preflight verify <PATH> [--expected-sha256 ...]` on downloaded artifacts.
- **Internal Probe Lock Contract**: Scripts using `native_gpu_execution()` (such as `run_cdna3_diagnostic_calibration.py`) require both `GPU_LOCK_DIR` and a full AMD-SMI `GPU_LOCK_UUID`.
