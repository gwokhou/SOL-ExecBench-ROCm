"""Opt-in stage memory observations, with explicit scope and missing values."""

import os
import threading
from pathlib import Path

_SAMPLE_SECONDS = 0.01
_STATM = Path("/proc/self/statm")


class StageMemoryObservation:
    """Sample process RSS and, when already initialized, allocator peak bytes."""

    def __init__(self, *, observe_device: bool) -> None:
        """Prepare a diagnostic sampler without initializing any GPU runtime."""
        self.stop = threading.Event()
        self.peak: int | None = None
        self.rss_error: str | None = None
        self.device_index: int | None = None
        self.device_error: str | None = None
        self.observe_device = observe_device
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        """Begin RSS sampling and optionally reset an owned device allocator peak."""
        self._read_rss()
        self.thread.start()
        if self.observe_device:
            import torch

            if torch.cuda.is_initialized():
                self.device_index = torch.cuda.current_device()
                torch.cuda.reset_peak_memory_stats(self.device_index)
            else:
                self.device_error = "CUDA/HIP runtime is not initialized"
        else:
            self.device_error = "CPU formal stage has no owned device interval"

    def finish(self) -> dict[str, object]:
        """Return sampled RSS (not an exact peak) and allocator-only GPU peak."""
        self.stop.set()
        self.thread.join(timeout=1)
        self._read_rss()
        device_peak = None
        if self.device_index is not None:
            import torch

            device_peak = torch.cuda.max_memory_allocated(self.device_index)
        return {
            "host_rss_sampled_peak_bytes": self.peak,
            "host_rss_sampling_seconds": _SAMPLE_SECONDS,
            "host_rss_scope": "current worker process; sampled lower bound, excluding mapper children",
            "host_rss_reason": self.rss_error,
            "device_peak_bytes": device_peak,
            "device_peak_scope": "current device PyTorch allocator only; excludes direct driver allocations",
            "device_peak_reason": self.device_error,
        }

    def _sample(self) -> None:
        while not self.stop.wait(_SAMPLE_SECONDS):
            self._read_rss()

    def _read_rss(self) -> None:
        try:
            resident = int(_STATM.read_text().split()[1]) * os.sysconf(
                "SC_PAGE_SIZE"
            )
            self.peak = max(self.peak or 0, resident)
        except (OSError, ValueError, IndexError) as exc:
            self.rss_error = str(exc)
