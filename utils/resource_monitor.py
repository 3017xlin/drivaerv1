"""ResourceMonitor — background CPU/GPU sampler with phase markers.

Adapted from tandemv1; emits a 4-panel PNG (CPU %, CPU RSS, GPU %, GPU GiB)
with vertical dashed lines marking phase switches:

    setup → load → train → curve → eval → viz → report
"""
import json
import os
import threading
import time
from collections import deque
from typing import Optional

import psutil


class ResourceMonitor:
    def __init__(self, sample_interval: float = 5.0, max_samples: int = 200000):
        self.sample_interval = float(sample_interval)
        self.samples: deque = deque(maxlen=max_samples)
        self.phases: list[tuple[float, str]] = []
        self._proc = psutil.Process()
        self._proc.cpu_percent(interval=None)            # prime
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0: Optional[float] = None
        self._pynvml = None
        self._gpu_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._pynvml = None

    def start(self) -> None:
        self._t0 = time.time()
        self.mark('setup')
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def mark(self, phase: str) -> None:
        t = time.time() - (self._t0 or time.time())
        self.phases.append((t, phase))

    def _sample(self) -> dict:
        cpu_pct = self._proc.cpu_percent(interval=None)
        rss_gib = self._proc.memory_info().rss / 1024**3
        gpu_pct = 0.0
        gpu_gib = 0.0
        if self._pynvml is not None:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(
                    self._gpu_handle)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_pct = float(util.gpu)
                gpu_gib = float(mem.used) / 1024**3
            except Exception:
                pass
        return {'t': time.time() - self._t0, 'cpu_pct': cpu_pct,
                'rss_gib': rss_gib, 'gpu_pct': gpu_pct, 'gpu_gib': gpu_gib}

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.sample_interval)

    def save_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump({'samples': list(self.samples),
                       'phases': self.phases}, f)

    def save_png(self, path: str) -> None:
        import matplotlib.pyplot as plt
        if not self.samples:
            return
        ts = [s['t'] for s in self.samples]
        fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
        axes = axes.ravel()
        panels = [
            ('cpu_pct', 'CPU %',        'tab:blue'),
            ('rss_gib', 'CPU RSS (GiB)', 'tab:green'),
            ('gpu_pct', 'GPU %',        'tab:red'),
            ('gpu_gib', 'GPU mem (GiB)', 'tab:orange'),
        ]
        for ax, (key, title, color) in zip(axes, panels):
            ax.plot(ts, [s[key] for s in self.samples], color=color, lw=1)
            ax.set_title(title)
            ax.grid(alpha=0.3)
            for pt, phase in self.phases:
                ax.axvline(pt, color='k', ls='--', alpha=0.4)
                ax.text(pt, ax.get_ylim()[1] * 0.92, phase, rotation=90,
                        fontsize=7, alpha=0.6, va='top')
        for ax in axes[2:]:
            ax.set_xlabel('time (s)')
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        import matplotlib.pyplot as _plt
        _plt.close(fig)
