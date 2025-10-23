from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional

from prometheus_client import REGISTRY

try:  # pragma: no cover - optional when V1 is available
    from vllm.v1.metrics.reader import Counter, Gauge, Metric
except ImportError:  # pragma: no cover - fallback for pure V0 builds
    Counter = Gauge = Metric = None  # type: ignore


@dataclass
class TrackerSummary:
    duration_s: float
    prompt_tokens: int
    generation_tokens: int
    prompt_throughput: float
    generation_throughput: float
    max_kv_usage: float
    total_kv_bytes: Optional[int]
    max_kv_bytes: Optional[float]

    @property
    def max_kv_mebibytes(self) -> Optional[float]:
        if self.max_kv_bytes is None:
            return None
        return self.max_kv_bytes / (1024 ** 2)

    def __str__(self) -> str:
        parts = [
            f"duration={self.duration_s:.2f}s",
            f"prompt_tokens={self.prompt_tokens}",
            f"generation_tokens={self.generation_tokens}",
            f"prompt_throughput={self.prompt_throughput:.2f} tok/s",
            f"generation_throughput={self.generation_throughput:.2f} tok/s",
            f"max_kv_usage={self.max_kv_usage * 100:.1f}%",
        ]
        if self.max_kv_bytes is not None:
            parts.append(f"max_kv_memory={self.max_kv_mebibytes:.2f} MiB")
        else:
            parts.append("max_kv_memory=N/A")
        return "[tracker] " + ", ".join(parts)


@dataclass
class MetricValues:
    prompt_tokens: int = 0
    generation_tokens: int = 0
    kv_usage: float = 0.0


class GenerationMetricsTracker:
    """Utility for recording KV cache usage and throughput metrics."""

    def __init__(self, llm, verbose: bool = True) -> None:
        self.llm = llm
        self.verbose = verbose
        self.enabled: bool = False
        self.disabled_reason: Optional[str] = None

        self._start_time: Optional[float] = None
        self._start_metrics: Optional[MetricValues] = None
        self.max_kv_usage: float = 0.0

        self._is_v1 = self._detect_v1_backend()
        self._label_filter = self._infer_label_filter()

        self.total_kv_bytes: Optional[int] = self._compute_total_kv_bytes()

    def start(self) -> None:
        metrics = self._try_read_metrics()
        if metrics is None:
            return

        self.enabled = True
        self._start_metrics = metrics
        self.max_kv_usage = metrics.kv_usage
        self._start_time = perf_counter()

    def sample(self) -> Optional[MetricValues]:
        if not self.enabled:
            return None
        metrics = self._try_read_metrics()
        if metrics is None:
            return None
        self.max_kv_usage = max(self.max_kv_usage, metrics.kv_usage)
        return metrics

    def stop(self) -> Optional[TrackerSummary]:
        if not self.enabled or self._start_time is None or \
                self._start_metrics is None:
            return None

        metrics = self.sample()
        if metrics is None:
            return None

        elapsed = perf_counter() - self._start_time
        elapsed = max(elapsed, 1e-9)

        prompt_tokens = max(
            0, metrics.prompt_tokens - self._start_metrics.prompt_tokens)
        generation_tokens = max(
            0, metrics.generation_tokens - self._start_metrics.generation_tokens)

        summary = TrackerSummary(
            duration_s=elapsed,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            prompt_throughput=prompt_tokens / elapsed,
            generation_throughput=generation_tokens / elapsed,
            max_kv_usage=self.max_kv_usage,
            total_kv_bytes=self.total_kv_bytes,
            max_kv_bytes=(self.max_kv_usage * self.total_kv_bytes
                          if self.total_kv_bytes is not None else None),
        )
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _try_read_metrics(self) -> Optional[MetricValues]:
        try:
            return self._read_metrics()
        except Exception as exc:  # noqa: BLE001
            if self.disabled_reason is None:
                self.disabled_reason = (
                    "metrics unavailable - ensure disable_log_stats=False "
                    f"and Prometheus stats are enabled ({exc})"
                )
                if self.verbose:
                    print(f"[tracker] {self.disabled_reason}")
            self.enabled = False
            return None

    def _read_metrics(self) -> MetricValues:
        if self._is_v1 and Counter is not None:
            return self._read_metrics_v1()
        return self._read_metrics_v0()

    def _read_metrics_v1(self) -> MetricValues:
        metrics = self.llm.get_metrics()
        prompt_tokens = 0
        generation_tokens = 0
        kv_usage = 0.0

        for metric in metrics:
            if isinstance(metric, Counter):
                if metric.name in ("vllm:prompt_tokens",
                                   "vllm:prompt_tokens_total"):
                    prompt_tokens += int(metric.value)
                elif metric.name in ("vllm:generation_tokens",
                                     "vllm:generation_tokens_total"):
                    generation_tokens += int(metric.value)
            elif isinstance(metric, Gauge):
                if metric.name in ("vllm:kv_cache_usage_perc",
                                   "vllm:gpu_cache_usage_perc"):
                    kv_usage = max(kv_usage, float(metric.value))

        return MetricValues(prompt_tokens, generation_tokens, kv_usage)

    def _read_metrics_v0(self) -> MetricValues:
        prompt_tokens = 0
        generation_tokens = 0
        kv_usage = 0.0

        for metric in REGISTRY.collect():
            if not metric.name.startswith("vllm:"):
                continue

            if metric.name not in {
                "vllm:prompt_tokens_total",
                "vllm:generation_tokens_total",
                "vllm:gpu_cache_usage_perc",
            }:
                continue

            for sample in metric.samples:
                if sample.name != metric.name:
                    continue
                if not self._labels_match(sample.labels):
                    continue
                if metric.name == "vllm:prompt_tokens_total":
                    prompt_tokens += int(sample.value)
                elif metric.name == "vllm:generation_tokens_total":
                    generation_tokens += int(sample.value)
                elif metric.name == "vllm:gpu_cache_usage_perc":
                    kv_usage = max(kv_usage, float(sample.value))

        return MetricValues(prompt_tokens, generation_tokens, kv_usage)

    def _labels_match(self, sample_labels: Dict[str, str]) -> bool:
        if not self._label_filter:
            return True
        for key, expected in self._label_filter.items():
            if sample_labels.get(key) != expected:
                return False
        return True

    def _detect_v1_backend(self) -> bool:
        try:
            from vllm.v1.engine.llm_engine import LLMEngine as V1Engine
            return isinstance(getattr(self.llm, "llm_engine", None), V1Engine)
        except ImportError:  # pragma: no cover - V1 not installed
            return False

    def _infer_label_filter(self) -> Optional[Dict[str, str]]:
        engine = getattr(self.llm, "llm_engine", None)
        model_config = getattr(engine, "model_config", None)
        if model_config is None:
            return None
        served_name = getattr(model_config, "served_model_name", None)
        if served_name:
            return {"model_name": served_name}
        model_name = getattr(model_config, "model", None)
        if model_name:
            return {"model_name": model_name}
        return None

    def _compute_total_kv_bytes(self) -> Optional[int]:
        engine = getattr(self.llm, "llm_engine", None)
        cache_config = getattr(engine, "cache_config", None)
        kv_tensors = getattr(cache_config, "kv_cache_tensors", None)
        if kv_tensors is None:
            return None
        total = 0
        for tensor in kv_tensors:
            size = getattr(tensor, "size", None)
            if size is None:
                return None
            total += int(size)
        return total if total > 0 else None
