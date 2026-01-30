"""Metrics tracking utilities for ISOTOPE."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SafetyMetrics:
    """Safety constraint metrics for reactor operation."""

    ppf_violation: bool  # True if PPF exceeds limit
    ppf_margin: float  # Distance below PPF limit (negative if violated)
    k_deviation: float  # Absolute deviation from k=1.0
    hotspot_count: int  # Number of assemblies above 1.3x average power
    max_local_power: float  # Maximum relative power in core


@dataclass
class OptimizationMetrics:
    """Training optimization metrics."""

    episode_reward: float
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0


@dataclass
class EpisodeMetrics:
    """Per-episode metrics tracking."""

    episode_id: int
    steps: int
    total_reward: float
    final_ppf: float
    final_k_effective: float
    ppf_violations: int  # Number of steps with PPF > limit
    mean_ppf: float
    min_ppf: float
    improvement: float  # Change in PPF from start to end
    terminated_early: bool = False
    success: bool = False  # True if ended with PPF < limit and k near 1.0

    @classmethod
    def from_history(
        cls,
        episode_id: int,
        rewards: List[float],
        ppf_history: List[float],
        k_history: List[float],
        ppf_limit: float = 1.5,
        k_tolerance: float = 0.05,
        terminated_early: bool = False,
    ) -> "EpisodeMetrics":
        """Create metrics from episode history."""
        steps = len(rewards)
        total_reward = sum(rewards)

        if ppf_history:
            final_ppf = ppf_history[-1]
            mean_ppf = np.mean(ppf_history)
            min_ppf = np.min(ppf_history)
            ppf_violations = sum(1 for p in ppf_history if p > ppf_limit)
            improvement = ppf_history[0] - ppf_history[-1] if len(ppf_history) > 1 else 0.0
        else:
            final_ppf = mean_ppf = min_ppf = 0.0
            ppf_violations = 0
            improvement = 0.0

        final_k = k_history[-1] if k_history else 1.0
        success = final_ppf < ppf_limit and abs(final_k - 1.0) < k_tolerance

        return cls(
            episode_id=episode_id,
            steps=steps,
            total_reward=total_reward,
            final_ppf=final_ppf,
            final_k_effective=final_k,
            ppf_violations=ppf_violations,
            mean_ppf=mean_ppf,
            min_ppf=min_ppf,
            improvement=improvement,
            terminated_early=terminated_early,
            success=success,
        )


@dataclass
class EvaluationMetrics:
    """Aggregate evaluation metrics across multiple episodes."""

    episodes: List[EpisodeMetrics] = field(default_factory=list)

    def add_episode(self, episode: EpisodeMetrics) -> None:
        """Add an episode to the evaluation."""
        self.episodes.append(episode)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics across all episodes."""
        if not self.episodes:
            return {}

        return {
            "num_episodes": len(self.episodes),
            "success_rate": np.mean([e.success for e in self.episodes]),
            "mean_reward": np.mean([e.total_reward for e in self.episodes]),
            "std_reward": np.std([e.total_reward for e in self.episodes]),
            "mean_final_ppf": np.mean([e.final_ppf for e in self.episodes]),
            "mean_final_k": np.mean([e.final_k_effective for e in self.episodes]),
            "mean_ppf_violations": np.mean([e.ppf_violations for e in self.episodes]),
            "mean_improvement": np.mean([e.improvement for e in self.episodes]),
            "early_termination_rate": np.mean([e.terminated_early for e in self.episodes]),
        }

    def get_best_episode(self) -> Optional[EpisodeMetrics]:
        """Get the episode with highest reward."""
        if not self.episodes:
            return None
        return max(self.episodes, key=lambda e: e.total_reward)

    def get_safest_episode(self) -> Optional[EpisodeMetrics]:
        """Get the episode with lowest final PPF."""
        if not self.episodes:
            return None
        return min(self.episodes, key=lambda e: e.final_ppf)


class RunningMeanStd:
    """
    Running mean and standard deviation calculator for reward normalization.

    Uses Welford's online algorithm for numerical stability.
    """

    def __init__(self, epsilon: float = 1e-4):
        """
        Initialize the running statistics.

        Args:
            epsilon: Small constant for numerical stability
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """
        Update statistics with new batch of values.

        Args:
            x: Array of new values
        """
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: float, batch_var: float, batch_count: int
    ) -> None:
        """Update from batch moments using parallel algorithm."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: float) -> float:
        """
        Normalize a value using running statistics.

        Args:
            x: Value to normalize

        Returns:
            Normalized value
        """
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def denormalize(self, x: float) -> float:
        """
        Denormalize a value.

        Args:
            x: Normalized value

        Returns:
            Original scale value
        """
        return x * np.sqrt(self.var) + self.mean


def compute_safety_metrics(
    power: np.ndarray,
    ppf: float,
    k_effective: float,
    ppf_limit: float = 1.5,
) -> SafetyMetrics:
    """
    Compute safety metrics from physics results.

    Args:
        power: Power distribution array
        ppf: Power Peaking Factor
        k_effective: Multiplication factor
        ppf_limit: Maximum allowable PPF

    Returns:
        SafetyMetrics dataclass
    """
    avg_power = np.mean(power)
    if avg_power > 0:
        relative_powers = power / avg_power
        hotspot_count = np.sum(relative_powers > 1.3)
        max_local_power = np.max(relative_powers)
    else:
        hotspot_count = 0
        max_local_power = 0.0

    return SafetyMetrics(
        ppf_violation=ppf > ppf_limit,
        ppf_margin=ppf_limit - ppf,
        k_deviation=abs(k_effective - 1.0),
        hotspot_count=int(hotspot_count),
        max_local_power=max_local_power,
    )
