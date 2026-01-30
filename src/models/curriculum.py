"""
Curriculum Learning Scheduler for ISOTOPE.

Implements progressive difficulty through grid size scaling:
- Stage 1: 4x4 grid (simple)
- Stage 2: 6x6 grid (intermediate)
- Stage 3: 8x8 grid (full complexity)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage."""

    name: str
    grid_size: int
    num_agents: int
    physics_complexity: str  # "simple", "medium", "full"
    success_threshold: float  # Required success rate to advance
    min_episodes: int  # Minimum episodes before advancement check
    max_episodes: Optional[int] = None  # Maximum episodes in this stage


@dataclass
class CurriculumConfig:
    """Configuration for curriculum scheduler."""

    enabled: bool = True
    stages: List[CurriculumStage] = field(default_factory=list)

    # Advancement criteria
    metric: str = "ppf_under_limit_rate"  # Metric to track
    window_size: int = 100  # Rolling window for metric calculation

    # Regression handling
    allow_regression: bool = True
    regression_threshold: float = 0.40  # Regress if success drops below

    # Transition settings
    warmup_episodes: int = 50
    cooldown_episodes: int = 100

    def __post_init__(self):
        if not self.stages:
            self.stages = self._default_stages()

    def _default_stages(self) -> List[CurriculumStage]:
        """Create default curriculum stages."""
        return [
            CurriculumStage(
                name="stage_1_4x4",
                grid_size=4,
                num_agents=1,
                physics_complexity="simple",
                success_threshold=0.75,
                min_episodes=500,
                max_episodes=2000,
            ),
            CurriculumStage(
                name="stage_2_6x6",
                grid_size=6,
                num_agents=2,
                physics_complexity="simple",
                success_threshold=0.70,
                min_episodes=1000,
                max_episodes=4000,
            ),
            CurriculumStage(
                name="stage_3_8x8",
                grid_size=8,
                num_agents=4,
                physics_complexity="full",
                success_threshold=0.65,
                min_episodes=2000,
                max_episodes=None,
            ),
        ]


class GridCurriculumScheduler:
    """
    Curriculum scheduler for progressive grid size.

    Tracks success rate and manages stage transitions based on
    agent performance on safety constraints.
    """

    def __init__(self, config: Optional[CurriculumConfig] = None):
        """
        Initialize curriculum scheduler.

        Args:
            config: Curriculum configuration
        """
        self.config = config or CurriculumConfig()
        self.stages = self.config.stages

        # State
        self.current_stage_idx = 0
        self.episodes_in_stage = 0
        self.total_episodes = 0
        self.episode_results: List[Dict] = []

        # Transition tracking
        self.last_advancement_episode = 0
        self.last_regression_episode = 0

    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]

    @property
    def current_grid_size(self) -> int:
        """Get current grid size."""
        return self.current_stage.grid_size

    @property
    def current_num_agents(self) -> int:
        """Get current number of agents."""
        return self.current_stage.num_agents

    @property
    def is_final_stage(self) -> bool:
        """Check if we're in the final stage."""
        return self.current_stage_idx >= len(self.stages) - 1

    def get_env_config(self) -> Dict:
        """
        Get environment configuration for current stage.

        Returns:
            Dict with grid_size, num_agents, physics_complexity
        """
        stage = self.current_stage
        return {
            "grid_size": stage.grid_size,
            "num_agents": stage.num_agents,
            "physics_complexity": stage.physics_complexity,
        }

    def get_fuel_inventory_config(self) -> Dict:
        """
        Get scaled fuel inventory for current grid size.

        Returns:
            Dict with fuel counts scaled to grid size
        """
        grid_size = self.current_stage.grid_size
        total_positions = grid_size * grid_size

        # Scale fuel counts proportionally
        # Base ratios from 8x8: 24 fresh, 20 used, 20 depleted (total 64)
        fresh_ratio = 24 / 64
        used_ratio = 20 / 64
        depleted_ratio = 20 / 64

        fresh_count = max(4, int(total_positions * fresh_ratio))
        used_count = max(4, int(total_positions * used_ratio))
        depleted_count = total_positions - fresh_count - used_count

        return {
            "fresh_count": fresh_count,
            "used_count": used_count,
            "depleted_count": depleted_count,
        }

    def update(self, episode_result: Dict) -> Dict:
        """
        Update curriculum with episode result.

        Args:
            episode_result: Dict containing at least:
                - ppf_under_limit: bool (was final PPF < limit?)
                - mean_reward: float
                - final_ppf: float
                - success: bool (optional)

        Returns:
            Dict with stage change info:
                - stage_changed: bool
                - direction: "advance" | "regress" | None
                - new_stage: str (if changed)
                - current_success_rate: float
        """
        self.episode_results.append(episode_result)
        self.episodes_in_stage += 1
        self.total_episodes += 1

        result = {
            "stage_changed": False,
            "direction": None,
            "new_stage": None,
            "current_stage": self.current_stage.name,
            "current_success_rate": self._compute_success_rate(),
            "episodes_in_stage": self.episodes_in_stage,
        }

        # Check for advancement
        if self._should_advance():
            result = self._advance_stage(result)

        # Check for regression (if enabled)
        elif self.config.allow_regression and self._should_regress():
            result = self._regress_stage(result)

        return result

    def _compute_success_rate(self) -> float:
        """Compute success rate over recent episodes."""
        if not self.episode_results:
            return 0.0

        # Use rolling window
        recent = self.episode_results[-self.config.window_size:]

        # Count successes based on configured metric
        if self.config.metric == "ppf_under_limit_rate":
            successes = sum(1 for r in recent if r.get("ppf_under_limit", False))
        elif self.config.metric == "success_rate":
            successes = sum(1 for r in recent if r.get("success", False))
        else:
            successes = sum(1 for r in recent if r.get("ppf_under_limit", False))

        return successes / len(recent)

    def _should_advance(self) -> bool:
        """Check if conditions are met to advance to next stage."""
        # Can't advance from final stage
        if self.is_final_stage:
            return False

        # Must have completed minimum episodes
        if self.episodes_in_stage < self.current_stage.min_episodes:
            return False

        # Must be past warmup
        if self.episodes_in_stage < self.config.warmup_episodes:
            return False

        # Must be past cooldown from last transition
        if (
            self.total_episodes - self.last_advancement_episode
            < self.config.cooldown_episodes
        ):
            return False

        # Check success rate
        success_rate = self._compute_success_rate()
        return success_rate >= self.current_stage.success_threshold

    def _should_regress(self) -> bool:
        """Check if conditions are met to regress to previous stage."""
        # Can't regress from first stage
        if self.current_stage_idx == 0:
            return False

        # Must have some episodes in current stage
        if self.episodes_in_stage < self.config.warmup_episodes:
            return False

        # Must be past cooldown from last transition
        if (
            self.total_episodes - self.last_regression_episode
            < self.config.cooldown_episodes
        ):
            return False

        # Check if success rate dropped too low
        success_rate = self._compute_success_rate()
        return success_rate < self.config.regression_threshold

    def _advance_stage(self, result: Dict) -> Dict:
        """Advance to next curriculum stage."""
        self.current_stage_idx += 1
        self.episodes_in_stage = 0
        self.last_advancement_episode = self.total_episodes

        result["stage_changed"] = True
        result["direction"] = "advance"
        result["new_stage"] = self.current_stage.name

        return result

    def _regress_stage(self, result: Dict) -> Dict:
        """Regress to previous curriculum stage."""
        self.current_stage_idx -= 1
        self.episodes_in_stage = 0
        self.last_regression_episode = self.total_episodes

        result["stage_changed"] = True
        result["direction"] = "regress"
        result["new_stage"] = self.current_stage.name

        return result

    def force_stage(self, stage_idx: int) -> None:
        """
        Force transition to a specific stage.

        Args:
            stage_idx: Index of stage to transition to
        """
        if 0 <= stage_idx < len(self.stages):
            self.current_stage_idx = stage_idx
            self.episodes_in_stage = 0

    def get_stats(self) -> Dict:
        """Get curriculum statistics."""
        return {
            "current_stage": self.current_stage.name,
            "current_stage_idx": self.current_stage_idx,
            "episodes_in_stage": self.episodes_in_stage,
            "total_episodes": self.total_episodes,
            "success_rate": self._compute_success_rate(),
            "grid_size": self.current_grid_size,
            "num_agents": self.current_num_agents,
            "is_final_stage": self.is_final_stage,
        }

    def save_state(self) -> Dict:
        """Save curriculum state for checkpointing."""
        return {
            "current_stage_idx": self.current_stage_idx,
            "episodes_in_stage": self.episodes_in_stage,
            "total_episodes": self.total_episodes,
            "episode_results": self.episode_results[-self.config.window_size:],
            "last_advancement_episode": self.last_advancement_episode,
            "last_regression_episode": self.last_regression_episode,
        }

    def load_state(self, state: Dict) -> None:
        """Load curriculum state from checkpoint."""
        self.current_stage_idx = state["current_stage_idx"]
        self.episodes_in_stage = state["episodes_in_stage"]
        self.total_episodes = state["total_episodes"]
        self.episode_results = state.get("episode_results", [])
        self.last_advancement_episode = state.get("last_advancement_episode", 0)
        self.last_regression_episode = state.get("last_regression_episode", 0)


def create_curriculum_from_config(cfg: Dict) -> GridCurriculumScheduler:
    """
    Create curriculum scheduler from Hydra config dict.

    Args:
        cfg: Curriculum configuration dictionary

    Returns:
        Configured GridCurriculumScheduler
    """
    stages = []
    for stage_cfg in cfg.get("stages", []):
        stages.append(
            CurriculumStage(
                name=stage_cfg["name"],
                grid_size=stage_cfg["grid_size"],
                num_agents=stage_cfg["num_agents"],
                physics_complexity=stage_cfg.get("physics_complexity", "simple"),
                success_threshold=stage_cfg["success_threshold"],
                min_episodes=stage_cfg["min_episodes"],
                max_episodes=stage_cfg.get("max_episodes"),
            )
        )

    config = CurriculumConfig(
        enabled=cfg.get("enabled", True),
        stages=stages if stages else None,
        metric=cfg.get("advancement", {}).get("metric", "ppf_under_limit_rate"),
        window_size=cfg.get("advancement", {}).get("window_size", 100),
        allow_regression=cfg.get("advancement", {}).get("allow_regression", True),
        regression_threshold=cfg.get("advancement", {}).get("regression_threshold", 0.40),
        warmup_episodes=cfg.get("transition", {}).get("warmup_episodes", 50),
        cooldown_episodes=cfg.get("transition", {}).get("cooldown_episodes", 100),
    )

    return GridCurriculumScheduler(config)
