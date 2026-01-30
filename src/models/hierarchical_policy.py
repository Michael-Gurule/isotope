"""
Hierarchical Multi-Agent Policy for ISOTOPE.

Implements a two-level hierarchy:
- Master Policy: Observes global state, outputs strategy embedding and region priorities
- Sub-Agent Policies: Conditioned on master strategy, control individual quadrants
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.models.mappo import get_activation, RolloutBuffer


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical policy."""

    # Architecture
    num_subagents: int = 4
    strategy_dim: int = 16  # Master strategy embedding dimension
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    activation: str = "relu"

    # Master update schedule
    strategy_update_interval: int = 5  # Update master every N steps

    # Learning rates
    master_lr: float = 1e-4
    subagent_lr: float = 3e-4

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    num_epochs: int = 4
    minibatch_size: int = 64

    # Device
    device: str = "cpu"


class MasterPolicy(nn.Module):
    """
    Master agent that sets high-level strategy.

    Observes global core state and outputs:
    - Strategy embedding: Conditioning signal for sub-agents
    - Region priorities: Attention weights for each quadrant
    """

    def __init__(
        self,
        global_obs_dim: int,
        strategy_dim: int,
        num_regions: int,
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        """
        Initialize master policy.

        Args:
            global_obs_dim: Dimension of full core observation
            strategy_dim: Dimension of strategy embedding
            num_regions: Number of regions/sub-agents
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
        """
        super().__init__()

        self.strategy_dim = strategy_dim
        self.num_regions = num_regions

        # Encoder
        layers = []
        in_dim = global_obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(get_activation(activation))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Strategy head (learned embedding for sub-agent conditioning)
        self.strategy_head = nn.Linear(hidden_dims[-1], strategy_dim)

        # Priority head (attention over regions)
        self.priority_head = nn.Linear(hidden_dims[-1], num_regions)

        # Value head (for master's value function)
        self.value_head = nn.Linear(hidden_dims[-1], 1)

    def forward(
        self, global_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            global_obs: Global observation tensor [batch, global_obs_dim]

        Returns:
            Tuple of (strategy, priorities, value)
            - strategy: [batch, strategy_dim]
            - priorities: [batch, num_regions] (softmax normalized)
            - value: [batch, 1]
        """
        features = self.encoder(global_obs)

        strategy = self.strategy_head(features)
        strategy = torch.tanh(strategy)  # Bound strategy to [-1, 1]

        priorities = F.softmax(self.priority_head(features), dim=-1)

        value = self.value_head(features)

        return strategy, priorities, value


class SubAgentPolicy(nn.Module):
    """
    Sub-agent policy conditioned on master strategy.

    Each sub-agent controls one region of the reactor core.
    """

    def __init__(
        self,
        local_obs_dim: int,
        strategy_dim: int,
        action_dims: List[int],
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        """
        Initialize sub-agent policy.

        Args:
            local_obs_dim: Dimension of local region observation
            strategy_dim: Dimension of master strategy embedding
            action_dims: Action dimensions for MultiDiscrete [type, source, target]
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
        """
        super().__init__()

        self.action_dims = action_dims

        # Combined encoder (local obs + strategy)
        combined_dim = local_obs_dim + strategy_dim
        layers = []
        in_dim = combined_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(get_activation(activation))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Action heads for MultiDiscrete
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], dim) for dim in action_dims
        ])

        # Value head
        self.value_head = nn.Linear(hidden_dims[-1], 1)

    def forward(
        self, local_obs: torch.Tensor, strategy: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            local_obs: Local observation [batch, local_obs_dim]
            strategy: Master strategy [batch, strategy_dim]

        Returns:
            Tuple of (action_logits_list, value)
        """
        combined = torch.cat([local_obs, strategy], dim=-1)
        features = self.encoder(combined)

        action_logits = [head(features) for head in self.action_heads]
        value = self.value_head(features)

        return action_logits, value

    def get_action(
        self,
        local_obs: torch.Tensor,
        strategy: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            local_obs: Local observation
            strategy: Master strategy embedding
            deterministic: If True, take argmax

        Returns:
            Tuple of (actions, log_probs, values)
        """
        logits_list, value = self(local_obs, strategy)

        actions = []
        log_probs = []

        for logits in logits_list:
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()

            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)

            actions.append(action)
            log_probs.append(log_prob)

        actions = torch.stack(actions, dim=-1)
        log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)

        return actions, log_probs, value.squeeze(-1)

    def evaluate_actions(
        self,
        local_obs: torch.Tensor,
        strategy: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.

        Args:
            local_obs: Local observation [batch, local_obs_dim]
            strategy: Master strategy [batch, strategy_dim]
            actions: Actions [batch, num_action_dims]

        Returns:
            Tuple of (log_probs, entropy, values)
        """
        logits_list, value = self(local_obs, strategy)

        log_probs = []
        entropies = []

        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(actions[:, i])
            entropy = dist.entropy()

            log_probs.append(log_prob)
            entropies.append(entropy)

        log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)
        entropy = torch.stack(entropies, dim=-1).mean(dim=-1)

        return log_probs, entropy, value.squeeze(-1)


class HierarchicalMAPPO:
    """
    Hierarchical Multi-Agent PPO with Master and Sub-agent policies.

    Training dynamics:
    - Master updated every strategy_update_interval steps
    - Sub-agents updated every step
    - Master receives aggregate reward signal from all sub-agents
    """

    def __init__(
        self,
        global_obs_dim: int,
        local_obs_dim: int,
        action_dims: List[int],
        num_agents: int,
        config: Optional[HierarchicalConfig] = None,
    ):
        """
        Initialize hierarchical MAPPO.

        Args:
            global_obs_dim: Dimension of global state (for master)
            local_obs_dim: Dimension of local observation (for sub-agents)
            action_dims: Action dimensions for MultiDiscrete
            num_agents: Number of sub-agents
            config: Configuration
        """
        self.config = config or HierarchicalConfig()
        self.num_agents = num_agents
        self.global_obs_dim = global_obs_dim
        self.local_obs_dim = local_obs_dim
        self.action_dims = action_dims

        self.device = torch.device(self.config.device)

        # Create master policy
        self.master = MasterPolicy(
            global_obs_dim=global_obs_dim,
            strategy_dim=self.config.strategy_dim,
            num_regions=num_agents,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
        ).to(self.device)

        # Create sub-agent policies (shared weights)
        self.subagent = SubAgentPolicy(
            local_obs_dim=local_obs_dim,
            strategy_dim=self.config.strategy_dim,
            action_dims=action_dims,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
        ).to(self.device)

        # Optimizers
        self.master_optimizer = torch.optim.Adam(
            self.master.parameters(), lr=self.config.master_lr
        )
        self.subagent_optimizer = torch.optim.Adam(
            self.subagent.parameters(), lr=self.config.subagent_lr
        )

        # Buffers
        self.master_buffer = {
            "global_obs": [],
            "strategies": [],
            "priorities": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }

        self.subagent_buffer = RolloutBuffer(
            buffer_size=2048,
            num_agents=num_agents,
            obs_dim=local_obs_dim,
            state_dim=global_obs_dim,
            action_dims=len(action_dims),
        )

        # Current strategy (updated periodically)
        self.current_strategy: Optional[torch.Tensor] = None
        self.current_priorities: Optional[torch.Tensor] = None

        # Counters
        self.step_count = 0
        self.training_step = 0

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        global_state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
        """
        Select actions for all agents.

        Args:
            observations: Dict mapping agent_id to local observation
            global_state: Global state for master
            deterministic: If True, use greedy actions

        Returns:
            Tuple of (actions, log_probs, values) dicts
        """
        actions = {}
        log_probs = {}
        values = {}

        with torch.no_grad():
            # Update master strategy periodically
            if (
                self.current_strategy is None
                or self.step_count % self.config.strategy_update_interval == 0
            ):
                global_tensor = torch.tensor(
                    global_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                strategy, priorities, master_value = self.master(global_tensor)
                self.current_strategy = strategy
                self.current_priorities = priorities

                # Store master transition
                self.master_buffer["global_obs"].append(global_state)
                self.master_buffer["strategies"].append(strategy.squeeze(0).cpu().numpy())
                self.master_buffer["priorities"].append(priorities.squeeze(0).cpu().numpy())
                self.master_buffer["values"].append(master_value.item())

            # Get sub-agent actions
            strategy = self.current_strategy

            for agent, obs in observations.items():
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                action, log_prob, value = self.subagent.get_action(
                    obs_tensor, strategy, deterministic
                )

                actions[agent] = action.squeeze(0).cpu().numpy()
                log_probs[agent] = log_prob.item()
                values[agent] = value.item()

        self.step_count += 1
        return actions, log_probs, values

    def store_transition(
        self,
        observations: Dict[str, np.ndarray],
        global_state: np.ndarray,
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        dones: Dict[str, bool],
        log_probs: Dict[str, float],
        values: Dict[str, float],
    ):
        """Store transition in buffers."""
        # Store sub-agent data
        self.subagent_buffer.add(
            observations=observations,
            state=global_state,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_probs=log_probs,
            values=values,
        )

        # Update master buffer with aggregate reward
        if self.master_buffer["global_obs"]:
            agg_reward = np.mean(list(rewards.values()))
            self.master_buffer["rewards"].append(agg_reward)
            self.master_buffer["dones"].append(any(dones.values()))

    def update(
        self, last_values: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update master and sub-agent policies.

        Args:
            last_values: Value estimates for final state

        Returns:
            Dict of training statistics
        """
        stats = {
            "master_loss": 0.0,
            "subagent_policy_loss": 0.0,
            "subagent_value_loss": 0.0,
            "entropy": 0.0,
        }

        # Update sub-agents
        subagent_stats = self._update_subagents(last_values)
        stats.update(subagent_stats)

        # Update master (less frequently or with different objective)
        if len(self.master_buffer["rewards"]) > 0:
            master_stats = self._update_master()
            stats.update(master_stats)

        self.training_step += 1
        return stats

    def _update_subagents(
        self, last_values: Dict[str, float]
    ) -> Dict[str, float]:
        """Update sub-agent policy using PPO."""
        data = self.subagent_buffer.get(self.device)
        agents = [a for a in data.keys() if a != "states"]

        stats = {
            "subagent_policy_loss": 0.0,
            "subagent_value_loss": 0.0,
            "entropy": 0.0,
        }
        num_updates = 0

        for agent in agents:
            agent_data = data[agent]

            # Compute GAE
            advantages, returns = self._compute_gae(
                agent_data["rewards"],
                agent_data["values"],
                agent_data["dones"],
                last_values[agent],
            )

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Get strategy for this rollout
            strategy = self.current_strategy.expand(len(advantages), -1)

            for _ in range(self.config.num_epochs):
                log_probs_new, entropy, values_new = self.subagent.evaluate_actions(
                    agent_data["observations"],
                    strategy,
                    agent_data["actions"],
                )

                # PPO losses
                ratio = torch.exp(log_probs_new - agent_data["log_probs"])
                policy_loss_1 = ratio * advantages
                policy_loss_2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                ) * advantages
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                value_loss = F.mse_loss(values_new, returns)

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy.mean()
                )

                self.subagent_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.subagent.parameters(), self.config.max_grad_norm
                )
                self.subagent_optimizer.step()

                stats["subagent_policy_loss"] += policy_loss.item()
                stats["subagent_value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                num_updates += 1

        # Average
        for key in stats:
            stats[key] /= max(num_updates, 1)

        self.subagent_buffer.reset()
        return stats

    def _update_master(self) -> Dict[str, float]:
        """Update master policy."""
        # Convert buffer to tensors
        global_obs = torch.tensor(
            np.array(self.master_buffer["global_obs"]),
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.tensor(
            np.array(self.master_buffer["rewards"]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            np.array(self.master_buffer["dones"]),
            dtype=torch.float32,
            device=self.device,
        )
        values = torch.tensor(
            np.array(self.master_buffer["values"]),
            dtype=torch.float32,
            device=self.device,
        )

        # Compute returns (simple discounted)
        returns = torch.zeros_like(rewards)
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.gamma * running_return * (1 - dones[t])
            returns[t] = running_return

        # Simple value loss for master
        _, _, values_new = self.master(global_obs)
        value_loss = F.mse_loss(values_new.squeeze(-1), returns)

        self.master_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.master.parameters(), self.config.max_grad_norm
        )
        self.master_optimizer.step()

        # Clear master buffer
        self.master_buffer = {
            "global_obs": [],
            "strategies": [],
            "priorities": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }

        return {"master_loss": value_loss.item()}

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE."""
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = (
                rewards[t]
                + self.config.gamma * next_value * (1 - dones[t])
                - values[t]
            )
            advantages[t] = last_gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            )

        returns = advantages + values
        return advantages, returns

    def save(self, path: str) -> None:
        """Save checkpoint."""
        torch.save(
            {
                "master_state_dict": self.master.state_dict(),
                "subagent_state_dict": self.subagent.state_dict(),
                "master_optimizer_state_dict": self.master_optimizer.state_dict(),
                "subagent_optimizer_state_dict": self.subagent_optimizer.state_dict(),
                "config": self.config,
                "training_step": self.training_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.master.load_state_dict(checkpoint["master_state_dict"])
        self.subagent.load_state_dict(checkpoint["subagent_state_dict"])
        self.master_optimizer.load_state_dict(checkpoint["master_optimizer_state_dict"])
        self.subagent_optimizer.load_state_dict(checkpoint["subagent_optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
