"""
Multi-Agent Proximal Policy Optimization (MAPPO) implementation for ISOTOPE.

Based on CleanRL-style single-file implementation with support for:
- Shared policy weights across agents
- Centralized value function with global state
- GAE advantage estimation
- PPO clipping
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO algorithm."""

    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    lr_schedule: str = "linear"  # "constant", "linear", "cosine"

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    clip_value: bool = True
    value_clip_epsilon: float = 0.2

    # Loss coefficients
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # Training
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    normalize_rewards: bool = True
    num_epochs: int = 4
    minibatch_size: int = 64

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    activation: str = "relu"

    # Device
    device: str = "cpu"


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
    }
    return activations.get(name, nn.ReLU())


class ActorNetwork(nn.Module):
    """
    Policy network for discrete actions.

    Outputs action logits for MultiDiscrete action space.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dims: List[int],  # For MultiDiscrete: [3, 16, 16]
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        super().__init__()

        self.action_dims = action_dims

        # Build encoder layers
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(get_activation(activation))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Separate heads for each action dimension
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], dim) for dim in action_dims
        ])

    def forward(self, obs: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observation tensor [batch, obs_dim]

        Returns:
            List of action logits for each dimension
        """
        features = self.encoder(obs)
        return [head(features) for head in self.action_heads]

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: Observation tensor
            deterministic: If True, take argmax instead of sampling

        Returns:
            Tuple of (actions, log_probs)
        """
        logits_list = self(obs)
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

        # Stack actions and sum log probs
        actions = torch.stack(actions, dim=-1)  # [batch, num_action_dims]
        log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)  # [batch]

        return actions, log_probs

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.

        Args:
            obs: Observation tensor [batch, obs_dim]
            actions: Action tensor [batch, num_action_dims]

        Returns:
            Tuple of (log_probs, entropy)
        """
        logits_list = self(obs)
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

        return log_probs, entropy


class CriticNetwork(nn.Module):
    """
    Value network (centralized critic).

    Takes global state and outputs value estimate.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        super().__init__()

        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(get_activation(activation))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Global state tensor [batch, state_dim]

        Returns:
            Value estimate [batch, 1]
        """
        return self.network(state)


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self, buffer_size: int, num_agents: int, obs_dim: int, state_dim: int, action_dims: int):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dims = action_dims

        self.reset()

    def reset(self):
        """Clear buffer."""
        self.observations = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.ptr = 0

    def add(
        self,
        observations: Dict[str, np.ndarray],
        state: np.ndarray,
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        dones: Dict[str, bool],
        log_probs: Dict[str, float],
        values: Dict[str, float],
    ):
        """Add transition to buffer."""
        self.observations.append(observations)
        self.states.append(state)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.ptr += 1

    def get(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Get all data from buffer as tensors.

        Returns dict with keys for each agent plus global state.
        """
        agents = list(self.observations[0].keys())

        # Stack data per agent
        data = {}
        for agent in agents:
            obs = torch.tensor(
                np.array([o[agent] for o in self.observations]),
                dtype=torch.float32,
                device=device,
            )
            actions = torch.tensor(
                np.array([a[agent] for a in self.actions]),
                dtype=torch.long,
                device=device,
            )
            rewards = torch.tensor(
                np.array([r[agent] for r in self.rewards]),
                dtype=torch.float32,
                device=device,
            )
            dones = torch.tensor(
                np.array([d[agent] for d in self.dones]),
                dtype=torch.float32,
                device=device,
            )
            log_probs = torch.tensor(
                np.array([lp[agent] for lp in self.log_probs]),
                dtype=torch.float32,
                device=device,
            )
            values = torch.tensor(
                np.array([v[agent] for v in self.values]),
                dtype=torch.float32,
                device=device,
            )

            data[agent] = {
                "observations": obs,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "log_probs": log_probs,
                "values": values,
            }

        # Global state
        data["states"] = torch.tensor(
            np.array(self.states),
            dtype=torch.float32,
            device=device,
        )

        return data

    def __len__(self):
        return self.ptr


class MAPPO:
    """
    Multi-Agent PPO trainer.

    Supports shared policy weights across agents with a centralized critic.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dims: List[int],
        num_agents: int,
        config: Optional[MAPPOConfig] = None,
    ):
        """
        Initialize MAPPO trainer.

        Args:
            obs_dim: Dimension of agent observations
            state_dim: Dimension of global state (for critic)
            action_dims: List of action dimensions for MultiDiscrete
            num_agents: Number of agents
            config: MAPPO configuration
        """
        self.config = config or MAPPOConfig()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dims = action_dims

        self.device = torch.device(self.config.device)

        # Create networks (shared across agents)
        self.actor = ActorNetwork(
            obs_dim=obs_dim,
            action_dims=action_dims,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
        ).to(self.device)

        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=2048,
            num_agents=num_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dims=len(action_dims),
        )

        # Running statistics for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 1e-4

        # Training step counter
        self.training_step = 0

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
        """
        Select actions for all agents.

        Args:
            observations: Dict mapping agent_id to observation
            state: Global state for centralized critic
            deterministic: If True, use greedy actions

        Returns:
            Tuple of (actions, log_probs, values) dicts
        """
        actions = {}
        log_probs = {}
        values = {}

        with torch.no_grad():
            # Get value from centralized critic
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            value = self.critic(state_tensor).item()

            for agent, obs in observations.items():
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                action, log_prob = self.actor.get_action(obs_tensor, deterministic)

                actions[agent] = action.squeeze(0).cpu().numpy()
                log_probs[agent] = log_prob.item()
                values[agent] = value  # Same value for all agents (centralized)

        return actions, log_probs, values

    def store_transition(
        self,
        observations: Dict[str, np.ndarray],
        state: np.ndarray,
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        dones: Dict[str, bool],
        log_probs: Dict[str, float],
        values: Dict[str, float],
    ):
        """Store transition in buffer."""
        self.buffer.add(
            observations=observations,
            state=state,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_probs=log_probs,
            values=values,
        )

    def update(
        self, last_values: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update policy and value networks using collected rollout.

        Args:
            last_values: Value estimates for final state

        Returns:
            Dict of training statistics
        """
        data = self.buffer.get(self.device)
        agents = [a for a in data.keys() if a != "states"]

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
        }

        num_updates = 0

        for agent in agents:
            agent_data = data[agent]
            states = data["states"]

            # Compute advantages using GAE
            advantages, returns = self._compute_gae(
                agent_data["rewards"],
                agent_data["values"],
                agent_data["dones"],
                last_values[agent],
            )

            # Normalize advantages
            if self.config.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update epochs
            for _ in range(self.config.num_epochs):
                # Get current policy outputs
                log_probs_new, entropy = self.actor.evaluate_actions(
                    agent_data["observations"], agent_data["actions"]
                )
                values_new = self.critic(states).squeeze(-1)

                # Policy loss (PPO clip objective)
                ratio = torch.exp(log_probs_new - agent_data["log_probs"])
                policy_loss_1 = ratio * advantages
                policy_loss_2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                ) * advantages
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss (optionally clipped)
                if self.config.clip_value:
                    values_clipped = agent_data["values"] + torch.clamp(
                        values_new - agent_data["values"],
                        -self.config.value_clip_epsilon,
                        self.config.value_clip_epsilon,
                    )
                    value_loss_1 = F.mse_loss(values_new, returns)
                    value_loss_2 = F.mse_loss(values_clipped, returns)
                    value_loss = torch.max(value_loss_1, value_loss_2)
                else:
                    value_loss = F.mse_loss(values_new, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Optimize
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Track statistics
                with torch.no_grad():
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_epsilon)
                        .float()
                        .mean()
                        .item()
                    )

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["clip_fraction"] += clip_fraction
                num_updates += 1

        # Average statistics
        for key in stats:
            stats[key] /= max(num_updates, 1)

        # Clear buffer
        self.buffer.reset()
        self.training_step += 1

        return stats

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Reward tensor [T]
            values: Value tensor [T]
            dones: Done flags [T]
            last_value: Value of final state

        Returns:
            Tuple of (advantages, returns)
        """
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
        """Save model checkpoint."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "config": self.config,
                "training_step": self.training_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
