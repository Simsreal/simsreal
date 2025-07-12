import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        state_dim,
        policy_dim,
        hidden_dim,
    ):
        super().__init__()
        self.fc0 = nn.Linear(state_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, policy_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc0(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = torch.tanh(self.value_head(x))
        return policy, value


class TitansAlphaSR:
    def __init__(self, titans_model, intrinsics, movement_symbols, device, **kwargs):
        self.titans_model = titans_model
        self.intrinsics = intrinsics
        self.movement_symbols = movement_symbols
        self.device = device
        self.c_puct = kwargs.get("c_puct", 1.0)
        self.discount = kwargs.get("discount", 0.99)
        self.decay_rate = kwargs.get("decay_rate", 0.9)
        self.max_visits = kwargs.get("max_visits", 10000)
        self.min_visits = kwargs.get("min_visits", 1000)

        # Policy dimensions
        self.intrinsic_dim = len(intrinsics)
        self.movement_dim = len(movement_symbols)
        self.total_policy_dim = self.intrinsic_dim + self.movement_dim

        self.Q = {}
        self.N = {}
        self.Ns = {}
        self.P = {}

    def hash_state(self, state):
        """Create a hash from state tensor for MCTS lookup"""
        if isinstance(state, torch.Tensor):
            # Quantize to reduce precision for consistent hashing
            quantized = (state * 1000).int()
            return tuple(quantized.cpu().numpy().flatten().tolist())
        return tuple(state)

    def ucb_score(self, hashkey, action_idx):
        """Calculate UCB score for action selection"""
        q = self.Q.get((hashkey, action_idx), 0.0)
        p = self.P.get(hashkey, {}).get(action_idx, 1.0 / self.movement_dim)
        n_s = self.Ns.get(hashkey, 0)
        n_sa = self.N.get((hashkey, action_idx), 0)
        return q + self.c_puct * p * math.sqrt(n_s) / (1 + n_sa)

    def back_prop(self, hashkey, action_idx, value):
        """Backpropagate value through MCTS tree"""
        self.Ns[hashkey] = min(self.Ns.get(hashkey, 0) + 1, self.max_visits)
        self.N[(hashkey, action_idx)] = min(
            self.N.get((hashkey, action_idx), 0) + 1, self.max_visits
        )
        q_old = self.Q.get((hashkey, action_idx), 0.0)
        n_sa = self.N[(hashkey, action_idx)]
        self.Q[(hashkey, action_idx)] = (q_old * (n_sa - 1) + value) / n_sa

    def simulate_with_raycast(self, state, ctx, simulated_reward, action_rewards):
        """Simplified MCTS simulation using raycast-based action rewards"""
        hashkey = self.hash_state(state)
        
        # Get policy from Titans model (simplified - no complex emotions)
        policy, value = self.titans_model(ctx)
        
        if hashkey not in self.P:
            # Initialize with action rewards as priors
            policy_dict = {}
            total_reward = sum(max(0, r) for r in action_rewards.values())
            if total_reward > 0:
                for i, action in enumerate(self.movement_symbols):
                    reward = action_rewards.get(action, 0.0)
                    policy_dict[i] = max(0, reward) / total_reward
            else:
                for i in range(self.movement_dim):
                    policy_dict[i] = 1.0 / self.movement_dim
            
            self.P[hashkey] = policy_dict
            self.Ns[hashkey] = 0
            for i in range(self.movement_dim):
                self.Q[(hashkey, i)] = action_rewards.get(self.movement_symbols[i], 0.0)
                self.N[(hashkey, i)] = 0
            
            return policy, value
        
        # MCTS action selection based on UCB scores
        best_action = max(range(self.movement_dim), 
                         key=lambda i: self.ucb_score(hashkey, i))
        
        # Use simulated reward for backpropagation
        reward = simulated_reward + self.discount * value.item()
        self.back_prop(hashkey, best_action, reward)
        
        return policy, value

    def get_action_visits(self, state):
        """Get visit counts for actions to guide policy"""
        hashkey = self.hash_state(state)
        if hashkey not in self.Ns:
            return {i: 1 for i in range(self.movement_dim)}
        
        visits = {}
        for i in range(self.movement_dim):
            visits[i] = self.N.get((hashkey, i), 0)
        
        return visits

    def forward(self, state, ctx, emotion_guidance=None, optimizer=None, 
                simulated_reward=0.0, action_rewards=None):
        """Simplified forward pass with raycast-based MCTS"""
        if action_rewards is None:
            action_rewards = {}
        
        # Ensure tensors are on correct device and have correct shapes
        state = state.to(self.device)
        ctx = ctx.to(self.device)
        
        # Ensure state is 1D for hashing and processing
        if state.dim() > 1:
            state = state.flatten()
        
        # Run simplified MCTS simulation
        policy, value = self.simulate_with_raycast(
            state, ctx, simulated_reward, action_rewards
        )
        
        # Get action visit counts for movement policy
        action_visits = self.get_action_visits(state)
        total_visits = sum(action_visits.values())
        
        if total_visits > 0:
            # Create movement policy based on visit counts
            movement_policy = torch.tensor([
                action_visits[i] / total_visits for i in range(self.movement_dim)
            ], dtype=torch.float32, device=self.device)
        else:
            # Fallback to uniform distribution
            movement_policy = torch.ones(self.movement_dim, device=self.device) / self.movement_dim
        
        # Split policy for governance if needed
        governance_policy = (
            policy[:self.intrinsic_dim] if self.intrinsic_dim > 0 
            else torch.tensor([], device=self.device)
        )
        
        # Calculate simplified reward
        reward = simulated_reward
        
        # Add emotion guidance reward if available (simplified)
        if emotion_guidance is not None and isinstance(emotion_guidance, torch.Tensor):
            if emotion_guidance.numel() == 1:
                # Simple scalar emotion guidance
                emotion_reward = emotion_guidance.item() * 0.3
                reward += emotion_reward
            elif emotion_guidance.numel() > 1:
                # Multi-dimensional emotion guidance - take mean
                emotion_reward = emotion_guidance.mean().item() * 0.3
                reward += emotion_reward
        
        # Learning and optimization
        if optimizer is not None:
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
            
            # Value prediction loss - ensure proper shapes
            if value.dim() > 1:
                value_reshaped = value.view(-1)
            else:
                value_reshaped = value
            
            # Ensure both tensors are scalars or have compatible shapes
            if value_reshaped.numel() == 1 and reward_tensor.numel() == 1:
                if value_reshaped.dim() != reward_tensor.dim():
                    value_reshaped = value_reshaped.view(reward_tensor.shape)
                loss = F.mse_loss(value_reshaped, reward_tensor)
            else:
                # Fallback: convert to scalars
                value_scalar = value_reshaped.mean() if value_reshaped.numel() > 1 else value_reshaped
                reward_scalar = reward_tensor.mean() if reward_tensor.numel() > 1 else reward_tensor
                loss = F.mse_loss(value_scalar.view(1), reward_scalar.view(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return {
            "movement_logits": movement_policy,
            "governance_policy": governance_policy,
            "value": value,
            "reward": reward,
            "action_visits": action_visits,
        }

    def decay_visits(self):
        for key in self.Ns:
            self.Ns[key] = int(self.Ns[key] * self.decay_rate)
        for key in self.N:
            self.N[key] = int(self.N[key] * self.decay_rate)

    def prune_states(self):
        keys_to_remove = [key for key in self.Ns if self.Ns[key] < self.min_visits]
        for key in keys_to_remove:
            del self.Ns[key]
            if key in self.P:
                for intrinsic in self.P[key]:
                    if (key, intrinsic) in self.N:
                        del self.N[(key, intrinsic)]
                    if (key, intrinsic) in self.Q:
                        del self.Q[(key, intrinsic)]
                del self.P[key]
