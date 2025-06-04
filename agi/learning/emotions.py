import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from utilities.torch.gradients import check_gradients


class PAD(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.fc(x)
        x = torch.tanh(x)
        return x


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


class AlphaSR:
    def __init__(
        self,
        policy_value_net,
        intrinsics,
        device,
        c_puct=1.0,
        discount=0.99,
        decay_rate=0.9,
        max_visits=10000,
        min_visits=1000,
    ):
        self.intrinsics = intrinsics
        self.device = device
        self.c_puct = c_puct
        self.discount = discount
        self.decay_rate = decay_rate
        self.max_visits = max_visits
        self.min_visits = min_visits

        self.policy_value_net = policy_value_net

        self.Q = {}
        self.N = {}
        self.Ns = {}
        self.P = {}

    def hash_state(self, state):
        return tuple(state.int().tolist())

    def forward(self, state, reward, optimizer):
        state = state.to(self.device)
        policy, value = self.simulate(state, reward)
        governance_output = self.get_governance(state)
        loss = 0

        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device)

        governance_tensor = torch.tensor(
            [governance_output[instinct] for instinct in self.intrinsics],
            dtype=torch.float32,
        ).to(self.device)

        governance_loss = F.mse_loss(policy, governance_tensor)
        reward_loss = F.mse_loss(value, reward_tensor)
        loss = governance_loss + reward_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return policy

    def ucb_score(self, hashkey, intrinsic):
        q = self.Q.get((hashkey, intrinsic), 0.0)
        p = self.P.get(hashkey, {}).get(intrinsic, 0.0)
        n_s = self.Ns.get(hashkey, 0)
        n_sa = self.N.get((hashkey, intrinsic), 0)

        return q + self.c_puct * p * math.sqrt(n_s) / (1 + n_sa)

    def back_prop(self, hashkey, intrinsic, value):
        self.Ns[hashkey] = min(self.Ns.get(hashkey, 0) + 1, self.max_visits)
        self.N[(hashkey, intrinsic)] = min(
            self.N.get((hashkey, intrinsic), 0) + 1, self.max_visits
        )

        q_old = self.Q.get((hashkey, intrinsic), 0.0)
        n_sa = self.N[(hashkey, intrinsic)]
        self.Q[(hashkey, intrinsic)] = (q_old * (n_sa - 1) + value) / n_sa

    def simulate(self, state, reward):
        hashkey = self.hash_state(state)
        policy, value = self.policy_value_net(state)

        if hashkey not in self.P:
            policy_dict = {}
            for intrinsic in self.intrinsics:
                policy_dict[intrinsic] = 1 / len(self.intrinsics)
            self.P[hashkey] = policy_dict
            self.Ns[hashkey] = 0
            for intrinsic in self.intrinsics:
                self.Q[(hashkey, intrinsic)] = 0.0
                self.N[(hashkey, intrinsic)] = 0
            return policy, value

        best_action = None
        best_ucb = -float("inf")
        for intrinsic in self.intrinsics:
            ucb_val = self.ucb_score(hashkey, intrinsic)
            if ucb_val > best_ucb:
                best_ucb = ucb_val
                best_action = intrinsic

        reward = reward + self.discount * value.item()
        self.back_prop(hashkey, best_action, reward)

        return policy, value

    def get_governance(self, state):
        hashkey = self.hash_state(state)

        if hashkey not in self.Ns:
            return dict(
                zip(
                    self.intrinsics,
                    [1.0 / len(self.intrinsics) for _ in self.intrinsics],
                )
            )

        visits = {
            intrinsic: self.N.get((hashkey, intrinsic), 0)
            for intrinsic in self.P.get(hashkey, {})
        }
        total_visits = sum(visits.values())
        if total_visits > 0:
            distribution = {
                intrinsic: count / total_visits for intrinsic, count in visits.items()
            }
        else:
            distribution = {
                intrinsic: 1.0 / len(self.P.get(hashkey, {}))
                for intrinsic in self.P.get(hashkey, {})
            }

        return distribution

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


class TitansAlphaSR:
    def __init__(self, titans_model, intrinsics, movement_symbols, device, **kwargs):
        self.titans_model = titans_model
        self.intrinsics = intrinsics
        self.movement_symbols = movement_symbols
        self.device = device
        self.c_puct = kwargs.get('c_puct', 1.0)
        self.discount = kwargs.get('discount', 0.99)
        self.decay_rate = kwargs.get('decay_rate', 0.9)
        self.max_visits = kwargs.get('max_visits', 10000)
        self.min_visits = kwargs.get('min_visits', 1000)
        
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

    def ucb_score(self, hashkey, intrinsic):
        q = self.Q.get((hashkey, intrinsic), 0.0)
        p = self.P.get(hashkey, {}).get(intrinsic, 0.0)
        n_s = self.Ns.get(hashkey, 0)
        n_sa = self.N.get((hashkey, intrinsic), 0)
        return q + self.c_puct * p * math.sqrt(n_s) / (1 + n_sa)

    def back_prop(self, hashkey, intrinsic, value):
        self.Ns[hashkey] = min(self.Ns.get(hashkey, 0) + 1, self.max_visits)
        self.N[(hashkey, intrinsic)] = min(self.N.get((hashkey, intrinsic), 0) + 1, self.max_visits)
        q_old = self.Q.get((hashkey, intrinsic), 0.0)
        n_sa = self.N[(hashkey, intrinsic)]
        self.Q[(hashkey, intrinsic)] = (q_old * (n_sa - 1) + value) / n_sa

    def simulate(self, state, ctx, reward):
        """Run MCTS simulation using both state and context"""
        hashkey = self.hash_state(state)
        
        # Use context for Titans model forward pass
        policy, value, emotions = self.titans_model(ctx)
        
        if hashkey not in self.P:
            # Initialize policy for this state
            policy_dict = {intrinsic: 1 / len(self.intrinsics) for intrinsic in self.intrinsics}
            self.P[hashkey] = policy_dict
            self.Ns[hashkey] = 0
            for intrinsic in self.intrinsics:
                self.Q[(hashkey, intrinsic)] = 0.0
                self.N[(hashkey, intrinsic)] = 0
            return policy, value, emotions

        # MCTS action selection
        best_action = max(self.intrinsics, key=lambda i: self.ucb_score(hashkey, i))
        reward = reward + self.discount * value.item()
        self.back_prop(hashkey, best_action, reward)
        
        return policy, value, emotions

    def get_governance(self, state):
        hashkey = self.hash_state(state)
        if hashkey not in self.Ns:
            return {intrinsic: 1.0 / len(self.intrinsics) for intrinsic in self.intrinsics}
        
        visits = {intrinsic: self.N.get((hashkey, intrinsic), 0) for intrinsic in self.P.get(hashkey, {})}
        total_visits = sum(visits.values())
        if total_visits > 0:
            return {intrinsic: count / total_visits for intrinsic, count in visits.items()}
        else:
            return {intrinsic: 1.0 / len(self.P.get(hashkey, {})) for intrinsic in self.P.get(hashkey, {})}

    def calculate_reward(self, emotions, emotion_guidance=None):
        """Calculate reward from emotions and guidance"""
        # Base emotion reward (magnitude + valence)
        emotion_magnitude = torch.norm(emotions, p=2).item()
        emotion_valence = torch.tanh(emotions).mean().item()
        base_reward = emotion_magnitude * 0.1 + emotion_valence * 0.5
        
        # Add emotion guidance reward if available
        guidance_reward = 0.0
        if emotion_guidance is not None:
            # Cosine similarity between current emotions and guidance
            guidance_reward = F.cosine_similarity(
                emotions.flatten(), 
                emotion_guidance.flatten(), 
                dim=0
            ).item() * 0.5
        
        return base_reward + guidance_reward

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

    def forward(self, state, ctx, emotion_guidance=None, optimizer=None):
        """Main forward pass with proper context and state handling"""
        # Ensure tensors are on correct device
        state = state.to(self.device)
        ctx = ctx.to(self.device)
        
        if emotion_guidance is not None:
            emotion_guidance = emotion_guidance.to(self.device)
        
        # Calculate reward from current state
        # For initial reward, we can use state information
        state_reward = torch.tanh(state).mean().item() * 0.1
        
        # Run MCTS simulation with context
        policy, value, emotions = self.simulate(state, ctx, state_reward)
        
        # Calculate comprehensive reward
        reward = self.calculate_reward(emotions, emotion_guidance)
        
        # Get governance from MCTS
        governance_output = self.get_governance(state)
        
        # Split policy into governance and movement parts
        # Policy from Titans is [intrinsic_actions..., movement_actions...]
        governance_policy = policy[:self.intrinsic_dim] if self.intrinsic_dim > 0 else torch.tensor([], device=self.device)
        movement_policy = policy[self.intrinsic_dim:self.intrinsic_dim + self.movement_dim]
        
        # Ensure we have the right number of movement actions
        if movement_policy.shape[0] != self.movement_dim:
            # Fallback: create uniform distribution
            movement_policy = torch.ones(self.movement_dim, device=self.device) / self.movement_dim
        
        # Learning and optimization
        if optimizer is not None:
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
            
            # Ensure value tensor has correct shape for loss computation
            if value.dim() > 1:
                value_reshaped = value.view(-1)
            else:
                value_reshaped = value
            
            # Main loss: value prediction - ensure tensors have matching shapes
            if value_reshaped.shape != reward_tensor.shape:
                if value_reshaped.numel() == 1 and reward_tensor.numel() == 1:
                    value_reshaped = value_reshaped.view(reward_tensor.shape)
                else:
                    # Fallback: use scalar values
                    value_reshaped = torch.tensor([value_reshaped.mean().item()], device=self.device)
            
            loss = F.mse_loss(value_reshaped, reward_tensor)
            
            # Add emotion guidance loss if available
            if emotion_guidance is not None:
                emotion_loss = F.mse_loss(emotions, emotion_guidance)
                loss += emotion_loss * 0.5  # Scale emotion loss
            
            # Add governance loss (align with MCTS visits)
            if self.intrinsic_dim > 0 and governance_policy.numel() > 0:
                governance_tensor = torch.tensor(
                    [governance_output.get(intrinsic, 0.0) for intrinsic in self.intrinsics],
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Ensure governance tensors have matching shapes
                if governance_policy.shape != governance_tensor.shape:
                    if governance_policy.numel() == governance_tensor.numel():
                        governance_policy = governance_policy.view(governance_tensor.shape)
                    else:
                        # Skip governance loss if shapes are incompatible
                        pass
                else:
                    governance_loss = F.mse_loss(governance_policy, governance_tensor)
                    loss += governance_loss * 0.3  # Scale governance loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return {
            'movement_logits': movement_policy,
            'governance_policy': governance_policy,
            'emotions': emotions,
            'value': value,
            'reward': reward,
            'governance_output': governance_output
        }
