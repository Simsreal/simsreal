import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from utilities.pytorch_utils.gradients import check_gradients


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
