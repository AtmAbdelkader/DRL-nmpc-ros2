#!/usr/bin/env python3 

import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0



import torch
import torch.nn as nn
import torch.optim as optim
# Assume a BNN library is imported, e.g., using a custom BNN layer
# from bnn_library import BNNLinear, BNNModule

# --- Define the BNN Actor (Policy) and Critic (Q-Networks) ---
class BNNPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(BNNPolicyNetwork, self).__init__()
        
        # BNN layers to represent the distribution over weights
        self.bnn_l1 = BNNLinear(state_dim, hidden_dim)
        self.bnn_l2 = BNNLinear(hidden_dim, hidden_dim)
        
        # Output layers for the mean and log standard deviation
        self.mean_layer = BNNLinear(hidden_dim, action_dim)
        self.log_std_layer = BNNLinear(hidden_dim, action_dim)

    def forward(self, state):
        # Forward pass through the BNN layers
        x = torch.relu(self.bnn_l1(state))
        x = torch.relu(self.bnn_l2(x))
        
        # Sample from the weight distributions to get a network realization
        # This is where the BNN magic happens
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        # Clamp log_std to a reasonable range
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        
        return mean, std, self.kl_divergence() # BNN returns a KL term

    def kl_divergence(self):
        # A method to calculate the total KL divergence for all BNN layers
        return self.bnn_l1.kl_divergence() + self.bnn_l2.kl_divergence() + \
               self.mean_layer.kl_divergence() + self.log_std_layer.kl_divergence()

# BNN Critic Network (two Q-networks for SAC)
class BNNQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(BNNQNetwork, self).__init__()

        self.bnn_l1 = BNNLinear(state_dim + action_dim, hidden_dim)
        self.bnn_l2 = BNNLinear(hidden_dim, hidden_dim)
        self.bnn_l3 = BNNLinear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        x = torch.relu(self.bnn_l1(sa))
        x = torch.relu(self.bnn_l2(x))
        q_value = self.bnn_l3(x)
        return q_value, self.kl_divergence()

    def kl_divergence(self):
        return self.bnn_l1.kl_divergence() + self.bnn_l2.kl_divergence() + \
               self.bnn_l3.kl_divergence()

# --- The SAC BNN Agent Class ---
class SAC_BNN_Agent:
    def __init__(self, state_dim, action_dim, lr, gamma, tau, alpha):
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha # Entropy temperature parameter
        
        # Networks
        self.policy = BNNPolicyNetwork(state_dim, action_dim, 256)
        self.q_network1 = BNNQNetwork(state_dim, action_dim, 256)
        self.q_network2 = BNNQNetwork(state_dim, action_dim, 256)
        
        # Target networks (for stability)
        self.target_q_network1 = BNNQNetwork(state_dim, action_dim, 256)
        self.target_q_network2 = BNNQNetwork(state_dim, action_dim, 256)
        self.target_q_network1.load_state_dict(self.q_network1.state_dict())
        self.target_q_network2.load_state_dict(self.q_network2.state_dict())

        # Optimizers (note: they optimize the variational parameters)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q_network1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q_network2.parameters(), lr=lr)

        # Replay buffer (not shown, but essential)
        self.replay_buffer = ReplayBuffer(...)

    def select_action(self, state):
        # Sample an action from the policy distribution
        mean, std, _ = self.policy(state)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        return action.cpu().numpy()

    def update_networks(self, batch_size):
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # 1. Update the Q-networks (Critics)
        # Calculate the target Q-value
        with torch.no_grad():
            next_mean, next_std, next_kl_policy = self.policy(next_states)
            next_normal = torch.distributions.Normal(next_mean, next_std)
            next_actions = next_normal.rsample()
            log_prob = next_normal.log_prob(next_actions)
            
            target_q1, _ = self.target_q_network1(next_states, next_actions)
            target_q2, _ = self.target_q_network2(next_states, next_actions)
            min_target_q = torch.min(target_q1, target_q2)
            
            # The entropy term is crucial for SAC
            target_q = rewards + self.gamma * (1 - dones) * (min_target_q - self.alpha * log_prob)
        
        # Get current Q-values from the BNNs
        current_q1, kl_q1 = self.q_network1(states, actions)
        current_q2, kl_q2 = self.q_network2(states, actions)

        # Calculate the Q-network loss
        # Note the addition of the KL divergence term
        q1_loss = ( (current_q1 - target_q).pow(2) ).mean() + kl_q1
        q2_loss = ( (current_q2 - target_q).pow(2) ).mean() + kl_q2
        
        # Optimize Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # 2. Update the Policy Network (Actor)
        mean, std, kl_policy = self.policy(states)
        normal = torch.distributions.Normal(mean, std)
        reparam_actions = normal.rsample() # Reparameterization trick
        log_prob = normal.log_prob(reparam_actions)

        # Get Q-values for the new actions from the current Q-networks
        q1_eval, _ = self.q_network1(states, reparam_actions)
        q2_eval, _ = self.q_network2(states, reparam_actions)
        min_q_eval = torch.min(q1_eval, q2_eval)

        # Calculate the policy loss
        # Maximize (min_Q_eval - alpha * log_prob)
        # Minimize - (min_Q_eval - alpha * log_prob)
        # Add the KL divergence term
        policy_loss = (self.alpha * log_prob - min_q_eval).mean() + kl_policy
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 3. Soft update of the target networks
        self.soft_update_targets()

    def soft_update_targets(self):
        # Standard soft update for SAC
        for target_param, param in zip(self.target_q_network1.parameters(), self.q_network1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_q_network2.parameters(), self.q_network2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
