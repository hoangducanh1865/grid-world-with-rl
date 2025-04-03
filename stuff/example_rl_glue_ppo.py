import numpy as np
import torch.nn as nn
import gym

class ActorCritic(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        
        return action_probs, state_value
    
def compute_advantages(rewards, values, gamma=0.99, lamda=0.95):
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = delta + gamma * lamda * last_advantage
        last_advantage = advantages[t]
    
    return advantages

def train_ppo():
    pass


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    