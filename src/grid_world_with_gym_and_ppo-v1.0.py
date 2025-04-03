import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical # De lam viec voi phan pho xac suat roi rac
from collections import deque # double-ennded queue: dung de luu tru bo nho kinh nghiem (replay buffer) voi kich thuoc toi da co dinh
import time

class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=7, goal_pos=(6,6)):
        super(GridWorldEnv, self).__init__() # Goi ham khoi tao lop cha
        self.grid_size = grid_size
        self.goal_position = goal_pos
        self.agent_position = [0, 0]
        
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(4)  # 0:up, 1:down, 2:left, 3:right
        self.observation_space = gym.spaces.Box(
            low=0, high=grid_size-1, shape=(2,), dtype=np.int32) # Box dai dien cho khong gian lien tuc hoac roi rac da chieu
        
    # Reset moi truong ve trang thai ban dau
    def reset(self):
        self.agent_position = [0, 0]
        return np.array(self.agent_position, dtype=np.int32)
    
    def step(self, action):
        x, y = self.agent_position
        
        # Move agent
        if action == 0:    # up
            y = max(0, y - 1)
        elif action == 1:  # down
            y = min(self.grid_size - 1, y + 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(self.grid_size - 1, x + 1)
        
        self.agent_position = [x, y]
        
        # Check if reached goal
        terminated = (x == self.goal_position[0]) and (y == self.goal_position[1])
        reward = 1.0 if terminated else -0.1  # Small negative reward for each step
        
        return np.array(self.agent_position, dtype=np.int32), reward, terminated, {} # {} la mot dictionary thong tin bo sung, o day la trong
    
    def render(self, mode='human'):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.goal_position[1]][self.goal_position[0]] = "G"
        grid[self.agent_position[1]][self.agent_position[0]] = "A"
        
        print("\n".join([" ".join(row) for row in grid]))
        print("-" * (self.grid_size * 2))
        time.sleep(0.01)

class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PPONetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Tao mot lop fully connected (Linear) dau tien
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, 
                 clip_ratio=0.2, ppo_epochs=4, batch_size=64):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.policy = PPONetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = deque(maxlen=2048)
        
    def get_action(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def remember(self, state, action, log_prob, reward, done):
        self.memory.append([state, action, log_prob, reward, done])
    
    def update(self):
        # Convert memory to numpy arrays
        states = np.array([x[0] for x in self.memory])
        actions = np.array([x[1] for x in self.memory])
        old_log_probs = np.array([x[2] for x in self.memory])
        rewards = np.array([x[3] for x in self.memory])
        dones = np.array([x[4] for x in self.memory])
        
        # Calculate discounted returns and advantages
        returns = []
        advantages = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # Optimize policy for K epochs
        for _ in range(self.ppo_epochs):
            for idx in range(0, len(states), self.batch_size):
                batch_states = states[idx:idx+self.batch_size]
                batch_actions = actions[idx:idx+self.batch_size]
                batch_old_log_probs = old_log_probs[idx:idx+self.batch_size]
                batch_returns = returns[idx:idx+self.batch_size]
                
                # Get new action probabilities and state values
                action_probs, state_values = self.policy(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calculate ratio (pi_theta / pi_theta_old)
                ratio = (new_log_probs - batch_old_log_probs).exp()
                
                # Calculate surrogate loss
                advantages = batch_returns - state_values.detach()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss = (advantages ** 2).mean()
                
                # Total loss
                loss = actor_loss + 0.5 * critic_loss
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Clear memory
        self.memory.clear()

def train():
    env = GridWorldEnv()
    agent = PPOAgent(state_size=2, action_size=4)
    
    max_episodes = 10000
    print_freq = 50
    
    for episode in range(1, max_episodes+1):
        state = env.reset()
        episode_reward = 0
        done = False
        
        env.render()
        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, log_prob, reward, done)
            
            state = next_state
            episode_reward += reward
            
            env.render()
            
            if done:
                break
        
        # Update policy
        if len(agent.memory) >= agent.batch_size:
            agent.update()
        
        # Print progress
        if episode % print_freq == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

if __name__ == "__main__":
    train()