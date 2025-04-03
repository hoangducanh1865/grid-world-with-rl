'''
Use RL-Glue
'''

from __future__ import print_function
from abc import ABCMeta, abstractmethod
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time 

class BaseAgent:
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass
    
    @abstractmethod
    def agent_init(self, agent_info={}):
        pass
    
    @abstractmethod
    def agent_start(self, observation):
        pass
    
    @abstractmethod
    def agent_step(self, reward, observation):
        pass
    
    @abstractmethod
    def agent_end(self, reward):
        pass
    
    def agent_cleanup(self):
        pass
    
    def agent_message(self, message):
        pass
    
class PolicyNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, hidden_size=16):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)
    
class REINFORCEAgent(BaseAgent):
    
    def __init__(self):
        super().__init__()
    
    def agent_init(self, agent_info={}):
        # Initialize parameters
        self.gamma = agent_info.get("gamma")
        self.learning_rate = agent_info.get("learning_rate")
        self.state_size = agent_info.get("state_size")
        self.action_size = agent_info.get("action_size")
        
        # Initialize policy network
        self.policy_network = PolicyNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        
        # Store (state, action, reward) tuples
        self.trajectory = []
    
    def agent_start(self, observation):
        state = torch.tensor(observation, dtype=torch.float32)
        action_probs = self.policy_network(state)
        action = np.random.choice(self.action_size, p=action_probs.detach().numpy())  
        self.trajectory.append((state, action, 0)) # At the start, there is no reward for any action
        
        return action
    
    def agent_step(self, reward, observation):
        state = torch.tensor(observation, dtype=torch.float32)
        action_probs = self.policy_network(state)
        action = np.random.choice(self.action_size, p=action_probs.detach().numpy())
        
        self.trajectory[-1] = (self.trajectory[-1][0], self.trajectory[-1][1], reward)
        self.trajectory.append((state, action, 0))
        
        return action
    
    def agent_end(self, reward):
        self.trajectory[-1] = (self.trajectory[-1][0], self.trajectory[-1][1], reward)
        self._update_policy()
        self.trajectory = []
    
    def _update_policy(self):
        returns = []
        G = 0
        for _, _, reward in reversed(self.trajectory):
            G = reward + self.gamma * G
            returns.insert(0, G)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # Nomalization to range [0, 1]
        
        policy_loss = []
        for (state, action, _), G in zip(self.trajectory, returns):
            action_probs = self.policy_network(state)
            log_prob = torch.log(action_probs[action]) # Each log_prob is a tensor
            policy_loss.append(-log_prob * G)
        
        loss = torch.stack(policy_loss).sum() # loss here is a tensor of a single number,
                                              # so that we can call function backward() to calculate gradient
        self.optimier.zero_grad()
        loss.backward()
        self.optimier.step()

class BaseEnvironment:
    __metaclass__ = ABCMeta
    
    def __init__(self):
        reward = None
        state = None 
        termination = None
        self.reward_state_term = (reward, state, termination)
    
    @abstractmethod
    def env_init(self, env_info={}):
        pass
    
    @abstractmethod
    def env_start(self):
        pass
    
    @abstractmethod
    def env_step(self, action):
        pass
    
    def env_cleanup(self):
        pass
    
    def env_message(self, message):
        pass
    
class GridWorldEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.grid_size = -1
        self.agent_position = (-1, -1)
        self.goal_position = (-1, -1)
    
    def env_init(self, env_info={}):
        self.grid_size = env_info.get("grid_size")
        self.goal_position = tuple(env_info.get("goal_position"))
        
    def env_start(self):
        self.agent_position = (0, 0)
        
        return self.agent_position    
    
    def env_step(self, action):
        x, y = self.agent_position
        
        if action == 0:
            y = max(0, y - 1)
        elif action == 1:
            y = min(self.grid_size - 1, y + 1)
        elif action == 2:
            x = max(0, x - 1)
        elif action == 3:
            x = min(self.grid_size - 1, x + 1)
        
        self.agent_position = (x, y)
        
        if self.agent_position == self.goal_position:
            terminated = True
            reward = 1.0
        else:
            terminated = False
            reward = -1.0
        
        return (reward, self.agent_position, terminated)
    
    def print_grid(self):
        """Prints the grid showing the agent's movement."""
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.goal_position[1]][self.goal_position[0]] = "G"
        grid[self.agent_position[1]][self.agent_position[0]] = "A"

        print("\n".join([" ".join(row) for row in grid]))
        print("-" * (self.grid_size * 2))
    
    
    
if __name__ == "__main__":
    env = GridWorldEnvironment()
    env_info = {"grid_size": 7, "goal_position": (6, 6)}
    env.env_init(env_info)
    
    agent = REINFORCEAgent()
    agent_info = {"gamma": 0.99, "learning_rate": 0.01, "state_size": 2, "action_size": 4}
    agent.agent_init(agent_info)
    
    for episode in range(10000):
        state = env.env_start()
        action = agent.agent_start(state)
        
        env.print_grid()
        time.sleep(0.02)
        while True:
            reward, next_state, terminated = env.env_step(action)
            env.print_grid()
            time.sleep(0.02)
            if terminated:
                agent.agent_end(reward)
                break
            action = agent.agent_step(reward, next_state)
    