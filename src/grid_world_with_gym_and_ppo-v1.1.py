"""
import torch.optim as optim
from torch.distributions import Categorical
import torch
import numpy as np
import gym
import torch.nn as nn

# Environment for the grid world
class GridWorldEnv(gym.Env):
    def __init__(self, grid_size, goal_position):
        '''
        This function is used to initialize the first state of the grid world.
        Args:
            env_info
        '''
        # Init the super class
        super(GridWorldEnv, self).__init__()
        
        ''' 
        Attributes of the class is: 
            +) grid_size
            +) agent_position
            +) goal_position
            
            +) action_space
            +) state_space
            (We will take these information from gym)
        '''
        self.grid_size = grid_size
        self.agent_position = [0, 0]
        self.goal_position = goal_position
        
        self.action_space = gym.spaces.Discrete(4) # 0: up, 1: down, 2: left, 3: right
        self.state_space = gym.spaces.Box(
            low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32
        )
    
    def reset(self):
        '''
        This function is used to reset the agent_position to start position.
        Returns:
            the start position
        '''
        self.agent_position = [0, 0]
        return np.array(self.agent_position, dtype=np.int32)
    
    def step(self, action):
        '''
        This function is used to change the agent_position to the next position, with taking the action 'action'.
        Args:
            action
        Returns:
            an array of: agent_position, reward, termination
        '''
        if action == 0: # up
            self.agent_position[1] = max(0, self.agent_position[1] + 1)
        elif action == 1: # down
            self.agent_position[1] = min(self.grid_size - 1, self.agent_position[1] + 1)
        elif action == 2: # left
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 3: # right
            self.agent_position[0] = min(self.grid_size - 1, self.agent_position[0] + 1)
            
        # Check if the agent has reached the gaol
        terminated = (self.agent_position == self.goal_position)
        reward = 1.0 if terminated else -0.1 
        
        return np.array(self.agent_position, dtype=np.int32), reward, terminated
        
    
    def render(self):
        '''
        This function is used to print the current state of the grid world.
        '''
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.goal_position[1]][self.goal_position[0]] = "G"
        grid[self.agent_position[1]][self.agent_position[0]] = "A"
        
        print("\n".join([" ".join(row) for row in grid]))
        print("-" * (self.grid_size * 2))
        

class PPONetwork(nn.Module):
    
    def __init__(self, state_size, action_size, hidden_size):
        '''
        This function is used to initialize a MLP as PPONetwork.
        Args:
            state_size
            hidden_size
            action_size
        '''
        super(PPONetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)


    def forward(self, x):
        '''
        This function is used to pass the input state x through the PPONetwork,
        and then output a vector of action probabilities, and state value.
        Args:
            x
        Returns:
            action_probs
            state_value
        '''
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        
        return action_probs, state_value
        

class PPOAgent(nn.Module):
    def __init__(self, gamma, clip_ratio, ppo_epochs, batch_size, state_size, action_size, learning_rate=3e-4):
        '''
        This function is used to initailize some parameter of the model,
        inlcude: 
        Args: 
            gamma
            clip_ratio
            ppo_epochs
            batch_size
            state_size
            action_size
            learning_rate
        '''
        super(PPOAgent, self).__init__()
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.policy = PPONetwork(state_size, action_size, hidden_size=64)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.memory = []
    
    def get_action(self, state):
        '''
        This function is used to pass a input state x through the PPONetwork,
        and then receive an action, and the probability to take this action.
        Args:
            state
        Returns:
            action
            log_prob (of that action)
        '''
        state = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def remember(self, state, action, log_prob, reward, done):
        '''
        This function is used to store experiences in memory.
        Args:
            state: Current state
            action: Action taken
            log_prob: Log probability of the action
            reward: Reward received
            done: Whether the episode is done
        '''
        self.memory.append((state, action, log_prob, reward, done))

    
    def update(self):
        '''
        This function is used to calculate the returns, ..., then update the policy.
        Args:
        '''
        
        states, actions, log_probs, rewards, dones = zip(*self.memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Compute discounted returns
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Update policy
        for _ in range(self.ppo_epochs):
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Calculate advantages
            advantages = returns - state_values.detach().squeeze()

            # Calculate ratio
            ratio = (new_log_probs - log_probs).exp()

            # Surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = (returns - state_values.squeeze()).pow(2).mean()

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear memory
        self.memory = []

def train():
    env = GridWorldEnv(grid_size=10, goal_position=[9, 9])
    agent = PPOAgent(gamma=0.99, clip_ratio=0.2, ppo_epochs=4, batch_size=64, state_size=2, action_size=4)
    
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, log_prob, reward, done)
            state = next_state
            episode_reward += reward
            
        agent.update()
    
if __name__ == "__main__":
    train()
"""   
    

import time
from torch.distributions import Categorical
import torch
import numpy as np
import gym
import torch.nn as nn
import torch.optim as optim

# Environment for the grid world
class GridWorldEnv(gym.Env):
    def __init__(self, grid_size, goal_position, walls=None):
        '''
        This function is used to initialize the first state of the grid world.
        Args:
            grid_size: Size of the grid
            goal_position: Position of the goal
            walls: List of wall positions (tuples)
        '''
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.agent_position = [0, 0]
        self.goal_position = goal_position
        self.walls = walls if walls else []  # List of wall positions

        self.action_space = gym.spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.state_space = gym.spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32
        )

    def reset(self):
        '''
        This function is used to reset the agent_position to start position.
        Returns:
            The start position
        '''
        self.agent_position = [0, 0]
        return np.array(self.agent_position, dtype=np.int32)

    def step(self, action):
        '''
        This function is used to change the agent_position to the next position, with taking the action 'action'.
        Args:
            action: The action to take
        Returns:
            An array of: agent_position, reward, termination, info
        '''
        next_position = self.agent_position.copy()

        if action == 0:  # up
            next_position[1] = max(0, self.agent_position[1] - 1)
        elif action == 1:  # down
            next_position[1] = min(self.grid_size - 1, self.agent_position[1] + 1)
        elif action == 2:  # left
            next_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 3:  # right
            next_position[0] = min(self.grid_size - 1, self.agent_position[0] + 1)

        # Check if the next position is a wall
        if tuple(next_position) not in self.walls:
            self.agent_position = next_position

        # Check if the agent reached the goal
        terminated = (self.agent_position == self.goal_position)
        reward = 1.0 if terminated else -0.1  # Reward for reaching the goal

        return np.array(self.agent_position, dtype=np.int32), reward, terminated, {}

    def render(self):
        '''
        This function is used to print the current state of the grid world.
        '''
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.goal_position[1]][self.goal_position[0]] = "G"
        grid[self.agent_position[1]][self.agent_position[0]] = "A"

        for wall in self.walls:
            grid[wall[1]][wall[0]] = "#"

        print("\n".join([" ".join(row) for row in grid]))
        print("-" * (self.grid_size * 2))


class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        '''
        This function is used to initialize a MLP as PPONetwork.
        Args:
            state_size: Size of the state
            hidden_size: Size of the hidden layers
            action_size: Size of the action space
        '''
        super(PPONetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)  # Output probabilities for actions
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output a single value for state value
        )

    def forward(self, x):
        '''
        This function is used to pass the input state x through the PPONetwork,
        and then output a vector of action probabilities, and state value.
        Args:
            x: Input state
        Returns:
            action_probs: Probabilities of actions
            state_value: Value of the state
        '''
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        
        return action_probs, state_value


class PPOAgent:
    def __init__(self, gamma, clip_ratio, ppo_epochs, batch_size, state_size, action_size, lr=3e-4):
        '''
        This function is used to initialize some parameters of the model.
        Args: 
            gamma: Discount factor
            clip_ratio: Clipping ratio for PPO
            ppo_epochs: Number of PPO epochs
            batch_size: Batch size
            state_size: Size of the state
            action_size: Size of the action space
            lr: Learning rate
        '''
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.policy = PPONetwork(state_size, action_size, hidden_size=64)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []

    def get_action(self, state):
        '''
        This function is used to pass an input state through the PPONetwork,
        and then receive an action, and the probability to take this action.
        Args:
            state: Input state
        Returns:
            action: Selected action
            log_prob: Log probability of the action
        '''
        state = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()

    def remember(self, state, action, log_prob, reward, done):
        '''
        This function is used to store experiences in memory.
        Args:
            state: Current state
            action: Action taken
            log_prob: Log probability of the action
            reward: Reward received
            done: Whether the episode is done
        '''
        self.memory.append((state, action, log_prob, reward, done))

    def update(self):
        '''
        This function is used to calculate the returns and update the policy.
        '''
        states, actions, log_probs, rewards, dones = zip(*self.memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Compute discounted returns
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Update policy
        for _ in range(self.ppo_epochs):
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Calculate advantages
            advantages = returns - state_values.detach().squeeze()

            # Calculate ratio
            ratio = (new_log_probs - log_probs).exp()

            # Surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = (returns - state_values.squeeze()).pow(2).mean()

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear memory
        self.memory = []


def train():
    # Define walls for the maze - một mê cung phức tạp 20x20
    walls = [
        # Bức tường ngang lớn ở giữa
        (2, 2), (2, 3), (2, 4), (2, 5), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17),
        
        # Bức tường dọc bên trái
        (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5), (13, 5), (14, 5), (15, 5), (16, 5), (17, 5),
        
        # Bức tường zigzag phía trên
        (5, 8), (5, 9), (5, 10), (5, 11), (5, 12),
        (6, 12), (7, 12), (8, 12),
        (8, 8), (8, 9), (8, 10), (8, 11),
        
        # Khu vực hình chữ U
        (10, 10), (11, 10), (12, 10), (13, 10), (14, 10),
        (14, 11), (14, 12), (14, 13), (14, 14),
        (10, 14), (11, 14), (12, 14), (13, 14),
        
        # Các bức tường nhỏ tạo ngõ cụt
        (4, 15), (4, 16), (4, 17),
        (7, 15), (7, 16), (7, 17),
        (10, 17), (11, 17), (12, 17),
        (15, 8), (15, 9), (15, 10),
        (17, 12), (17, 13), (17, 14),
        
        # Hình chữ S
        (12, 2), (13, 2), (14, 2), (15, 2), (16, 2),
        (12, 3), (12, 4),
        (16, 3), (16, 4),
        (12, 6), (13, 6), (14, 6), (15, 6), (16, 6),
        
        # Các bức tường ngẫu nhiên
        (18, 3), (18, 4), (18, 5),
        (6, 15), (6, 16),
        (9, 2), (9, 3),
        (13, 8), (13, 9),
        (17, 7), (17, 8),
        (18, 0), (18, 1), (18, 2),
        (2, 18), (2, 19)
    ]

    env = GridWorldEnv(grid_size=20, goal_position=[19, 19], walls=walls)
    agent = PPOAgent(gamma=0.99, clip_ratio=0.2, ppo_epochs=4, batch_size=64, state_size=2, action_size=4)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        print(f"Episode {episode + 1} starts:")
        while not done:
            env.render()  # Render the grid world to show the agent's movement
            action, log_prob = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, log_prob, reward, done)
            state = next_state
            episode_reward += reward

            # time.sleep(0.000000001)  # Add a delay between steps

        print(f"Episode {episode + 1} ends with reward: {episode_reward}\n")
        time.sleep(1)
        agent.update()

if __name__ == "__main__":
    train()

if __name__ == "__main__":
    train()