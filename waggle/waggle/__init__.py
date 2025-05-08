import functools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Callable, Any, Optional, Dict, List, AsyncGenerator

class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class WaggleAgent:
    def __init__(
        self,
        reward_function: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        self.reward_function = reward_function
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = []
        self.batch_size = 64
        
    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def update_model(self, state: np.ndarray, action: int, next_state: np.ndarray):
        reward = self.reward_function(state, action, next_state)
        self.memory.append((state, action, reward, next_state))
        
        if len(self.memory) < self.batch_size:
            return
        
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states = zip(*[self.memory[i] for i in batch])
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if len(self.memory) > 1000:
            self.memory = self.memory[-1000:]

def waggle(
    reward_function: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    state_dim: int,
    action_dim: int,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    epsilon: float = 0.1
):
    """
    Decorator to add reinforcement learning capabilities to an agent.
    
    Args:
        reward_function: Function that computes the reward given state, action, and next_state
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        learning_rate: Learning rate for the DQN
        gamma: Discount factor for future rewards
        epsilon: Exploration rate
    """
    def decorator(func):
        agent = WaggleAgent(
            reward_function=reward_function,
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon
        )
        
        @functools.wraps(func)
        async def wrapper(messages: List[Any]) -> AsyncGenerator[Dict[str, Any], None]:
            state = np.zeros(state_dim)  # Initial state
            
            async for yield_value in func(messages):
                if isinstance(yield_value, dict) and "thought" in yield_value:
                    # This is a thought, use it to update state
                    state = np.array([ord(c) for c in yield_value["thought"]])[:state_dim]
                    if len(state) < state_dim:
                        state = np.pad(state, (0, state_dim - len(state)))
                    yield yield_value
                else:
                    # This is an action, get the next action from the agent
                    action = agent.select_action(state)
                    next_state = np.array([ord(c) for c in str(yield_value)])[:state_dim]
                    if len(next_state) < state_dim:
                        next_state = np.pad(next_state, (0, state_dim - len(next_state)))
                    
                    agent.update_model(state, action, next_state)
                    state = next_state
                    yield yield_value
        
        return wrapper
    return decorator 