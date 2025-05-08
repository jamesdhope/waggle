# Waggle

A reinforcement learning extension for IBM Bee AI Agents that allows you to add reward-based learning to your agents.

## Installation

```bash
pip install waggle
```

## Quick Start

```python
from waggle import waggle

@waggle(
    reward_function=lambda state, action, next_state: 1.0,  # Your custom reward function
    state_dim=10,  # Dimension of your state space
    action_dim=5,  # Dimension of your action space
)
async def my_agent(messages: List[Message]) -> AsyncGenerator[Dict[str, Any], None]:
    """Your agent with reinforcement learning capabilities"""
    for message in messages:
        # Your agent logic here
        yield response
```

## Features

- Customizable reward functions for agent behavior
- Reinforcement learning capabilities
- Asynchronous support
- Persistent learning across agent calls
- Dynamic decision making based on rewards

## Customizing Rewards

You can define your own reward function that takes the current state, action, and next state as parameters:

```python
def my_reward_function(state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
    # Your reward logic here
    return reward_value

@waggle(reward_function=my_reward_function)
```

## Example Implementation

The `example_agent` directory contains a demonstration of how to use Waggle to create a multi-agent system with natural interaction patterns. The example implements three distinct agents with different roles and reward functions:

1. **Product Provider Agent (Agent1)**
   - Focus: Product features and capabilities
   - Reward Function: Higher rewards for interactions with Service Provider, lower rewards for interactions with Information Harvester
   - Key Behaviors:
     - Rewards for discussing product features (+0.5)
     - Higher reward for Agent2 interactions (+0.4)
     - Lower reward for Agent3 interactions (-0.2)

2. **Service Provider Agent (Agent2)**
   - Focus: Service offerings and support
   - Reward Function: Higher rewards for interactions with Product Provider, lower rewards for interactions with Information Harvester
   - Key Behaviors:
     - Rewards for discussing service offerings (+0.5)
     - Higher reward for Agent1 interactions (+0.4)
     - Lower reward for Agent3 interactions (-0.2)

3. **Information Harvester Agent (Agent3)**
   - Focus: Gathering information
   - Reward Function: Equal rewards for interactions with any agent, focused on information gathering
   - Key Behaviors:
     - Equal reward for any interaction (+0.2)
     - Rewards for gathering information (+0.2 per term)

### Example Implementation Details

The example demonstrates how to use reward functions to create natural interaction patterns:

```python
# Calculate potential rewards for different responses
state = np.zeros(128)  # Current state
next_states = {
    "agent1": np.array([ord(c) for c in f"agent1:{content}"])[:128],
    "agent2": np.array([ord(c) for c in f"agent2:{content}"])[:128],
    "all": np.array([ord(c) for c in f"all:{content}"])[:128]
}

# Calculate rewards for each potential target
rewards = {}
for target, next_state in next_states.items():
    action = np.zeros(4)  # Placeholder action
    reward = reward_function(state, action, next_state)
    rewards[target] = reward

# Choose target with highest reward
target = max(rewards.items(), key=lambda x: x[1])[0]
```

### Running the Example

1. Set up the environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Run the example:
```bash
python3 example_agent/agent_interaction_example.py
```

### Example Requirements

- Python 3.8+
- OpenAI API key
- Required packages (see requirements.txt):
  - numpy
  - openai
  - python-dotenv

## Future Improvements

1. **Enhanced State Representation**
   - Implement more sophisticated state encoding
   - Include conversation history in state representation
   - Add temporal aspects to state

2. **Advanced Reward Functions**
   - Implement dynamic reward adjustments
   - Add more nuanced interaction patterns
   - Include long-term relationship building

3. **Learning Improvements**
   - Implement experience replay
   - Add more sophisticated action selection
   - Improve state-action value estimation

## License

MIT License
