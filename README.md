# Waggle Example Agent Implementation

This repository contains an example implementation of a multi-agent system using the Waggle library, demonstrating how reward functions can be used to create natural interaction patterns between agents.

## Overview

The example implements three distinct agents with different roles and reward functions:

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

## Implementation Details

### Reward-Based Decision Making

The agents use their reward functions to make decisions about who to interact with:

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

### Key Findings

1. **Natural Interaction Patterns**
   - Agent1 and Agent2 naturally form a collaborative relationship due to their mutually beneficial reward functions
   - Agent3 maintains a more neutral stance but focuses on information gathering
   - Interaction patterns emerge from the reward functions rather than being hardcoded

2. **Dynamic Decision Making**
   - Agents actively evaluate potential interactions based on their reward functions
   - Decisions about who to interact with are made in real-time
   - The system demonstrates emergent behavior through reward-based learning

3. **Effective Information Control**
   - Product and Service providers naturally limit information sharing with the Information Harvester
   - The Information Harvester adapts its approach based on the responses it receives
   - The system maintains professional boundaries while allowing for natural conversation flow

## Usage

To run the example:

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

## Requirements

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
