# Waggle

A reinforcement learning extension for IBM Bee AI Agents that allows you to add reward-based learning to your agents.

## Installation

```bash
pip install waggle
```

## Quick Start

```python
from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from waggle import waggle

server = Server()

@server.agent()
@waggle(
    reward_function=lambda state, action, next_state: 1.0,  # Your custom reward function
    state_dim=10,  # Dimension of your state space
    action_dim=5,  # Dimension of your action space
)
async def my_agent(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Your agent with reinforcement learning capabilities"""
    for message in input:
        # Your agent logic here
        yield message
```

## Features

- Seamless integration with IBM Bee AI Framework
- Customizable reward functions
- Reinforcement learning capabilities
- Asynchronous support
- Persistent learning across agent calls

## Customizing Rewards

You can define your own reward function that takes the current state, action, and next state as parameters:

```python
def my_reward_function(state, action, next_state):
    # Your reward logic here
    return reward_value

@waggle(reward_function=my_reward_function)
```

## License

MIT License # waggle
