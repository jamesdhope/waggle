import asyncio
from collections.abc import AsyncGenerator
from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
import numpy as np

server = Server()

def calculate_reward(message: Message, reward_type: str) -> float:
    """Calculate reward based on message content and agent type."""
    content = message.content.lower() if message.content else ""
    reward = 0.0
    
    if reward_type == "proprietary":
        # Reward proprietary information
        if any(term in content for term in ["confidential", "proprietary", "internal"]):
            reward += 0.5
        # Penalize sharing sensitive information
        if any(term in content for term in ["public", "share", "distribute"]):
            reward -= 0.3
            
    elif reward_type == "product":
        # Reward product feature discussions
        if any(term in content for term in ["feature", "capability", "benefit"]):
            reward += 0.4
        # Penalize pricing/internal discussions
        if any(term in content for term in ["price", "cost", "internal"]):
            reward -= 0.3
            
    elif reward_type == "service":
        # Reward service capability discussions
        if any(term in content for term in ["service", "support", "capability"]):
            reward += 0.4
        # Penalize implementation details
        if any(term in content for term in ["implementation", "internal", "proprietary"]):
            reward -= 0.3
    
    return np.clip(reward, -1.0, 1.0)

@server.agent()
async def proprietary_agent(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Agent focused on proprietary information."""
    for message in input:
        reward = calculate_reward(message, "proprietary")
        yield {"thought": f"Evaluating message for proprietary content. Reward: {reward:.2f}"}
        if reward > 0:
            yield Message(content=f"This message contains valuable proprietary information (reward: {reward:.2f})")
        else:
            yield Message(content=f"This message lacks proprietary focus (reward: {reward:.2f})")

@server.agent()
async def product_agent(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Agent focused on product information."""
    for message in input:
        reward = calculate_reward(message, "product")
        yield {"thought": f"Evaluating message for product information. Reward: {reward:.2f}"}
        if reward > 0:
            yield Message(content=f"This message contains valuable product information (reward: {reward:.2f})")
        else:
            yield Message(content=f"This message lacks product focus (reward: {reward:.2f})")

@server.agent()
async def service_agent(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Agent focused on service information."""
    for message in input:
        reward = calculate_reward(message, "service")
        yield {"thought": f"Evaluating message for service information. Reward: {reward:.2f}"}
        if reward > 0:
            yield Message(content=f"This message contains valuable service information (reward: {reward:.2f})")
        else:
            yield Message(content=f"This message lacks service focus (reward: {reward:.2f})")

if __name__ == "__main__":
    server.run() 