import asyncio
from collections.abc import AsyncGenerator
import numpy as np
from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from waggle import waggle

# Define a simple sentiment scoring function
def calculate_sentiment(text: str) -> float:
    positive_words = {'happy', 'good', 'great', 'excellent', 'wonderful', 'amazing'}
    negative_words = {'sad', 'bad', 'terrible', 'awful', 'horrible', 'poor'}
    
    words = text.lower().split()
    score = 0
    for word in words:
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1
    return score

# Define our reward function
def sentiment_reward(state: np.ndarray, action: int, next_state: np.ndarray) -> float:
    # Convert the state back to text (this is a simplification of our state representation)
    response_text = ''.join(chr(int(x)) for x in next_state if x != 0)
    sentiment = calculate_sentiment(response_text)
    
    # Normalize the reward between -1 and 1
    return np.tanh(sentiment)

server = Server()

@server.agent()
@waggle(
    reward_function=sentiment_reward,
    state_dim=50,  # We'll use a 50-dimensional state space
    action_dim=10,  # We'll have 10 different response templates
    learning_rate=0.001,
    gamma=0.99,
    epsilon=0.1
)
async def sentiment_agent(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """An agent that learns to respond with positive sentiment"""
    
    # Define some response templates
    response_templates = [
        "I'm happy to help!",
        "That's wonderful to hear!",
        "I'm glad you asked that.",
        "That's a great question!",
        "I'm excited to assist you.",
        "That's interesting!",
        "I understand your concern.",
        "Let me help you with that.",
        "I'm here to support you.",
        "That's a good point!"
    ]
    
    for message in input:
        # First, yield a thought about the message
        thought = f"I'm thinking about how to respond to: {message.content[:20]}..."
        yield {"thought": thought}
        
        # The waggle decorator will handle the reinforcement learning
        # and select the best response template based on learned behavior
        response = response_templates[np.random.randint(0, len(response_templates))]
        yield response

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create some test messages
        messages = [
            Message(content="I'm feeling sad today"),
            Message(content="This is a great day!"),
            Message(content="I need help with something"),
        ]
        
        # Process the messages
        async for response in sentiment_agent(messages, Context()):
            if isinstance(response, dict) and "thought" in response:
                print(f"Agent thought: {response['thought']}")
            else:
                print(f"Agent response: {response}")
    
    # Run the example
    asyncio.run(main()) 