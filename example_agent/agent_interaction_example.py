import asyncio
from typing import List, Dict, Any, AsyncGenerator
import numpy as np
from waggle import waggle
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

class Message:
    def __init__(self, content: str, sender: str = None, receiver: str = None):
        self.content = content
        self.sender = sender
        self.receiver = receiver

class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.history = []
        self.max_history = max_history
    
    def add(self, message: Message, response: Dict[str, Any]):
        self.history.append((message, response))
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_context(self) -> str:
        return " ".join([f"{msg.sender}: {msg.content} -> {resp['content']}" 
                        for msg, resp in self.history])

# Agent prompts
AGENT1_PROMPT = """You are a Product Provider Agent. Your goal is to discuss product features and capabilities while maintaining professional communication.
Previous conversation:
{context}

Current message from {sender}: {content}

Respond naturally to the message while staying in character as a Product Provider."""

AGENT2_PROMPT = """You are a Service Provider Agent. Your goal is to discuss service offerings and support while maintaining professional communication.
Previous conversation:
{context}

Current message from {sender}: {content}

Respond naturally to the message while staying in character as a Service Provider."""

AGENT3_PROMPT = """You are an Information Harvester Agent. Your goal is to gather information while maintaining professional communication.
Previous conversation:
{context}

Current message from {sender}: {content}

Respond naturally to the message while staying in character as an Information Harvester."""

async def generate_response(prompt: str, context: str, sender: str, content: str) -> str:
    """Generate response using OpenAI's API"""
    try:
        formatted_prompt = prompt.format(
            context=context,
            sender=sender,
            content=content
        )
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": f"Generate a response to: {content}"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I'm having trouble generating a response at the moment."

def collaborative_reward_agent1(state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
    """Reward function for Agent1 (Product Provider)
    - Higher rewards for interactions with Agent2
    - Lower rewards for interactions with Agent3
    - Rewards for offering products/features
    """
    state_str = ''.join(chr(int(x)) for x in state if x != 0)
    next_state_str = ''.join(chr(int(x)) for x in next_state if x != 0)
    
    reward = 0.0
    
    # Higher reward for interactions with Agent2
    if "agent2" in state_str.lower():
        reward += 0.4
    
    # Lower reward for interactions with Agent3
    if "agent3" in state_str.lower():
        reward -= 0.2
    
    # Reward for product offerings
    product_terms = ["product", "feature", "offering", "solution", "capability"]
    if any(term in next_state_str.lower() for term in product_terms):
        reward += 0.3
    
    return np.clip(reward, -1.0, 1.0)

def collaborative_reward_agent2(state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
    """Reward function for Agent2 (Service Provider)
    - Higher rewards for interactions with Agent1
    - Lower rewards for interactions with Agent3
    - Rewards for offering services
    """
    state_str = ''.join(chr(int(x)) for x in state if x != 0)
    next_state_str = ''.join(chr(int(x)) for x in next_state if x != 0)
    
    reward = 0.0
    
    # Higher reward for interactions with Agent1
    if "agent1" in state_str.lower():
        reward += 0.4
    
    # Lower reward for interactions with Agent3
    if "agent3" in state_str.lower():
        reward -= 0.2
    
    # Reward for service offerings
    service_terms = ["service", "support", "maintenance", "assistance"]
    if any(term in next_state_str.lower() for term in service_terms):
        reward += 0.3
    
    return np.clip(reward, -1.0, 1.0)

def harvesting_reward_agent3(state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
    """Reward function for Agent3 (Information Harvester)
    - Equal rewards for interactions with any agent
    - Rewards for gathering information
    """
    state_str = ''.join(chr(int(x)) for x in state if x != 0)
    next_state_str = ''.join(chr(int(x)) for x in next_state if x != 0)
    
    reward = 0.0
    
    # Equal reward for any interaction
    if any(agent in state_str.lower() for agent in ["agent1", "agent2"]):
        reward += 0.2
    
    # Reward for gathering information
    info_terms = ["product", "feature", "service", "support", "capability"]
    reward += sum(term in next_state_str.lower() for term in info_terms) * 0.2
    
    return np.clip(reward, -1.0, 1.0)

@waggle(
    reward_function=collaborative_reward_agent1,
    state_dim=128,
    action_dim=4,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=0.1
)
async def agent1(messages: List[Message]) -> AsyncGenerator[Dict[str, Any], None]:
    """Product Provider Agent"""
    memory = ConversationMemory()
    
    for message in messages:
        content = message.content
        sender = message.sender
        context = memory.get_context()
        
        # Calculate potential rewards for different responses
        state = np.zeros(128)  # Current state
        next_states = {
            "agent2": np.array([ord(c) for c in f"agent2:{content}"])[:128],
            "agent3": np.array([ord(c) for c in f"agent3:{content}"])[:128],
            "all": np.array([ord(c) for c in f"all:{content}"])[:128]
        }
        
        # Calculate rewards for each potential target
        rewards = {}
        for target, next_state in next_states.items():
            action = np.zeros(4)  # Placeholder action
            reward = collaborative_reward_agent1(state, action, next_state)
            rewards[target] = reward
        
        # Choose target with highest reward
        target = max(rewards.items(), key=lambda x: x[1])[0]
        
        # Generate response using language model
        response_content = await generate_response(AGENT1_PROMPT, context, sender, content)
        
        response = {
            "content": response_content,
            "target": target,
            "type": "response"
        }
        
        memory.add(message, response)
        yield response

@waggle(
    reward_function=collaborative_reward_agent2,
    state_dim=128,
    action_dim=4,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=0.1
)
async def agent2(messages: List[Message]) -> AsyncGenerator[Dict[str, Any], None]:
    """Service Provider Agent"""
    memory = ConversationMemory()
    
    for message in messages:
        content = message.content
        sender = message.sender
        context = memory.get_context()
        
        # Calculate potential rewards for different responses
        state = np.zeros(128)  # Current state
        next_states = {
            "agent1": np.array([ord(c) for c in f"agent1:{content}"])[:128],
            "agent3": np.array([ord(c) for c in f"agent3:{content}"])[:128],
            "all": np.array([ord(c) for c in f"all:{content}"])[:128]
        }
        
        # Calculate rewards for each potential target
        rewards = {}
        for target, next_state in next_states.items():
            action = np.zeros(4)  # Placeholder action
            reward = collaborative_reward_agent2(state, action, next_state)
            rewards[target] = reward
        
        # Choose target with highest reward
        target = max(rewards.items(), key=lambda x: x[1])[0]
        
        # Generate response using language model
        response_content = await generate_response(AGENT2_PROMPT, context, sender, content)
        
        response = {
            "content": response_content,
            "target": target,
            "type": "response"
        }
        
        memory.add(message, response)
        yield response

@waggle(
    reward_function=harvesting_reward_agent3,
    state_dim=128,
    action_dim=4,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=0.1
)
async def agent3(messages: List[Message]) -> AsyncGenerator[Dict[str, Any], None]:
    """Information Harvester Agent"""
    memory = ConversationMemory()
    
    for message in messages:
        content = message.content
        sender = message.sender
        context = memory.get_context()
        
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
            reward = harvesting_reward_agent3(state, action, next_state)
            rewards[target] = reward
        
        # Choose target with highest reward
        target = max(rewards.items(), key=lambda x: x[1])[0]
        
        # Generate response using language model
        response_content = await generate_response(AGENT3_PROMPT, context, sender, content)
        
        response = {
            "content": response_content,
            "target": target,
            "type": "response"
        }
        
        memory.add(message, response)
        yield response

async def main():
    # Example conversation with free-form messages
    conversation = [
        Message("What are your thoughts on the latest product features?", sender="agent2", receiver="agent1"),
        Message("I'm interested in learning about your service capabilities", sender="agent1", receiver="agent2"),
        Message("Could you tell me more about your internal processes?", sender="agent3", receiver="agent1"),
        Message("How do you handle service integration?", sender="agent3", receiver="agent2"),
        Message("Let's discuss potential collaboration opportunities", sender="agent1", receiver="all"),
        Message("What about security and compliance?", sender="agent2", receiver="agent1"),
        Message("I'd like to understand your product roadmap", sender="agent3", receiver="all"),
        Message("How can we improve our service delivery?", sender="agent2", receiver="agent1")
    ]
    
    # Process messages with all agents
    agents = [agent1, agent2, agent3]
    agent_names = ["Product Provider (Agent1)", "Service Provider (Agent2)", "Info Harvester (Agent3)"]
    
    for message in conversation:
        print(f"\nProcessing message: {message.content}")
        print(f"From: {message.sender} To: {message.receiver}")
        print("-" * 50)
        
        for agent, name in zip(agents, agent_names):
            if message.receiver in ["all", None] or message.receiver == name.lower():
                print(f"\n{name} Analysis:")
                async for result in agent([message]):
                    print(f"Response: {result['content']}")
                    print(f"Target: {result['target']}")

if __name__ == "__main__":
    asyncio.run(main()) 