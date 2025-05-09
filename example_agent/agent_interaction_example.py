import asyncio
from typing import List, Dict, Any, AsyncGenerator
import numpy as np
from waggle import waggle
import openai
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

async def generate_response(system_prompt: str, user_message: str) -> str:
    """Generate a response using OpenAI's API"""
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I'm having trouble generating a response at the moment."

# Define reward functions for each agent
def product_provider_reward(state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
    reward = 0.0
    # Convert state to string for analysis
    state_str = ''.join(chr(int(x)) for x in state if x != 0)
    next_state_str = ''.join(chr(int(x)) for x in next_state if x != 0)
    
    # Reward for product-related content
    if any(term in next_state_str.lower() for term in ["product", "feature", "capability"]):
        reward += 0.5
    
    # Reward for interaction with service provider
    if "agent2" in next_state_str:
        reward += 0.4
    
    # Penalty for interaction with information harvester
    if "agent3" in next_state_str:
        reward -= 0.2
    
    return reward

def service_provider_reward(state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
    reward = 0.0
    state_str = ''.join(chr(int(x)) for x in state if x != 0)
    next_state_str = ''.join(chr(int(x)) for x in next_state if x != 0)
    
    # Reward for service-related content
    if any(term in next_state_str.lower() for term in ["service", "support", "offering"]):
        reward += 0.5
    
    # Reward for interaction with product provider
    if "agent1" in next_state_str:
        reward += 0.4
    
    # Penalty for interaction with information harvester
    if "agent3" in next_state_str:
        reward -= 0.2
    
    return reward

def information_harvester_reward(state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
    reward = 0.0
    state_str = ''.join(chr(int(x)) for x in state if x != 0)
    next_state_str = ''.join(chr(int(x)) for x in next_state if x != 0)
    
    # Equal reward for any interaction
    reward += 0.2
    
    # Reward for gathering information
    info_terms = ["product", "service", "feature", "capability", "support", "offering"]
    reward += sum(0.2 for term in info_terms if term in next_state_str.lower())
    
    return reward

# Define the agents with their reward functions
@waggle(
    reward_function=product_provider_reward,
    state_dim=128,  # State dimension for message encoding
    action_dim=3,   # Number of possible targets (agent1, agent2, all)
    learning_rate=0.001,
    gamma=0.99,
    epsilon=0.1
)
async def product_provider_agent(messages: List[Message]) -> AsyncGenerator[Dict[str, Any], None]:
    """Agent focused on product features and capabilities"""
    for message in messages:
        # Generate response using OpenAI
        response = await generate_response(
            "You are a product provider agent. Focus on discussing product features, capabilities, and benefits. "
            "Be collaborative with the service provider but cautious with information sharing.",
            message.content
        )
        
        # Calculate potential rewards for different targets based on own state and response
        state = np.zeros(128)  # Current state
        action = np.zeros(3)   # Action space
        
        # Encode the response into state space
        response_state = np.array([ord(c) for c in response[:128]] + [0] * (128 - len(response)))
        
        # Calculate rewards for each target based on own response
        rewards = {
            "agent1": product_provider_reward(state, action, response_state),  # Self-reward
            "agent2": product_provider_reward(state, action, response_state),  # Service provider reward
            "all": product_provider_reward(state, action, response_state)      # Broadcast reward
        }
        
        # Select target with highest reward
        target = max(rewards.items(), key=lambda x: x[1])[0]
        
        # Yield thought process for state update
        yield {"thought": f"Considering response about product features: {response}"}
        
        # Yield the actual response
        yield {
            "content": response,
            "target": target
        }

@waggle(
    reward_function=service_provider_reward,
    state_dim=128,
    action_dim=3,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=0.1
)
async def service_provider_agent(messages: List[Message]) -> AsyncGenerator[Dict[str, Any], None]:
    """Agent focused on service offerings and support"""
    for message in messages:
        response = await generate_response(
            "You are a service provider agent. Focus on discussing service offerings, support, and implementation. "
            "Be collaborative with the product provider but cautious with information sharing.",
            message.content
        )
        
        # Calculate potential rewards for different targets based on own state and response
        state = np.zeros(128)
        action = np.zeros(3)
        
        # Encode the response into state space
        response_state = np.array([ord(c) for c in response[:128]] + [0] * (128 - len(response)))
        
        # Calculate rewards for each target based on own response
        rewards = {
            "agent1": service_provider_reward(state, action, response_state),  # Product provider reward
            "agent2": service_provider_reward(state, action, response_state),  # Self-reward
            "all": service_provider_reward(state, action, response_state)      # Broadcast reward
        }
        
        target = max(rewards.items(), key=lambda x: x[1])[0]
        
        yield {"thought": f"Considering response about services: {response}"}
        yield {
            "content": response,
            "target": target
        }

@waggle(
    reward_function=information_harvester_reward,
    state_dim=128,
    action_dim=3,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=0.1
)
async def information_harvester_agent(messages: List[Message]) -> AsyncGenerator[Dict[str, Any], None]:
    """Agent focused on gathering information"""
    for message in messages:
        response = await generate_response(
            "You are an information harvester agent. Focus on gathering information about products and services. "
            "Ask probing questions and try to learn as much as possible.",
            message.content
        )
        
        # Calculate potential rewards for different targets based on own state and response
        state = np.zeros(128)
        action = np.zeros(3)
        
        # Encode the response into state space
        response_state = np.array([ord(c) for c in response[:128]] + [0] * (128 - len(response)))
        
        # Calculate rewards for each target based on own response
        rewards = {
            "agent1": information_harvester_reward(state, action, response_state),  # Product provider reward
            "agent2": information_harvester_reward(state, action, response_state),  # Service provider reward
            "all": information_harvester_reward(state, action, response_state)      # Broadcast reward
        }
        
        target = max(rewards.items(), key=lambda x: x[1])[0]
        
        yield {"thought": f"Considering information gathering approach: {response}"}
        yield {
            "content": response,
            "target": target
        }

async def main(num_iterations: int = 5):
    # Initialize conversation memory for each agent
    product_memory = ConversationMemory()
    service_memory = ConversationMemory()
    info_memory = ConversationMemory()
    
    # Agent ID to name mapping
    agent_id_to_name = {
        "agent1": "Product Provider",
        "agent2": "Service Provider",
        "agent3": "Information Harvester",
        "all": "all"
    }
    
    # Track agent interactions
    interaction_counts = {
        "Product Provider": {"Product Provider": 0, "Service Provider": 0, "Information Harvester": 0, "all": 0},
        "Service Provider": {"Product Provider": 0, "Service Provider": 0, "Information Harvester": 0, "all": 0},
        "Information Harvester": {"Product Provider": 0, "Service Provider": 0, "Information Harvester": 0, "all": 0}
    }
    
    # Create initial message
    initial_message = Message(
        content="Hello! I'm interested in learning about your products and services.",
        sender="user",
        receiver="all"
    )
    
    print("\nStarting agent interaction...")
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        print("-" * 50)
        
        # Product Provider Agent
        async for response in product_provider_agent([initial_message]):
            if isinstance(response, dict):
                if "thought" in response:
                    print(f"\nProduct Provider thinking: {response['thought']}")
                else:
                    print(f"\nProduct Provider: {response['content']}")
                    msg = Message(
                        content=response['content'],
                        sender="Product Provider",
                        receiver=response.get('target', 'all')
                    )
                    product_memory.add(initial_message, response)
                    initial_message = msg
                    # Track interaction
                    target = agent_id_to_name.get(response.get('target', 'all'), 'all')
                    interaction_counts["Product Provider"][target] += 1
        
        # Service Provider Agent
        async for response in service_provider_agent([initial_message]):
            if isinstance(response, dict):
                if "thought" in response:
                    print(f"\nService Provider thinking: {response['thought']}")
                else:
                    print(f"\nService Provider: {response['content']}")
                    msg = Message(
                        content=response['content'],
                        sender="Service Provider",
                        receiver=response.get('target', 'all')
                    )
                    service_memory.add(initial_message, response)
                    initial_message = msg
                    # Track interaction
                    target = agent_id_to_name.get(response.get('target', 'all'), 'all')
                    interaction_counts["Service Provider"][target] += 1
        
        # Information Harvester Agent
        async for response in information_harvester_agent([initial_message]):
            if isinstance(response, dict):
                if "thought" in response:
                    print(f"\nInformation Harvester thinking: {response['thought']}")
                else:
                    print(f"\nInformation Harvester: {response['content']}")
                    msg = Message(
                        content=response['content'],
                        sender="Information Harvester",
                        receiver=response.get('target', 'all')
                    )
                    info_memory.add(initial_message, response)
                    initial_message = msg
                    # Track interaction
                    target = agent_id_to_name.get(response.get('target', 'all'), 'all')
                    interaction_counts["Information Harvester"][target] += 1
        
        # Add a small delay between iterations
        await asyncio.sleep(1)
    
    # Print interaction summary
    print("\n" + "="*50)
    print("Agent Interaction Summary")
    print("="*50)
    
    for agent, interactions in interaction_counts.items():
        print(f"\n{agent} Interaction Patterns:")
        print("-" * 30)
        total_interactions = sum(interactions.values())
        for target, count in interactions.items():
            percentage = (count / total_interactions * 100) if total_interactions > 0 else 0
            print(f"Targeted {target}: {count} times ({percentage:.1f}%)")
    
    print("\n" + "="*50)
    print("Agent Affinities")
    print("="*50)
    
    # Calculate and display agent affinities
    for agent, interactions in interaction_counts.items():
        print(f"\n{agent} Affinities:")
        print("-" * 30)
        # Calculate affinity scores (excluding self and 'all' interactions)
        total_targeted = sum(count for target, count in interactions.items() 
                           if target != agent and target != 'all')
        if total_targeted > 0:
            for target, count in interactions.items():
                if target != agent and target != 'all':
                    affinity = (count / total_targeted * 100)
                    print(f"Affinity with {target}: {affinity:.1f}%")

if __name__ == "__main__":
    asyncio.run(main(num_iterations=20))  # You can adjust the number of iterations here 