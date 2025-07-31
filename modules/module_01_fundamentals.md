# Module 1: Fundamentals & Core Concepts
## Detailed Code Examples

### Learning Objectives
- Understand Pydantic AI's architecture and design philosophy
- Create and run basic agents
- Master the Agent class and its core parameters
- Understand the difference between `run()` and `run_sync()`

---

## 1.1 Basic Agent Creation - Customer Service Bot

```python
# basic_customer_service.py
import asyncio
from pydantic_ai import Agent
from pydantic import BaseModel
from typing import Literal

# Basic agent for customer inquiries
customer_agent = Agent(
    'openai:gpt-4o',
    system_prompt="""
    You are a helpful customer service representative for TechCorp, 
    a technology company that sells laptops, smartphones, and accessories.
    Be polite, professional, and helpful. Always try to solve the customer's problem.
    """,
    retries=2  # Retry on failures
)

# Run a simple interaction
async def basic_example():
    result = await customer_agent.run(
        "Hi, I'm having trouble with my laptop battery. It doesn't seem to charge properly."
    )
    print("Agent Response:", result.output)
    print("Usage Stats:", result.usage())

if __name__ == "__main__":
    asyncio.run(basic_example())
```

**Key Concepts:**
- **Agent Creation**: Simple instantiation with model and system prompt
- **System Prompt**: Defines the agent's role and behavior
- **Retries**: Automatic retry mechanism for failures
- **Usage Tracking**: Built-in token and cost tracking

---

## 1.2 Model Configuration & Environment Setup

```python
# model_configuration.py
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.openai import OpenAIProvider

# Multiple model configurations
class ModelManager:
    def __init__(self):
        # Primary model with custom settings
        self.primary_agent = Agent(
            OpenAIModel(
                'gpt-4o',
                provider=OpenAIProvider(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    organization=os.getenv('OPENAI_ORG_ID')  # Optional
                )
            ),
            system_prompt="You are a senior customer service specialist."
        )
        
        # Fallback model
        self.fallback_agent = Agent(
            AnthropicModel('claude-3-sonnet-20241022'),
            system_prompt="You are a backup customer service agent."
        )
        
        # Fast model for simple queries
        self.quick_agent = Agent(
            'openai:gpt-4o-mini',
            system_prompt="Provide quick, concise responses to simple questions."
        )

async def demonstrate_models():
    manager = ModelManager()
    
    query = "What's your return policy?"
    
    # Try primary, fallback to others if needed
    try:
        result = await manager.primary_agent.run(query)
        print("Primary agent response:", result.output)
    except Exception as e:
        print(f"Primary failed: {e}")
        result = await manager.fallback_agent.run(query)
        print("Fallback agent response:", result.output)

if __name__ == "__main__":
    asyncio.run(demonstrate_models())
```

**Key Concepts:**
- **Multiple Model Support**: OpenAI, Anthropic, Google, etc.
- **Provider Configuration**: Custom API keys and settings
- **Model Selection Strategy**: Primary/fallback patterns
- **Environment Variables**: Secure API key management

---

## 1.3 Structured Output Types

```python
# structured_outputs.py
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class CustomerIssue(BaseModel):
    """Structured representation of a customer issue"""
    category: Literal["technical", "billing", "shipping", "general"] = Field(
        description="The main category of the customer issue"
    )
    priority: TicketPriority = Field(
        description="Priority level based on issue severity and customer status"
    )
    summary: str = Field(
        description="Brief summary of the issue in 1-2 sentences"
    )
    suggested_actions: List[str] = Field(
        description="List of recommended actions to resolve the issue"
    )
    estimated_resolution_hours: Optional[int] = Field(
        description="Estimated hours to resolve, if determinable"
    )
    requires_escalation: bool = Field(
        description="Whether this issue needs to be escalated to a specialist"
    )

# Agent that analyzes customer messages and returns structured data
issue_analyzer = Agent(
    'openai:gpt-4o',
    output_type=CustomerIssue,
    system_prompt="""
    You are an AI that analyzes customer service inquiries and categorizes them.
    
    Guidelines for categorization:
    - Technical: Hardware/software problems, setup issues, troubleshooting
    - Billing: Payment issues, subscription questions, refunds
    - Shipping: Delivery problems, tracking, returns
    - General: Product info, company policies, general questions
    
    Priority levels:
    - URGENT: System down, security issues, VIP customers
    - HIGH: Major functionality broken, billing disputes
    - MEDIUM: Minor bugs, general questions
    - LOW: Feature requests, general inquiries
    """,
    retries=3
)

async def analyze_customer_message():
    customer_message = """
    I purchased a MacBook Pro last week and it arrived yesterday, but when I try to turn it on,
    nothing happens. The screen stays black and I don't see any lights. I tried different power
    outlets and the charging cable seems fine. I have an important presentation tomorrow and
    really need this resolved quickly. My order number is #12345.
    """
    
    result = await issue_analyzer.run(customer_message)
    issue = result.output
    
    print("=== CUSTOMER ISSUE ANALYSIS ===")
    print(f"Category: {issue.category}")
    print(f"Priority: {issue.priority}")
    print(f"Summary: {issue.summary}")
    print(f"Suggested Actions:")
    for i, action in enumerate(issue.suggested_actions, 1):
        print(f"  {i}. {action}")
    print(f"Estimated Resolution: {issue.estimated_resolution_hours} hours")
    print(f"Requires Escalation: {'Yes' if issue.requires_escalation else 'No'}")

if __name__ == "__main__":
    asyncio.run(analyze_customer_message())
```

**Key Concepts:**
- **Structured Outputs**: Using Pydantic models for consistent data
- **Output Validation**: Automatic validation of LLM responses
- **Enums and Literals**: Constraining possible values
- **Field Descriptions**: Helping the LLM understand requirements

---

## 1.4 Error Handling and Retries

```python
# error_handling.py
import asyncio
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import UserError
import random

class ReliableCustomerAgent:
    def __init__(self):
        self.agent = Agent(
            'openai:gpt-4o',
            system_prompt="""
            You are a customer service agent. Always be helpful and professional.
            If you cannot answer a question, say so clearly and offer to escalate.
            """,
            retries=3  # Automatic retries for model failures
        )
    
    async def handle_customer_query(self, message: str, max_attempts: int = 3) -> str:
        """Handle customer query with comprehensive error handling"""
        
        for attempt in range(max_attempts):
            try:
                # Simulate intermittent network issues
                if random.random() < 0.2:  # 20% chance of simulated failure
                    raise Exception("Simulated network error")
                
                result = await self.agent.run(message)
                
                # Validate response quality
                if len(result.output.strip()) < 10:
                    raise ModelRetry("Response too short, please provide more detail")
                
                return result.output
                
            except ModelRetry as e:
                print(f"Attempt {attempt + 1}: Model retry needed - {e}")
                if attempt == max_attempts - 1:
                    return "I apologize, but I'm having trouble generating a proper response. Please try again or contact our human support team."
                
            except UserError as e:
                print(f"User error: {e}")
                return "I'm sorry, but there seems to be an issue with your request. Please rephrase and try again."
                
            except Exception as e:
                print(f"Attempt {attempt + 1}: Unexpected error - {e}")
                if attempt == max_attempts - 1:
                    return "I'm experiencing technical difficulties. Please contact our support team directly for assistance."
                
                # Wait before retry (exponential backoff)
                await asyncio.sleep(2 ** attempt)
        
        return "Maximum retry attempts reached. Please contact support."

async def demonstrate_error_handling():
    agent = ReliableCustomerAgent()
    
    queries = [
        "What's your return policy?",
        "My laptop won't start, help!",
        "",  # Empty query to test validation
        "Can you help me with a complex technical issue involving driver conflicts?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        response = await agent.handle_customer_query(query)
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(demonstrate_error_handling())
```

**Key Concepts:**
- **ModelRetry**: Forcing the model to retry with better instructions
- **UserError**: Handling user input validation errors
- **Exponential Backoff**: Smart retry timing
- **Quality Validation**: Checking response quality before returning

---

## 1.5 Agent Configuration Best Practices

```python
# configuration_best_practices.py
import asyncio
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic import BaseModel
from typing import Optional
import os

class AgentConfig(BaseModel):
    """Configuration model for agents"""
    model_name: str = "openai:gpt-4o-mini"
    max_retries: int = 2
    system_prompt: Optional[str] = None
    max_tokens_per_request: int = 1000
    max_requests_per_session: int = 100
    temperature: float = 0.7

class ConfigurableAgent:
    """Agent with comprehensive configuration management"""
    
    def __init__(self, config: AgentConfig, agent_name: str = "assistant"):
        self.config = config
        self.agent_name = agent_name
        
        # Set default system prompt if none provided
        system_prompt = config.system_prompt or f"""
        You are {agent_name}, a helpful AI assistant for TechCorp.
        Always be professional, accurate, and helpful.
        If you're unsure about something, say so clearly.
        """
        
        # Create agent with configuration
        self.agent = Agent(
            config.model_name,
            system_prompt=system_prompt,
            retries=config.max_retries
        )
        
        # Usage limits for safety
        self.usage_limits = UsageLimits(
            request_limit=config.max_requests_per_session,
            total_tokens_limit=config.max_tokens_per_request * config.max_requests_per_session
        )
        
        # Track usage
        self.session_requests = 0
        self.total_tokens = 0
    
    async def process_request(self, user_input: str) -> dict:
        """Process a user request with full configuration applied"""
        
        # Check session limits
        if self.session_requests >= self.config.max_requests_per_session:
            return {
                "status": "error",
                "message": "Session request limit reached. Please start a new session."
            }
        
        try:
            # Run with usage limits
            result = await self.agent.run(
                user_input,
                usage_limits=self.usage_limits
            )
            
            # Update tracking
            self.session_requests += 1
            usage = result.usage()
            self.total_tokens += usage.total_tokens
            
            return {
                "status": "success",
                "response": result.output,
                "usage": {
                    "tokens_used": usage.total_tokens,
                    "requests_in_session": self.session_requests,
                    "tokens_remaining": (self.config.max_tokens_per_request * 
                                       self.config.max_requests_per_session) - self.total_tokens
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Agent error: {str(e)}"
            }
    
    def get_session_stats(self) -> dict:
        """Get current session statistics"""
        return {
            "agent_name": self.agent_name,
            "model": self.config.model_name,
            "session_requests": self.session_requests,
            "total_tokens_used": self.total_tokens,
            "max_requests": self.config.max_requests_per_session,
            "requests_remaining": self.config.max_requests_per_session - self.session_requests
        }

# Configuration factory for different environments
class AgentFactory:
    """Factory for creating agents with different configurations"""
    
    @staticmethod
    def create_development_agent() -> ConfigurableAgent:
        """Development environment agent - more permissive"""
        config = AgentConfig(
            model_name="openai:gpt-4o-mini",
            max_retries=3,
            max_tokens_per_request=2000,
            max_requests_per_session=50
        )
        return ConfigurableAgent(config, "dev_assistant")
    
    @staticmethod
    def create_production_agent() -> ConfigurableAgent:
        """Production environment agent - more restrictive"""
        config = AgentConfig(
            model_name="openai:gpt-4o",
            max_retries=2,
            max_tokens_per_request=1000,
            max_requests_per_session=100
        )
        return ConfigurableAgent(config, "prod_assistant")
    
    @staticmethod
    def create_demo_agent() -> ConfigurableAgent:
        """Demo environment agent - very restrictive"""
        config = AgentConfig(
            model_name="openai:gpt-4o-mini",
            max_retries=1,
            max_tokens_per_request=500,
            max_requests_per_session=10
        )
        return ConfigurableAgent(config, "demo_assistant")

async def demonstrate_configuration():
    """Demonstrate different agent configurations"""
    
    # Create agents for different environments
    environments = {
        "development": AgentFactory.create_development_agent(),
        "production": AgentFactory.create_production_agent(),
        "demo": AgentFactory.create_demo_agent()
    }
    
    test_query = "What are your laptop specifications and pricing?"
    
    for env_name, agent in environments.items():
        print(f"\n=== {env_name.upper()} ENVIRONMENT ===")
        
        # Process request
        result = await agent.process_request(test_query)
        
        if result["status"] == "success":
            print(f"Response: {result['response'][:100]}...")
            print(f"Usage: {result['usage']}")
        else:
            print(f"Error: {result['message']}")
        
        # Show session stats
        stats = agent.get_session_stats()
        print(f"Session stats: {stats['session_requests']}/{stats['max_requests']} requests used")

if __name__ == "__main__":
    asyncio.run(demonstrate_configuration())
```

**Key Concepts:**
- **Configuration Management**: Structured approach to agent setup
- **Usage Limits**: Preventing runaway costs and abuse
- **Environment-Specific Configs**: Different settings for dev/prod/demo
- **Factory Pattern**: Clean agent creation with predefined configurations
- **Session Tracking**: Monitoring usage across requests

---

## Practice Exercises for Module 1

### Exercise 1: Basic Agent Variations
Create three different agents for different customer service scenarios:
1. A technical support agent with detailed system prompts
2. A billing specialist with specific constraints
3. A general inquiry agent with broad knowledge

### Exercise 2: Error Recovery
Implement an agent that can:
- Handle network timeouts gracefully
- Retry with different models on failure
- Provide fallback responses when all else fails

### Exercise 3: Structured Analysis
Build an agent that analyzes customer feedback and returns:
- Sentiment score (1-10)
- Key topics mentioned
- Recommended follow-up actions
- Urgency level

### Exercise 4: Configuration System
Create a configuration system that allows:
- Loading agent settings from environment variables
- Different configurations for different customer tiers (basic, premium, enterprise)
- Runtime configuration updates

### Next Steps
Once you've mastered these fundamentals, you'll be ready for **Module 2: Tools & Function Calling**, where we'll add interactive capabilities to your agents.

**Key Takeaways:**
- Start simple with basic agents and gradually add complexity
- Always implement proper error handling and retry logic
- Use structured outputs for consistent, parseable responses
- Configure agents appropriately for different environments
- Monitor usage and implement safety limits