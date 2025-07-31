# Module 4: Advanced Agent Features
## Detailed Code Examples

### Learning Objectives
- Master system prompts and instructions
- Implement output validation
- Handle streaming responses
- Understand message history and conversation management

---

## 4.1 Dynamic System Prompts and Context Awareness

```python
# dynamic_prompts.py
import asyncio
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel

@dataclass
class UserContext:
    user_id: str
    name: str
    tier: str
    timezone: str
    preferences: Dict[str, Any]
    interaction_history: List[str]
    current_session_start: datetime

@dataclass
class SystemContext:
    current_load: int  # Number of active sessions
    maintenance_mode: bool
    business_hours: bool
    available_agents: int

@dataclass
class ContextualDeps:
    user_context: UserContext
    system_context: SystemContext
    knowledge_base: Dict[str, str]

contextual_agent = Agent(
    'openai:gpt-4o',
    deps_type=ContextualDeps,
    retries=2
)

@contextual_agent.system_prompt
async def dynamic_system_prompt(ctx: RunContext[ContextualDeps]) -> str:
    """Generate dynamic system prompt based on context"""
    user = ctx.deps.user_context
    system = ctx.deps.system_context
    
    # Base prompt
    prompt = f"""
    You are an AI customer service representative for TechCorp.
    
    CURRENT USER: {user.name} ({user.tier} tier customer)
    USER ID: {user.user_id}
    TIMEZONE: {user.timezone}
    SESSION START: {user.current_session_start.strftime('%Y-%m-%d %H:%M')}
    """
    
    # Tier-specific behavior
    if user.tier == "enterprise":
        prompt += """
        
    PRIORITY SERVICE: This is an enterprise customer. Provide premium support with:
    - Immediate escalation options
    - Technical detail when requested
    - Direct contact information for account managers
    """
    elif user.tier == "premium":
        prompt += """
        
    PREMIUM SERVICE: This is a premium customer. Provide enhanced support with:
    - Priority handling
    - Proactive solutions
    - Extended service options
    """
    else:
        prompt += """
        
    STANDARD SERVICE: Provide helpful, efficient support with:
    - Clear solutions
    - Self-service options when appropriate
    - Upgrade suggestions when relevant
    """
    
    # Business hours adjustment
    if not system.business_hours:
        prompt += """
        
    AFTER HOURS: You're operating in after-hours mode:
    - For urgent issues, provide emergency contact information
    - For non-urgent issues, set expectations for response time
    - Offer self-service options when possible
    """
    
    # System load adjustment
    if system.current_load > 80:
        prompt += """
        
    HIGH LOAD: System is experiencing high volume:
    - Be efficient and concise
    - Prioritize quick resolution
    - Direct to self-service when appropriate
    """
    
    # Maintenance mode
    if system.maintenance_mode:
        prompt += """
        
    MAINTENANCE MODE: Some services may be limited:
    - Inform users of potential delays
    - Focus on critical issues only
    - Provide alternative solutions
    """
    
    # User preferences
    if user.preferences.get('communication_style') == 'technical':
        prompt += """
        
    COMMUNICATION STYLE: User prefers technical details and precise information.
    """
    elif user.preferences.get('communication_style') == 'simple':
        prompt += """
        
    COMMUNICATION STYLE: User prefers simple, non-technical explanations.
    """
    
    # Interaction history context
    if user.interaction_history:
        recent_interactions = user.interaction_history[-3:]  # Last 3 interactions
        prompt += f"""
        
    RECENT INTERACTION CONTEXT:
    {chr(10).join(f'- {interaction}' for interaction in recent_interactions)}
    
    Keep this context in mind to provide continuity.
    """
    
    return prompt

@contextual_agent.tool
async def get_personalized_recommendations(ctx: RunContext[ContextualDeps]) -> str:
    """Get personalized product recommendations based on user context.
    
    Args:
        None (uses context from user profile)
    """
    user = ctx.deps.user_context
    
    # Mock recommendation engine based on user tier and preferences
    recommendations = {
        "enterprise": [
            "Enterprise Security Suite - Advanced threat protection",
            "Business Analytics Platform - Custom reporting tools",
            "Priority Support Package - 24/7 dedicated support"
        ],
        "premium": [
            "Premium Backup Solution - Automated cloud backup",
            "Advanced Monitoring Tools - Real-time system monitoring",
            "Premium Support Extension - Enhanced support hours"
        ],
        "basic": [
            "Basic Security Package - Essential protection",
            "Standard Backup Service - Weekly cloud backup",
            "Self-Service Training - Online tutorials and guides"
        ]
    }
    
    user_recommendations = recommendations.get(user.tier, recommendations["basic"])
    
    result = f"Personalized recommendations for {user.name} ({user.tier} tier):\n\n"
    for i, rec in enumerate(user_recommendations, 1):
        result += f"{i}. {rec}\n"
    
    # Add preference-based recommendations
    if user.preferences.get('interested_in') == 'security':
        result += "\nBased on your interest in security, we also recommend:\n"
        result += "- Security Audit Service - Comprehensive security assessment\n"
        result += "- Threat Intelligence Feed - Real-time security updates\n"
    
    return result

@contextual_agent.tool
async def update_user_preferences(
    ctx: RunContext[ContextualDeps], 
    preference_key: str, 
    preference_value: str
) -> str:
    """Update user preferences for better personalization.
    
    Args:
        preference_key: The preference to update (e.g., 'communication_style', 'interested_in')
        preference_value: The new value for the preference
    """
    valid_preferences = {
        'communication_style': ['simple', 'technical', 'balanced'],
        'interested_in': ['security', 'performance', 'cost_saving', 'innovation'],
        'contact_method': ['email', 'phone', 'chat'],
        'response_time': ['immediate', 'same_day', 'next_day']
    }
    
    if preference_key not in valid_preferences:
        return f"Invalid preference key. Valid keys: {', '.join(valid_preferences.keys())}"
    
    if preference_value not in valid_preferences[preference_key]:
        return f"Invalid value for {preference_key}. Valid values: {', '.join(valid_preferences[preference_key])}"
    
    # Update the preference (in real app, this would update the database)
    ctx.deps.user_context.preferences[preference_key] = preference_value
    
    return f"✅ Updated {preference_key} to '{preference_value}'. This will improve your future interactions!"

class ContextManager:
    """Manages user and system context for the agents"""
    
    def __init__(self):
        self.active_sessions: Dict[str, UserContext] = {}
        self.system_context = SystemContext(
            current_load=45,
            maintenance_mode=False,
            business_hours=self._is_business_hours(),
            available_agents=12
        )
    
    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours"""
        now = datetime.now().time()
        return time(9, 0) <= now <= time(18, 0)  # 9 AM to 6 PM
    
    def get_user_context(self, user_id: str) -> UserContext:
        """Get or create user context"""
        if user_id not in self.active_sessions:
            # Mock user data - in real app, load from database
            mock_users = {
                "user123": UserContext(
                    user_id="user123",
                    name="John Doe",
                    tier="premium",
                    timezone="EST",
                    preferences={"communication_style": "technical", "interested_in": "security"},
                    interaction_history=[
                        "Asked about laptop specifications",
                        "Requested security software recommendations",
                        "Inquired about enterprise pricing"
                    ],
                    current_session_start=datetime.now()
                ),
                "user456": UserContext(
                    user_id="user456",
                    name="Jane Smith",
                    tier="basic",
                    timezone="PST",
                    preferences={"communication_style": "simple"},
                    interaction_history=["Asked about return policy"],
                    current_session_start=datetime.now()
                )
            }
            
            self.active_sessions[user_id] = mock_users.get(
                user_id, 
                UserContext(
                    user_id=user_id,
                    name="Guest User",
                    tier="basic",
                    timezone="UTC",
                    preferences={},
                    interaction_history=[],
                    current_session_start=datetime.now()
                )
            )
        
        return self.active_sessions[user_id]
    
    def update_system_load(self, new_load: int):
        """Update system load for dynamic prompt adjustment"""
        self.system_context.current_load = new_load
    
    def set_maintenance_mode(self, enabled: bool):
        """Enable/disable maintenance mode"""
        self.system_context.maintenance_mode = enabled

async def demonstrate_contextual_agents():
    context_manager = ContextManager()
    knowledge_base = {
        "return_policy": "30-day return policy for all items",
        "shipping": "Free shipping on orders over $50",
        "warranty": "1-year warranty on all products"
    }
    
    # Test different user contexts
    test_scenarios = [
        ("user123", "I need help with security for my enterprise network"),
        ("user456", "What's your return policy?"),
        ("user123", "Can you recommend some monitoring tools?")
    ]
    
    for user_id, message in test_scenarios:
        print(f"\n{'='*70}")
        print(f"User {user_id}: {message}")
        print('='*70)
        
        # Get user context
        user_context = context_manager.get_user_context(user_id)
        
        # Create dependencies
        deps = ContextualDeps(
            user_context=user_context,
            system_context=context_manager.system_context,
            knowledge_base=knowledge_base
        )
        
        # Run agent
        result = await contextual_agent.run(message, deps=deps)
        print(result.output)
        
        # Update interaction history
        user_context.interaction_history.append(f"Asked: {message}")

if __name__ == "__main__":
    asyncio.run(demonstrate_contextual_agents())
```

**Key Concepts:**
- **Dynamic Prompts**: System prompts that adapt based on context
- **User Personalization**: Customizing behavior based on user profile
- **System State Awareness**: Adjusting behavior based on system conditions
- **Context Continuity**: Maintaining awareness across interactions

---

## 4.2 Output Validation and Quality Control

```python
# output_validation.py
import asyncio
import re
from typing import List, Dict
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, validator
from dataclasses import dataclass
from datetime import datetime

class QualityMetrics(BaseModel):
    """Metrics for response quality assessment"""
    sentiment_score: float  # -1 to 1
    professionalism_score: float  # 0 to 1
    completeness_score: float  # 0 to 1
    accuracy_score: float  # 0 to 1
    
    @validator('sentiment_score')
    def validate_sentiment(cls, v):
        if not -1 <= v <= 1:
            raise ValueError('Sentiment score must be between -1 and 1')
        return v
    
    @validator('professionalism_score', 'completeness_score', 'accuracy_score')
    def validate_scores(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v

@dataclass
class ValidationDeps:
    quality_threshold: float = 0.7
    banned_words: List[str] = None
    required_phrases: List[str] = None
    
    def __post_init__(self):
        if self.banned_words is None:
            self.banned_words = ["stupid", "dumb", "idiot", "hate", "worst"]
        if self.required_phrases is None:
            self.required_phrases = []

class ResponseValidator:
    """Advanced response validation system"""
    
    def __init__(self, deps: ValidationDeps):
        self.deps = deps
        self.validation_history: List[Dict] = []
    
    def validate_content(self, response: str) -> tuple[bool, List[str]]:
        """Validate response content for appropriateness"""
        issues = []
        
        # Check for banned words
        response_lower = response.lower()
        for word in self.deps.banned_words:
            if word in response_lower:
                issues.append(f"Contains inappropriate word: '{word}'")
        
        # Check for required phrases (if any)
        for phrase in self.deps.required_phrases:
            if phrase.lower() not in response_lower:
                issues.append(f"Missing required phrase: '{phrase}'")
        
        # Check length
        if len(response.strip()) < 10:
            issues.append("Response too short")
        
        if len(response) > 2000:
            issues.append("Response too long")
        
        # Check for personal information patterns
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', response):  # SSN pattern
            issues.append("Contains potential SSN")
        
        if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', response):  # Credit card pattern
            issues.append("Contains potential credit card number")
        
        return len(issues) == 0, issues
    
    def calculate_quality_metrics(self, response: str, context: str = "") -> QualityMetrics:
        """Calculate quality metrics for a response"""
        
        # Mock sentiment analysis (in production, use proper NLP library)
        sentiment_score = self._analyze_sentiment(response)
        
        # Professionalism score
        professionalism_score = self._analyze_professionalism(response)
        
        # Completeness score
        completeness_score = self._analyze_completeness(response, context)
        
        # Accuracy score (placeholder - would need domain knowledge)
        accuracy_score = 0.8  # Default assumption
        
        return QualityMetrics(
            sentiment_score=sentiment_score,
            professionalism_score=professionalism_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score
        )
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis"""
        positive_words = ["good", "great", "excellent", "happy", "pleased", "satisfied", "help", "solution"]
        negative_words = ["bad", "terrible", "awful", "angry", "frustrated", "problem", "issue", "complaint"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0  # Neutral
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _analyze_professionalism(self, text: str) -> float:
        """Analyze professionalism of response"""
        professional_indicators = ["please", "thank you", "i understand", "i'd be happy", "certainly", "apologize"]
        unprofessional_indicators = ["whatever", "duh", "obviously", "geez", "ugh"]
        
        text_lower = text.lower()
        professional_count = sum(1 for phrase in professional_indicators if phrase in text_lower)
        unprofessional_count = sum(1 for phrase in unprofessional_indicators if phrase in text_lower)
        
        # Base score
        score = 0.7
        
        # Adjust based on indicators
        score += professional_count * 0.1
        score -= unprofessional_count * 0.2
        
        # Check for proper capitalization and punctuation
        if text[0].isupper() and text.endswith(('.', '!', '?')):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _analyze_completeness(self, text: str, context: str) -> float:
        """Analyze completeness of response"""
        # Basic completeness indicators
        if "?" in context:  # If question was asked
            if not any(indicator in text.lower() for indicator in ["yes", "no", "here", "this", "that", "solution", "answer"]):
                return 0.5  # Might not be answering the question
        
        # Check for actionable information
        action_words = ["can", "will", "should", "would", "please", "contact", "visit", "try"]
        has_action = any(word in text.lower() for word in action_words)
        
        # Length-based completeness
        length_score = min(1.0, len(text) / 100)  # Assume 100 chars is "complete"
        
        return (length_score + (0.3 if has_action else 0)) / 1.3

validated_agent = Agent(
    'openai:gpt-4o',
    deps_type=ValidationDeps,
    system_prompt="""
    You are a professional customer service representative.
    Always be helpful, courteous, and provide accurate information.
    If you're unsure about something, acknowledge it and offer to find out more.
    """,
    retries=3
)

@validated_agent.output_validator
async def validate_response_quality(ctx: RunContext[ValidationDeps], output: str) -> str:
    """Validate response quality and appropriateness"""
    validator = ResponseValidator(ctx.deps)
    
    # Content validation
    is_appropriate, content_issues = validator.validate_content(output)
    
    if not is_appropriate:
        raise ModelRetry(f"Response validation failed: {'; '.join(content_issues)}")
    
    # Quality metrics
    metrics = validator.calculate_quality_metrics(output)
    
    # Calculate overall quality score
    overall_score = (
        (metrics.sentiment_score + 1) / 2 * 0.2 +  # Normalize sentiment to 0-1
        metrics.professionalism_score * 0.4 +
        metrics.completeness_score * 0.3 +
        metrics.accuracy_score * 0.1
    )
    
    # Log validation results
    validation_record = {
        "timestamp": datetime.now().isoformat(),
        "response_length": len(output),
        "metrics": metrics.dict(),
        "overall_score": overall_score,
        "passed": overall_score >= ctx.deps.quality_threshold
    }
    validator.validation_history.append(validation_record)
    
    if overall_score < ctx.deps.quality_threshold:
        raise ModelRetry(
            f"Response quality too low (score: {overall_score:.2f}, threshold: {ctx.deps.quality_threshold}). "
            f"Issues: Sentiment={metrics.sentiment_score:.2f}, "
            f"Professionalism={metrics.professionalism_score:.2f}, "
            f"Completeness={metrics.completeness_score:.2f}"
        )
    
    return output

@validated_agent.tool
async def escalate_to_human(ctx: RunContext[ValidationDeps], reason: str, customer_info: str) -> str:
    """Escalate issue to human agent when AI cannot handle it adequately.
    
    Args:
        reason: Reason for escalation
        customer_info: Relevant customer information
    """
    escalation_id = f"ESC{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    return f"""
    ✅ Issue escalated to human agent
    
    Escalation ID: {escalation_id}
    Reason: {reason}
    Customer Info: {customer_info}
    
    A human agent will contact you within 15 minutes during business hours,
    or within 2 hours outside business hours.
    
    You will receive an email confirmation with your escalation reference number.
    """

async def demonstrate_output_validation():
    # Test different validation scenarios
    test_scenarios = [
        {
            "deps": ValidationDeps(quality_threshold=0.8),
            "message": "What's your return policy?",
            "description": "Standard validation"
        },
        {
            "deps": ValidationDeps(
                quality_threshold=0.9,
                required_phrases=["thank you", "policy"]
            ),
            "message": "I want to return my laptop",
            "description": "High threshold with required phrases"
        },
        {
            "deps": ValidationDeps(
                quality_threshold=0.6,
                banned_words=["stupid", "dumb"]
            ),
            "message": "Your product is terrible and I hate it",
            "description": "Low threshold, testing banned words"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario['description']}")
        print(f"Message: {scenario['message']}")
        print('='*70)
        
        try:
            result = await validated_agent.run(
                scenario['message'], 
                deps=scenario['deps']
            )
            print(f"✅ Response passed validation:")
            print(result.output)
            print(f"Usage: {result.usage()}")
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    asyncio.run(demonstrate_output_validation())
```

---

## 4.3 Streaming Responses and Real-time Processing

```python
# streaming_responses.py
import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Dict, Any
from pydantic_ai import Agent, RunContext
from datetime import datetime
import json

@dataclass
class StreamingDeps:
    user_id: str
    session_id: str
    real_time_data: Dict[str, Any] = None

streaming_agent = Agent(
    'openai:gpt-4o',
    deps_type=StreamingDeps,
    system_prompt="""
    You are a real-time customer service agent. Provide helpful responses and use
    streaming tools when appropriate to show progress for long-running operations.
    """,
    retries=2
)

@streaming_agent.tool
async def process_large_request(
    ctx: RunContext[StreamingDeps], 
    request_type: str,
    complexity: str = "medium"
) -> str:
    """Process a complex request that takes time, with progress updates.
    
    Args:
        request_type: Type of request to process
        complexity: Complexity level (simple, medium, complex)
    """
    # Simulate processing time based on complexity
    processing_times = {
        "simple": 2,
        "medium": 5,
        "complex": 10
    }
    
    total_time = processing_times.get(complexity, 5)
    steps = 4
    step_time = total_time / steps
    
    progress_messages = [
        f"🔄 Starting {request_type} processing...",
        f"📊 Analyzing {request_type} data...",
        f"⚙️ Computing {request_type} results...",
        f"✅ {request_type} processing complete!"
    ]
    
    result_data = []
    
    for i, message in enumerate(progress_messages):
        await asyncio.sleep(step_time)
        result_data.append({
            "step": i + 1,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "progress": ((i + 1) / len(progress_messages)) * 100
        })
    
    return f"""
    Processing Results for {request_type}:
    
    {json.dumps(result_data, indent=2)}
    
    Total processing time: {total_time} seconds
    Complexity: {complexity}
    """

@streaming_agent.tool
async def get_real_time_status(ctx: RunContext[StreamingDeps]) -> str:
    """Get real-time system status information.
    
    Args:
        None
    """
    # Mock real-time data
    real_time_info = {
        "system_status": "operational",
        "current_load": "medium",
        "queue_length": 12,
        "average_response_time": "2.3 seconds",
        "active_agents": 8,
        "timestamp": datetime.now().isoformat()
    }
    
    return f"""
    🔴 LIVE System Status:
    
    Status: {real_time_info['system_status'].upper()}
    Load: {real_time_info['current_load']}
    Queue: {real_time_info['queue_length']} requests
    Avg Response Time: {real_time_info['average_response_time']}
    Active Agents: {real_time_info['active_agents']}
    
    Last Updated: {real_time_info['timestamp']}
    """

class StreamingResponseHandler:
    """Handles streaming responses from agents"""
    
    def __init__(self):
        self.active_streams: Dict[str, asyncio.Queue] = {}
    
    async def create_stream(self, stream_id: str) -> asyncio.Queue:
        """Create a new response stream"""
        queue = asyncio.Queue()
        self.active_streams[stream_id] = queue
        return queue
    
    async def send_to_stream(self, stream_id: str, data: Dict[str, Any]):
        """Send data to a specific stream"""
        if stream_id in self.active_streams:
            await self.active_streams[stream_id].put(data)
    
    async def close_stream(self, stream_id: str):
        """Close a response stream"""
        if stream_id in self.active_streams:
            await self.active_streams[stream_id].put("STREAM_END")
            del self.active_streams[stream_id]
    
    async def stream_agent_response(
        self, 
        agent: Agent, 
        message: str, 
        deps: StreamingDeps,
        stream_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream agent response with real-time updates"""
        
        # Send initial message
        yield {
            "type": "status",
            "message": "Processing your request...",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # For demonstration, we'll simulate streaming by running the agent
            # and yielding intermediate results
            start_time = datetime.now()
            
            yield {
                "type": "progress",
                "message": "Agent is thinking...",
                "progress": 25,
                "timestamp": datetime.now().isoformat()
            }
            
            # Simulate some processing time
            await asyncio.sleep(1)
            
            yield {
                "type": "progress",
                "message": "Generating response...",
                "progress": 50,
                "timestamp": datetime.now().isoformat()
            }
            
            # Run the agent
            result = await agent.run(message, deps=deps)
            
            yield {
                "type": "progress",
                "message": "Finalizing response...",
                "progress": 75,
                "timestamp": datetime.now().isoformat()
            }
            
            # Final response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            yield {
                "type": "response",
                "content": result.output,
                "processing_time": processing_time,
                "usage": result.usage().dict(),
                "progress": 100,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Error processing request: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

class RealTimeCustomerService:
    """Real-time customer service with streaming responses"""
    
    def __init__(self):
        self.stream_handler = StreamingResponseHandler()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def start_session(self, user_id: str) -> str:
        """Start a new customer service session"""
        session_id = f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "start_time": datetime.now(),
            "message_count": 0,
            "status": "active"
        }
        
        return session_id
    
    async def process_message_stream(
        self, 
        session_id: str, 
        message: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process message with streaming response"""
        
        if session_id not in self.active_sessions:
            yield {
                "type": "error",
                "message": "Invalid session ID",
                "timestamp": datetime.now().isoformat()
            }
            return
        
        session = self.active_sessions[session_id]
        session["message_count"] += 1
        
        # Create dependencies
        deps = StreamingDeps(
            user_id=session["user_id"],
            session_id=session_id,
            real_time_data={"session_info": session}
        )
        
        # Stream the response
        async for chunk in self.stream_handler.stream_agent_response(
            streaming_agent, message, deps, session_id
        ):
            yield chunk
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        return self.active_sessions.get(session_id)
    
    def close_session(self, session_id: str):
        """Close a customer service session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = "closed"
            self.active_sessions[session_id]["end_time"] = datetime.now()

async def demonstrate_streaming_responses():
    """Demonstrate streaming responses"""
    
    service = RealTimeCustomerService()
    
    # Start session
    user_id = "user123"
    session_id = service.start_session(user_id)
    
    print(f"🎬 Started streaming session: {session_id}")
    
    # Test messages that trigger different streaming behaviors
    test_messages = [
        "Can you process my complex refund request?",
        "What's the current system status?",
        "I need help with a technical issue that might take some time to resolve"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{'='*60}")
        print(f"Message {i}: {message}")
        print('='*60)
        
        # Process with streaming
        async for chunk in service.process_message_stream(session_id, message):
            chunk_type = chunk.get("type", "unknown")
            
            if chunk_type == "status":
                print(f"📊 Status: {chunk['message']}")
            elif chunk_type == "progress":
                progress = chunk.get("progress", 0)
                print(f"⏳ Progress: {progress}% - {chunk['message']}")
            elif chunk_type == "response":
                print(f"✅ Final Response:")
                print(f"   {chunk['content']}")
                print(f"   Processing time: {chunk['processing_time']:.2f}s")
            elif chunk_type == "error":
                print(f"❌ Error: {chunk['message']}")
        
        # Small delay between messages
        await asyncio.sleep(1)
    
    # Show session summary
    session_info = service.get_session_info(session_id)
    print(f"\n📊 Session Summary:")
    print(f"   Messages processed: {session_info['message_count']}")
    print(f"   Session duration: {datetime.now() - session_info['start_time']}")
    
    # Close session
    service.close_session(session_id)
    print(f"✅ Session {session_id} closed")

if __name__ == "__main__":
    asyncio.run(demonstrate_streaming_responses())
```

**Key Concepts:**
- **Streaming Responses**: Real-time response generation with progress updates
- **Async Generators**: Using Python's async generators for streaming
- **Progress Tracking**: Showing processing progress to users
- **Real-time Updates**: Live status information and updates

---

## Practice Exercises for Module 4

### Exercise 1: Advanced Prompt Engineering
Create a system that generates prompts based on:
- User behavior patterns
- Conversation sentiment analysis
- Business rules and policies
- Time-sensitive context (business hours, promotions, etc.)

### Exercise 2: Multi-layered Validation
Implement a validation system with:
- Content appropriateness checking
- Factual accuracy verification
- Brand voice consistency
- Legal compliance checking

### Exercise 3: Streaming Analytics
Build a streaming system that provides:
- Real-time response quality metrics
- User engagement tracking
- Performance monitoring
- Cost optimization insights

### Exercise 4: Context Evolution
Create a system where context:
- Learns from user interactions
- Adapts to changing business conditions
- Maintains long-term user preferences
- Handles context conflicts intelligently

### Next Steps
Once you've mastered advanced agent features, you'll be ready for **Module 5: Multi-Agent Fundamentals**, where we'll start building systems with multiple coordinating agents.

**Key Takeaways:**
- Dynamic prompts enable context-aware responses
- Output validation ensures quality and safety
- Streaming provides better user experience for long operations
- Quality metrics help improve system performance over time
- Context management is crucial for personalized experiences