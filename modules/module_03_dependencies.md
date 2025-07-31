# Module 3: Dependencies & State Management
## Detailed Code Examples

### Learning Objectives
- Understand the dependency injection system
- Manage stateful agents with RunContext
- Handle async and sync dependencies
- Implement proper resource management

---

## 3.1 Database Integration with Connection Pooling

```python
# database_integration.py
import asyncio
import asyncpg
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
import os
import json
from datetime import datetime

# Database models
class CustomerRecord(BaseModel):
    id: str
    name: str
    email: str
    phone: Optional[str]
    tier: str
    created_at: datetime
    last_activity: Optional[datetime]

class TicketRecord(BaseModel):
    id: str
    customer_id: str
    subject: str
    description: str
    status: str
    priority: str
    created_at: datetime
    updated_at: datetime
    assigned_agent: Optional[str]

# Database connection pool manager
class DatabaseManager:
    def __init__(self, database_url: str, min_connections: int = 5, max_connections: int = 20):
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize the connection pool"""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=self.min_connections,
            max_size=self.max_connections,
            command_timeout=30
        )
        await self._create_tables()
    
    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()
    
    async def _create_tables(self):
        """Create tables if they don't exist (for demo purposes)"""
        async with self.pool.acquire() as conn:
            # Create customers table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    phone VARCHAR(50),
                    tier VARCHAR(50) NOT NULL DEFAULT 'basic',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP
                )
            """)
            
            # Create tickets table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id VARCHAR(50) PRIMARY KEY,
                    customer_id VARCHAR(50) REFERENCES customers(id),
                    subject VARCHAR(500) NOT NULL,
                    description TEXT,
                    status VARCHAR(50) NOT NULL DEFAULT 'open',
                    priority VARCHAR(50) NOT NULL DEFAULT 'medium',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    assigned_agent VARCHAR(255)
                )
            """)
            
            # Insert sample data
            await self._insert_sample_data(conn)
    
    async def _insert_sample_data(self, conn):
        """Insert sample data for demonstration"""
        # Check if data already exists
        existing = await conn.fetchval("SELECT COUNT(*) FROM customers")
        if existing > 0:
            return
        
        # Insert sample customers
        customers_data = [
            ("CUST001", "John Doe", "john@example.com", "+1234567890", "premium"),
            ("CUST002", "Jane Smith", "jane@example.com", "+1234567891", "basic"),
            ("CUST003", "Bob Johnson", "bob@example.com", None, "enterprise"),
        ]
        
        for customer in customers_data:
            await conn.execute("""
                INSERT INTO customers (id, name, email, phone, tier)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (email) DO NOTHING
            """, *customer)
        
        # Insert sample tickets
        tickets_data = [
            ("TICK001", "CUST001", "Laptop not charging", "My laptop battery won't charge", "open", "high"),
            ("TICK002", "CUST002", "Refund request", "Want to return my purchase", "in_progress", "medium"),
            ("TICK003", "CUST003", "Account access", "Cannot log into my account", "resolved", "low"),
        ]
        
        for ticket in tickets_data:
            await conn.execute("""
                INSERT INTO tickets (id, customer_id, subject, description, status, priority)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO NOTHING
            """, *ticket)
    
    async def get_customer_by_email(self, email: str) -> Optional[CustomerRecord]:
        """Get customer by email"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM customers WHERE email = $1", email
            )
            if row:
                return CustomerRecord(**dict(row))
            return None
    
    async def get_customer_tickets(self, customer_id: str) -> List[TicketRecord]:
        """Get all tickets for a customer"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM tickets WHERE customer_id = $1 ORDER BY created_at DESC", 
                customer_id
            )
            return [TicketRecord(**dict(row)) for row in rows]
    
    async def create_ticket(self, customer_id: str, subject: str, description: str, priority: str = "medium") -> str:
        """Create a new support ticket"""
        ticket_id = f"TICK{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tickets (id, customer_id, subject, description, priority)
                VALUES ($1, $2, $3, $4, $5)
            """, ticket_id, customer_id, subject, description, priority)
        
        return ticket_id
    
    async def update_ticket_status(self, ticket_id: str, status: str, assigned_agent: Optional[str] = None) -> bool:
        """Update ticket status"""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE tickets 
                SET status = $1, updated_at = CURRENT_TIMESTAMP, assigned_agent = $2
                WHERE id = $3
            """, status, assigned_agent, ticket_id)
            
            return result == "UPDATE 1"

# Dependencies for database-connected agents
@dataclass
class DatabaseDeps:
    db: DatabaseManager
    current_agent: str = "AI_Agent_001"

# Create database-connected customer service agent
database_agent = Agent(
    'openai:gpt-4o',
    deps_type=DatabaseDeps,
    system_prompt="""
    You are a customer service agent with access to a customer database and ticketing system.
    Always look up customer information before helping them.
    Create tickets for issues that require follow-up.
    Update existing tickets when appropriate.
    """,
    retries=2
)

@database_agent.tool
async def lookup_customer_db(ctx: RunContext[DatabaseDeps], email: str) -> str:
    """Look up customer information from the database.
    
    Args:
        email: Customer's email address
    """
    try:
        customer = await ctx.deps.db.get_customer_by_email(email)
        if not customer:
            return f"No customer found with email: {email}"
        
        # Get customer's tickets
        tickets = await ctx.deps.db.get_customer_tickets(customer.id)
        
        result = f"""
        Customer Information:
        - Name: {customer.name}
        - Email: {customer.email}
        - Phone: {customer.phone or 'Not provided'}
        - Tier: {customer.tier}
        - Member since: {customer.created_at.strftime('%B %d, %Y')}
        - Last activity: {customer.last_activity.strftime('%B %d, %Y') if customer.last_activity else 'Never'}
        
        Recent Tickets ({len(tickets)}):
        """
        
        for ticket in tickets[:3]:  # Show last 3 tickets
            result += f"""
        - {ticket.id}: {ticket.subject} ({ticket.status})
          Created: {ticket.created_at.strftime('%B %d, %Y')}
        """
        
        return result
        
    except Exception as e:
        return f"Error looking up customer: {str(e)}"

@database_agent.tool
async def create_support_ticket(
    ctx: RunContext[DatabaseDeps], 
    customer_email: str, 
    subject: str, 
    description: str,
    priority: str = "medium"
) -> str:
    """Create a new support ticket for a customer.
    
    Args:
        customer_email: Customer's email address
        subject: Brief description of the issue
        description: Detailed description of the issue
        priority: Priority level (low, medium, high, urgent)
    """
    try:
        # Look up customer
        customer = await ctx.deps.db.get_customer_by_email(customer_email)
        if not customer:
            return f"Cannot create ticket: Customer not found with email {customer_email}"
        
        # Validate priority
        valid_priorities = ["low", "medium", "high", "urgent"]
        if priority not in valid_priorities:
            priority = "medium"
        
        # Create ticket
        ticket_id = await ctx.deps.db.create_ticket(
            customer.id, subject, description, priority
        )
        
        return f"""
        ✅ Support ticket created successfully!
        
        Ticket ID: {ticket_id}
        Customer: {customer.name}
        Subject: {subject}
        Priority: {priority}
        
        We'll get back to you within:
        - Urgent: 1 hour
        - High: 4 hours  
        - Medium: 24 hours
        - Low: 48 hours
        """
        
    except Exception as e:
        return f"Error creating ticket: {str(e)}"

@database_agent.tool
async def update_ticket(
    ctx: RunContext[DatabaseDeps],
    ticket_id: str,
    new_status: str,
    notes: Optional[str] = None
) -> str:
    """Update the status of an existing ticket.
    
    Args:
        ticket_id: The ticket ID to update
        new_status: New status (open, in_progress, resolved, closed)
        notes: Optional notes about the update
    """
    try:
        valid_statuses = ["open", "in_progress", "resolved", "closed"]
        if new_status not in valid_statuses:
            return f"Invalid status. Valid options: {', '.join(valid_statuses)}"
        
        success = await ctx.deps.db.update_ticket_status(
            ticket_id, new_status, ctx.deps.current_agent
        )
        
        if success:
            result = f"✅ Ticket {ticket_id} updated to '{new_status}'"
            if notes:
                result += f"\nNotes: {notes}"
            return result
        else:
            return f"❌ Ticket {ticket_id} not found or could not be updated"
            
    except Exception as e:
        return f"Error updating ticket: {str(e)}"

# Example usage and lifecycle management
class CustomerServiceSystem:
    def __init__(self, database_url: str):
        self.db = DatabaseManager(database_url)
        self.deps = DatabaseDeps(self.db)
    
    async def startup(self):
        """Initialize the system"""
        await self.db.initialize()
        print("✅ Customer service system initialized")
    
    async def shutdown(self):
        """Cleanup resources"""
        await self.db.close()
        print("✅ Customer service system shut down")
    
    @asynccontextmanager
    async def get_agent_session(self):
        """Context manager for agent sessions"""
        try:
            yield database_agent, self.deps
        except Exception as e:
            print(f"Session error: {e}")
            raise

async def demonstrate_database_integration():
    # Use SQLite for demo (replace with PostgreSQL in production)
    database_url = "postgresql://user:password@localhost/customerdb"
    
    system = CustomerServiceSystem(database_url)
    
    try:
        await system.startup()
        
        # Test interactions
        test_scenarios = [
            "Hi, I'm john@example.com and I need help with my laptop",
            "I want to check the status of my existing tickets",
            "My laptop is still not working, can you escalate my ticket TICK001?"
        ]
        
        async with system.get_agent_session() as (agent, deps):
            for scenario in test_scenarios:
                print(f"\n{'='*60}")
                print(f"Customer: {scenario}")
                print('='*60)
                
                result = await agent.run(scenario, deps=deps)
                print(result.output)
                
    finally:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(demonstrate_database_integration())
```

**Key Concepts:**
- **Connection Pooling**: Efficient database connection management
- **Dependency Injection**: Passing database access through RunContext
- **Resource Management**: Proper startup/shutdown lifecycle
- **Error Handling**: Graceful handling of database errors

---

## 3.2 Redis Cache Integration

```python
# cache_integration.py
import asyncio
import redis.asyncio as redis
import json
from typing import Optional, Any, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pydantic_ai import Agent, RunContext
import hashlib

class CacheManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        print("✅ Redis cache connected")
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        if not self.redis_client:
            return False
        
        try:
            serialized = json.dumps(value, default=str)
            await self.redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str):
        """Delete key from cache"""
        if self.redis_client:
            await self.redis_client.delete(key)

@dataclass
class CachedDeps:
    cache: CacheManager
    db: DatabaseManager  # From previous example

cached_agent = Agent(
    'openai:gpt-4o',
    deps_type=CachedDeps,
    system_prompt="""
    You are an efficient customer service agent that uses caching to provide fast responses.
    Always check cache first for frequently requested information.
    """,
    retries=2
)

@cached_agent.tool
async def get_customer_info_cached(ctx: RunContext[CachedDeps], email: str) -> str:
    """Get customer information with caching.
    
    Args:
        email: Customer's email address
    """
    # Create cache key
    cache_key = f"customer:{hashlib.md5(email.encode()).hexdigest()}"
    
    # Try cache first
    cached_data = await ctx.deps.cache.get(cache_key)
    if cached_data:
        cached_data['source'] = 'cache'
        return f"Customer Info (cached): {json.dumps(cached_data, indent=2)}"
    
    # If not in cache, get from database
    try:
        customer = await ctx.deps.db.get_customer_by_email(email)
        if not customer:
            return f"No customer found with email: {email}"
        
        # Prepare data for caching
        customer_data = {
            'id': customer.id,
            'name': customer.name,
            'email': customer.email,
            'tier': customer.tier,
            'created_at': customer.created_at.isoformat(),
            'source': 'database'
        }
        
        # Cache for 1 hour
        await ctx.deps.cache.set(cache_key, customer_data, ttl=3600)
        
        return f"Customer Info (from DB): {json.dumps(customer_data, indent=2)}"
        
    except Exception as e:
        return f"Error retrieving customer info: {str(e)}"

@cached_agent.tool
async def get_faq_answer(ctx: RunContext[CachedDeps], question: str) -> str:
    """Get FAQ answers with caching.
    
    Args:
        question: The FAQ question
    """
    # Normalize question for cache key
    normalized_q = question.lower().strip()
    cache_key = f"faq:{hashlib.md5(normalized_q.encode()).hexdigest()}"
    
    # Check cache
    cached_answer = await ctx.deps.cache.get(cache_key)
    if cached_answer:
        return f"FAQ Answer (cached): {cached_answer}"
    
    # Mock FAQ database lookup
    faq_db = {
        "return policy": "You can return items within 30 days of purchase for a full refund.",
        "shipping time": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 days.",
        "warranty": "All products come with a 1-year manufacturer warranty.",
        "payment methods": "We accept credit cards, PayPal, and Apple Pay.",
        "contact hours": "Customer service is available Monday-Friday 9AM-6PM EST."
    }
    
    # Simple keyword matching
    answer = None
    for key, value in faq_db.items():
        if key in normalized_q:
            answer = value
            break
    
    if not answer:
        answer = "I don't have information about that. Please contact our support team for assistance."
    
    # Cache for 24 hours
    await ctx.deps.cache.set(cache_key, answer, ttl=86400)
    
    return f"FAQ Answer: {answer}"

async def demonstrate_caching():
    # Initialize components
    cache = CacheManager()
    db = DatabaseManager("postgresql://user:password@localhost/customerdb")
    
    try:
        await cache.initialize()
        await db.initialize()
        
        deps = CachedDeps(cache, db)
        
        # Test caching behavior
        print("=== First Request (will hit database) ===")
        result1 = await cached_agent.run(
            "Look up customer john@example.com", 
            deps=deps
        )
        print(result1.output)
        
        print("\n=== Second Request (should hit cache) ===")
        result2 = await cached_agent.run(
            "Look up customer john@example.com", 
            deps=deps
        )
        print(result2.output)
        
        print("\n=== FAQ Test ===")
        faq_result = await cached_agent.run(
            "What's your return policy?", 
            deps=deps
        )
        print(faq_result.output)
        
    finally:
        await cache.close()
        await db.close()

if __name__ == "__main__":
    asyncio.run(demonstrate_caching())
```

---

## 3.3 Session State Management

```python
# session_state_management.py
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
import json
import uuid

class SessionData(BaseModel):
    """Session data structure"""
    session_id: str
    customer_id: Optional[str] = None
    customer_email: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = []
    context_data: Dict[str, Any] = {}
    created_at: datetime
    last_activity: datetime
    active: bool = True

class SessionManager:
    """Manages user sessions and state"""
    
    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, SessionData] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start session management"""
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        print("✅ Session manager started")
    
    async def stop(self):
        """Stop session management"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        print("✅ Session manager stopped")
    
    def create_session(self, customer_email: Optional[str] = None) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = SessionData(
            session_id=session_id,
            customer_email=customer_email,
            created_at=now,
            last_activity=now
        )
        
        self.sessions[session_id] = session
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data"""
        session = self.sessions.get(session_id)
        if session and session.active:
            # Update last activity
            session.last_activity = datetime.now()
            return session
        return None
    
    def update_session(self, session_id: str, **updates):
        """Update session data"""
        session = self.sessions.get(session_id)
        if session:
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            session.last_activity = datetime.now()
    
    def add_to_conversation(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add message to conversation history"""
        session = self.sessions.get(session_id)
        if session:
            session.conversation_history.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            })
            session.last_activity = datetime.now()
    
    def set_context_data(self, session_id: str, key: str, value: Any):
        """Set context data for session"""
        session = self.sessions.get(session_id)
        if session:
            session.context_data[key] = value
            session.last_activity = datetime.now()
    
    def get_context_data(self, session_id: str, key: str) -> Any:
        """Get context data from session"""
        session = self.sessions.get(session_id)
        if session:
            return session.context_data.get(key)
        return None
    
    async def _cleanup_expired_sessions(self):
        """Periodically clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                now = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if now - session.last_activity > self.session_timeout:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.sessions[session_id]
                
                if expired_sessions:
                    print(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
                    
            except Exception as e:
                print(f"Error in session cleanup: {e}")

@dataclass
class StatefulDeps:
    session_manager: SessionManager
    session_id: str
    db: Optional[DatabaseManager] = None

# Create stateful agent
stateful_agent = Agent(
    'openai:gpt-4o',
    deps_type=StatefulDeps,
    system_prompt="""
    You are a stateful customer service agent that maintains conversation context.
    Use session tools to remember customer information and conversation history.
    Provide personalized responses based on previous interactions.
    """,
    retries=2
)

@stateful_agent.tool
async def remember_customer_info(
    ctx: RunContext[StatefulDeps], 
    customer_email: str, 
    additional_info: str = ""
) -> str:
    """Remember customer information for this session.
    
    Args:
        customer_email: Customer's email address
        additional_info: Any additional information to remember
    """
    session_manager = ctx.deps.session_manager
    session_id = ctx.deps.session_id
    
    # Update session with customer info
    session_manager.update_session(
        session_id, 
        customer_email=customer_email
    )
    
    # Store additional info in context
    if additional_info:
        session_manager.set_context_data(session_id, "additional_info", additional_info)
    
    return f"✅ Remembered customer email: {customer_email}"

@stateful_agent.tool
async def recall_conversation_context(ctx: RunContext[StatefulDeps]) -> str:
    """Recall previous conversation context.
    
    Args:
        None
    """
    session_manager = ctx.deps.session_manager
    session_id = ctx.deps.session_id
    
    session = session_manager.get_session(session_id)
    if not session:
        return "No session context available"
    
    context_info = f"""
    Session Context:
    - Session ID: {session.session_id}
    - Customer Email: {session.customer_email or 'Not provided'}
    - Session Duration: {datetime.now() - session.created_at}
    - Messages in History: {len(session.conversation_history)}
    
    Recent Conversation:
    """
    
    # Show last 3 messages
    recent_messages = session.conversation_history[-3:] if session.conversation_history else []
    for msg in recent_messages:
        context_info += f"- {msg['role']}: {msg['content'][:100]}...\n"
    
    # Show context data
    if session.context_data:
        context_info += f"\nContext Data: {json.dumps(session.context_data, indent=2)}"
    
    return context_info

@stateful_agent.tool
async def set_conversation_context(
    ctx: RunContext[StatefulDeps], 
    key: str, 
    value: str
) -> str:
    """Set context information for future reference.
    
    Args:
        key: Context key
        value: Context value
    """
    session_manager = ctx.deps.session_manager
    session_id = ctx.deps.session_id
    
    session_manager.set_context_data(session_id, key, value)
    
    return f"✅ Set context: {key} = {value}"

@stateful_agent.tool
async def get_conversation_summary(ctx: RunContext[StatefulDeps]) -> str:
    """Get a summary of the current conversation.
    
    Args:
        None
    """
    session_manager = ctx.deps.session_manager
    session_id = ctx.deps.session_id
    
    session = session_manager.get_session(session_id)
    if not session or not session.conversation_history:
        return "No conversation history available"
    
    # Count message types
    user_messages = len([msg for msg in session.conversation_history if msg['role'] == 'user'])
    agent_messages = len([msg for msg in session.conversation_history if msg['role'] == 'assistant'])
    
    # Get topics discussed (simple keyword extraction)
    all_text = " ".join([msg['content'] for msg in session.conversation_history])
    keywords = ["order", "refund", "shipping", "account", "technical", "billing", "support"]
    topics = [kw for kw in keywords if kw in all_text.lower()]
    
    return f"""
    Conversation Summary:
    - Total Messages: {len(session.conversation_history)}
    - User Messages: {user_messages}
    - Agent Messages: {agent_messages}
    - Duration: {datetime.now() - session.created_at}
    - Topics Discussed: {', '.join(topics) if topics else 'General inquiry'}
    - Customer: {session.customer_email or 'Anonymous'}
    """

class StatefulCustomerService:
    """Stateful customer service system"""
    
    def __init__(self):
        self.session_manager = SessionManager(session_timeout_minutes=30)
        self.active_sessions: Dict[str, str] = {}  # customer_email -> session_id
    
    async def start(self):
        """Start the service"""
        await self.session_manager.start()
        print("✅ Stateful customer service started")
    
    async def stop(self):
        """Stop the service"""
        await self.session_manager.stop()
        print("✅ Stateful customer service stopped")
    
    def start_conversation(self, customer_email: Optional[str] = None) -> str:
        """Start a new conversation"""
        # Check if customer has existing session
        if customer_email and customer_email in self.active_sessions:
            existing_session_id = self.active_sessions[customer_email]
            if self.session_manager.get_session(existing_session_id):
                return existing_session_id
        
        # Create new session
        session_id = self.session_manager.create_session(customer_email)
        
        if customer_email:
            self.active_sessions[customer_email] = session_id
        
        return session_id
    
    async def process_message(self, session_id: str, message: str) -> str:
        """Process a message in the context of a session"""
        # Add user message to conversation history
        self.session_manager.add_to_conversation(session_id, "user", message)
        
        # Create dependencies
        deps = StatefulDeps(
            session_manager=self.session_manager,
            session_id=session_id
        )
        
        try:
            # Process with agent
            result = await stateful_agent.run(message, deps=deps)
            
            # Add agent response to conversation history
            self.session_manager.add_to_conversation(
                session_id, 
                "assistant", 
                result.output,
                {"usage": result.usage().dict()}
            )
            
            return result.output
            
        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error: {str(e)}"
            self.session_manager.add_to_conversation(session_id, "assistant", error_msg)
            return error_msg
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        session = self.session_manager.get_session(session_id)
        if session:
            return {
                "session_id": session.session_id,
                "customer_email": session.customer_email,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "message_count": len(session.conversation_history),
                "context_keys": list(session.context_data.keys())
            }
        return None

async def demonstrate_stateful_service():
    """Demonstrate stateful customer service"""
    
    service = StatefulCustomerService()
    
    try:
        await service.start()
        
        # Start conversation
        session_id = service.start_conversation("john@example.com")
        print(f"Started session: {session_id}")
        
        # Simulate conversation
        conversation = [
            "Hi, I'm John and I need help with my order",
            "My order number is #12345 and it hasn't arrived yet",
            "Can you check the status and let me know when it will arrive?",
            "Also, what's your return policy if I'm not satisfied?",
            "Thanks for the help! Can you summarize what we discussed?"
        ]
        
        for message in conversation:
            print(f"\n{'='*60}")
            print(f"Customer: {message}")
            print('='*60)
            
            response = await service.process_message(session_id, message)
            print(f"Agent: {response}")
            
            # Show session info after each exchange
            session_info = service.get_session_info(session_id)
            print(f"Session Info: {session_info['message_count']} messages, Context: {session_info['context_keys']}")
        
    finally:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(demonstrate_stateful_service())
```

**Key Concepts:**
- **Session Management**: Tracking user sessions and conversation state
- **Context Persistence**: Maintaining context across multiple interactions
- **Memory Management**: Automatic cleanup of expired sessions
- **Stateful Interactions**: Building on previous conversation context

---

## Practice Exercises for Module 3

### Exercise 1: Database Optimization
Enhance the database integration with:
- Query optimization and indexing
- Connection pool monitoring
- Transaction management
- Database migration handling

### Exercise 2: Advanced Caching
Implement advanced caching strategies:
- Cache invalidation patterns
- Distributed caching with Redis Cluster
- Cache warming strategies
- Performance metrics

### Exercise 3: Session Security
Add security features to session management:
- Session encryption
- CSRF protection
- Session hijacking prevention
- Audit logging

### Exercise 4: State Synchronization
Build a system that synchronizes state across:
- Multiple agent instances
- Different services
- Database and cache consistency
- Conflict resolution

### Next Steps
Once you've mastered dependencies and state management, you'll be ready for **Module 4: Advanced Agent Features**, where we'll explore dynamic prompts, output validation, and streaming capabilities.

**Key Takeaways:**
- Use dependency injection for clean separation of concerns
- Implement proper resource lifecycle management
- Cache frequently accessed data for performance
- Maintain session state for personalized interactions
- Handle errors gracefully at all dependency levels
- Monitor and clean up resources automatically