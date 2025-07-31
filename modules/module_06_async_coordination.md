        # Prepare analysis context
        analysis_context = f"""
        Customer Message: {customer_message}
        
        Customer Information:
        - Name: {customer_info.get('name', 'Unknown')}
        - Tier: {customer_info.get('tier', 'basic')}
        - Account Status: {customer_info.get('status', 'active')}
        - History: {customer_info.get('interaction_count', 0)} previous interactions
        """
        
        print(f"🚀 Starting concurrent analysis for request {request_id}")
        
        # Create analysis tasks
        tasks = {}
        for agent_type, agent in self.agents.items():
            task = asyncio.create_task(
                self._run_analysis_with_timeout(
                    agent, analysis_context, deps, agent_type
                ),
                name=f"analysis_{agent_type}"
            )
            tasks[agent_type] = task
        
        # Wait for all analyses to complete
        results = {}
        completed_tasks = 0
        total_tasks = len(tasks)
        
        try:
            # Use asyncio.gather with return_exceptions=True for better error handling
            task_results = await asyncio.gather(
                *tasks.values(),
                return_exceptions=True
            )
            
            # Process results
            for (agent_type, task), result in zip(tasks.items(), task_results):
                if isinstance(result, Exception):
                    print(f"❌ {agent_type} analysis failed: {result}")
                    results[agent_type] = {
                        "error": str(result),
                        "status": "failed"
                    }
                else:
                    print(f"✅ {agent_type} analysis completed")
                    results[agent_type] = result
                    completed_tasks += 1
        
        except Exception as e:
            print(f"❌ Concurrent analysis failed: {e}")
            # Cancel any remaining tasks
            for task in tasks.values():
                if not task.done():
                    task.cancel()
        
        total_time = time.time() - start_time
        
        # Compile comprehensive result
        return {
            "request_id": request_id,
            "total_processing_time": total_time,
            "completed_analyses": completed_tasks,
            "total_analyses": total_tasks,
            "success_rate": completed_tasks / total_tasks,
            "results": results,
            "summary": self._generate_analysis_summary(results),
            "recommended_actions": self._extract_recommended_actions(results)
        }
    
    async def _run_analysis_with_timeout(
        self, 
        agent: Agent, 
        context: str, 
        deps: ConcurrentDeps, 
        agent_type: str
    ) -> Dict[str, Any]:
        """Run analysis with timeout and error handling"""
        
        analysis_start = time.time()
        
        try:
            # Run analysis with timeout
            result = await asyncio.wait_for(
                agent.run(context, deps=deps),
                timeout=deps.timeout
            )
            
            processing_time = time.time() - analysis_start
            
            # Update the result with timing information
            if hasattr(result.output, 'processing_time'):
                result.output.processing_time = processing_time
            
            return {
                "status": "completed",
                "result": result.output.dict() if hasattr(result.output, 'dict') else result.output,
                "processing_time": processing_time,
                "usage": result.usage().dict()
            }
            
        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "error": f"Analysis timed out after {deps.timeout} seconds",
                "processing_time": time.time() - analysis_start
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - analysis_start
            }
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of all analysis results"""
        summary_parts = []
        
        for agent_type, result in results.items():
            if result.get("status") == "completed":
                analysis_data = result.get("result", {})
                if isinstance(analysis_data, dict):
                    confidence = analysis_data.get("confidence", 0)
                    requires_action = analysis_data.get("requires_action", False)
                    summary_parts.append(
                        f"{agent_type.title()}: {confidence:.1%} confidence, "
                        f"{'Action required' if requires_action else 'No immediate action'}"
                    )
        
        return "; ".join(summary_parts) if summary_parts else "No successful analyses completed"
    
    def _extract_recommended_actions(self, results: Dict[str, Any]) -> List[str]:
        """Extract and prioritize recommended actions from all analyses"""
        all_actions = []
        
        for agent_type, result in results.items():
            if result.get("status") == "completed":
                analysis_data = result.get("result", {})
                if isinstance(analysis_data, dict):
                    recommendations = analysis_data.get("recommendations", [])
                    for rec in recommendations:
                        all_actions.append(f"{agent_type.title()}: {rec}")
        
        return all_actions[:10]  # Return top 10 actions

# Advanced concurrent processing with task coordination
class TaskCoordinator:
    """Coordinates complex multi-agent workflows with dependencies"""
    
    def __init__(self):
        self.analysis_system = ConcurrentAnalysisSystem()
        self.task_queues: Dict[str, asyncio.Queue] = {
            "high_priority": asyncio.Queue(),
            "normal_priority": asyncio.Queue(),
            "low_priority": asyncio.Queue()
        }
        self.worker_pools: Dict[str, List[asyncio.Task]] = {}
        self.active_workers = 0
        self.max_workers = 10
    
    async def start_workers(self):
        """Start worker pools for different priority levels"""
        for priority, queue in self.task_queues.items():
            worker_count = 2 if priority == "high_priority" else 1
            workers = []
            
            for i in range(worker_count):
                worker = asyncio.create_task(
                    self._worker(priority, queue),
                    name=f"worker_{priority}_{i}"
                )
                workers.append(worker)
            
            self.worker_pools[priority] = workers
        
        print(f"✅ Started workers for {len(self.task_queues)} priority levels")
    
    async def stop_workers(self):
        """Stop all workers gracefully"""
        # Cancel all workers
        for workers in self.worker_pools.values():
            for worker in workers:
                worker.cancel()
        
        # Wait for cancellation
        all_workers = [w for workers in self.worker_pools.values() for w in workers]
        await asyncio.gather(*all_workers, return_exceptions=True)
        
        print("✅ All workers stopped")
    
    async def _worker(self, priority: str, queue: asyncio.Queue):
        """Worker that processes tasks from a priority queue"""
        worker_name = f"worker_{priority}"
        
        while True:
            try:
                # Get task from queue
                task_data = await queue.get()
                
                if task_data is None:  # Shutdown signal
                    break
                
                print(f"🔄 {worker_name} processing {task_data['id']}")
                
                # Process the task
                start_time = time.time()
                result = await self.analysis_system.analyze_customer_issue(
                    task_data['message'],
                    task_data['customer_info'],
                    priority.split('_')[0]  # Extract priority level
                )
                
                processing_time = time.time() - start_time
                
                # Store result (in production, save to database)
                task_data['result'] = result
                task_data['processing_time'] = processing_time
                task_data['status'] = 'completed'
                
                print(f"✅ {worker_name} completed {task_data['id']} in {processing_time:.2f}s")
                
                # Mark task as done
                queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ {worker_name} error: {e}")
                queue.task_done()
    
    async def submit_analysis_request(
        self, 
        message: str, 
        customer_info: Dict,
        priority: str = "normal"
    ) -> str:
        """Submit analysis request to appropriate priority queue"""
        
        request_id = f"TASK{datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]}"
        
        task_data = {
            "id": request_id,
            "message": message,
            "customer_info": customer_info,
            "submitted_at": datetime.now().isoformat(),
            "status": "queued"
        }
        
        # Select appropriate queue
        queue_key = f"{priority}_priority"
        if queue_key not in self.task_queues:
            queue_key = "normal_priority"
        
        await self.task_queues[queue_key].put(task_data)
        
        print(f"📥 Submitted {request_id} to {queue_key} queue")
        return request_id

async def demonstrate_concurrent_agents():
    """Demonstrate concurrent agent execution patterns"""
    
    print("🎬 Concurrent Multi-Agent Analysis System Demo")
    print("="*60)
    
    # Initialize system
    analysis_system = ConcurrentAnalysisSystem()
    
    # Test scenarios with different complexity levels
    test_scenarios = [
        {
            "message": "URGENT: Our entire email system is down! All 500 employees can't access email and we have critical client communications waiting. This is costing us thousands per hour!",
            "customer_info": {
                "name": "Tech Corp Inc",
                "tier": "enterprise",
                "status": "active",
                "interaction_count": 12
            },
            "description": "High-impact enterprise emergency"
        },
        {
            "message": "I'm having trouble with my laptop battery. It seems to drain faster than usual lately.",
            "customer_info": {
                "name": "John Smith",
                "tier": "basic",
                "status": "active",
                "interaction_count": 2
            },
            "description": "Simple technical issue"
        },
        {
            "message": "I was overcharged $500 on my last invoice and nobody is responding to my emails. I'm considering switching to your competitor if this isn't resolved immediately!",
            "customer_info": {
                "name": "Sarah Wilson",
                "tier": "premium",
                "status": "active",
                "interaction_count": 8
            },
            "description": "Billing dispute with churn risk"
        }
    ]
    
    # Run concurrent analysis for all scenarios
    analysis_tasks = []
    for i, scenario in enumerate(test_scenarios):
        print(f"\n📋 Scenario {i+1}: {scenario['description']}")
        print(f"Customer: {scenario['message'][:100]}...")
        
        task = asyncio.create_task(
            analysis_system.analyze_customer_issue(
                scenario['message'],
                scenario['customer_info'],
                "high" if "URGENT" in scenario['message'] else "normal"
            ),
            name=f"scenario_{i+1}"
        )
        analysis_tasks.append((task, scenario['description']))
    
    # Wait for all analyses to complete
    print(f"\n⏳ Running {len(analysis_tasks)} concurrent analyses...")
    
    results = await asyncio.gather(
        *[task for task, _ in analysis_tasks],
        return_exceptions=True
    )
    
    # Display results
    print(f"\n📊 ANALYSIS RESULTS")
    print("="*60)
    
    for (task, description), result in zip(analysis_tasks, results):
        print(f"\n🔍 {description}:")
        
        if isinstance(result, Exception):
            print(f"❌ Analysis failed: {result}")
            continue
        
        print(f"  ⏱️  Total time: {result['total_processing_time']:.2f}s")
        print(f"  ✅ Success rate: {result['success_rate']:.1%}")
        print(f"  📋 Summary: {result['summary']}")
        
        if result['recommended_actions']:
            print(f"  🎯 Top actions:")
            for action in result['recommended_actions'][:3]:
                print(f"     • {action}")
    
    # Demonstrate task coordination system
    print(f"\n" + "="*60)
    print("🔄 TASK COORDINATION SYSTEM DEMO")
    print("="*60)
    
    coordinator = TaskCoordinator()
    
    try:
        # Start workers
        await coordinator.start_workers()
        
        # Submit multiple requests with different priorities
        request_ids = []
        
        priorities = ["high", "normal", "low"]
        for i, (scenario, priority) in enumerate(zip(test_scenarios, priorities)):
            request_id = await coordinator.submit_analysis_request(
                scenario['message'],
                scenario['customer_info'],
                priority
            )
            request_ids.append(request_id)
        
        # Wait for processing (in production, you'd check status differently)
        await asyncio.sleep(3)
        
        print(f"📈 Processing completed for {len(request_ids)} requests")
        
    finally:
        await coordinator.stop_workers()

if __name__ == "__main__":
    asyncio.run(demonstrate_concurrent_agents())
```

---

## 6.2 Async Communication Patterns and State Sharing

```python
# async_communication.py
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from datetime import datetime
from enum import Enum
import json
import weakref

class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    STATUS_UPDATE = "status_update"
    ERROR = "error"

class AgentMessage(BaseModel):
    """Message structure for inter-agent communication"""
    id: str
    sender: str
    recipient: Optional[str] = None  # None for broadcasts
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None  # For request/response correlation
    priority: int = 0  # Higher number = higher priority

@dataclass
class SharedState:
    """Thread-safe shared state between agents"""
    data: Dict[str, Any] = field(default_factory=dict)
    locks: Dict[str, asyncio.Lock] = field(default_factory=dict)
    
    async def get(self, key: str) -> Any:
        """Get value from shared state"""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        
        async with self.locks[key]:
            return self.data.get(key)
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in shared state"""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        
        async with self.locks[key]:
            self.data[key] = value
    
    async def update(self, key: str, updater: Callable[[Any], Any]) -> Any:
        """Atomically update value in shared state"""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        
        async with self.locks[key]:
            current_value = self.data.get(key)
            new_value = updater(current_value)
            self.data[key] = new_value
            return new_value

class MessageBus:
    """Async message bus for agent communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}
        self.message_history: List[AgentMessage] = []
        self.max_history = 1000
    
    async def subscribe(self, agent_id: str) -> asyncio.Queue:
        """Subscribe an agent to receive messages"""
        queue = asyncio.Queue(maxsize=100)
        
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        
        self.subscribers[agent_id].append(queue)
        return queue
    
    async def unsubscribe(self, agent_id: str, queue: asyncio.Queue):
        """Unsubscribe an agent from messages"""
        if agent_id in self.subscribers:
            try:
                self.subscribers[agent_id].remove(queue)
                if not self.subscribers[agent_id]:
                    del self.subscribers[agent_id]
            except ValueError:
                pass  # Queue was already removed
    
    async def publish(self, message: AgentMessage):
        """Publish message to appropriate subscribers"""
        # Add to history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        # Determine recipients
        recipients = []
        
        if message.recipient:
            # Direct message
            recipients = [message.recipient]
        else:
            # Broadcast to all except sender
            recipients = [agent_id for agent_id in self.subscribers.keys() 
                         if agent_id != message.sender]
        
        # Send to recipients
        for recipient in recipients:
            if recipient in self.subscribers:
                for queue in self.subscribers[recipient]:
                    try:
                        await asyncio.wait_for(queue.put(message), timeout=1.0)
                    except asyncio.TimeoutError:
                        print(f"⚠️  Message queue full for {recipient}")
    
    def get_message_history(self, limit: int = 50) -> List[AgentMessage]:
        """Get recent message history"""
        return self.message_history[-limit:]

@dataclass
class CommunicatingAgentDeps:
    agent_id: str
    message_bus: MessageBus
    shared_state: SharedState
    message_queue: Optional[asyncio.Queue] = None

class CommunicatingAgent:
    """Agent that can communicate with other agents via message bus"""
    
    def __init__(self, agent_id: str, agent: Agent, message_bus: MessageBus, shared_state: SharedState):
        self.agent_id = agent_id
        self.agent = agent
        self.message_bus = message_bus
        self.shared_state = shared_state
        self.message_queue: Optional[asyncio.Queue] = None
        self.running = False
        self.message_handler_task: Optional[asyncio.Task] = None
        
        # Add communication tools to the agent
        self._add_communication_tools()
    
    def _add_communication_tools(self):
        """Add communication tools to the agent"""
        
        @self.agent.tool
        async def send_message_to_agent(
            ctx: RunContext[CommunicatingAgentDeps],
            recipient: str,
            message_content: str,
            message_type: str = "request"
        ) -> str:
            """Send a message to another agent.
            
            Args:
                recipient: ID of the recipient agent
                message_content: Content of the message
                message_type: Type of message (request, response, broadcast, status_update)
            """
            message = AgentMessage(
                id=f"MSG{datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]}",
                sender=ctx.deps.agent_id,
                recipient=recipient,
                message_type=MessageType(message_type),
                content={"text": message_content},
                timestamp=datetime.now()
            )
            
            await ctx.deps.message_bus.publish(message)
            return f"✅ Message sent to {recipient}: {message_content[:50]}..."
        
        @self.agent.tool
        async def broadcast_message(
            ctx: RunContext[CommunicatingAgentDeps],
            message_content: str,
            message_type: str = "broadcast"
        ) -> str:
            """Broadcast a message to all other agents.
            
            Args:
                message_content: Content of the broadcast
                message_type: Type of message
            """
            message = AgentMessage(
                id=f"MSG{datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]}",
                sender=ctx.deps.agent_id,
                recipient=None,  # Broadcast
                message_type=MessageType(message_type),
                content={"text": message_content},
                timestamp=datetime.now()
            )
            
            await ctx.deps.message_bus.publish(message)
            return f"📢 Broadcast sent: {message_content[:50]}..."
        
        @self.agent.tool
        async def read_shared_data(
            ctx: RunContext[CommunicatingAgentDeps],
            key: str
        ) -> str:
            """Read data from shared state.
            
            Args:
                key: Key to read from shared state
            """
            value = await ctx.deps.shared_state.get(key)
            if value is None:
                return f"No data found for key: {key}"
            return f"Shared data [{key}]: {value}"
        
        @self.agent.tool
        async def write_shared_data(
            ctx: RunContext[CommunicatingAgentDeps],
            key: str,
            value: str
        ) -> str:
            """Write data to shared state.
            
            Args:
                key: Key to write to shared state
                value: Value to write
            """
            await ctx.deps.shared_state.set(key, value)
            return f"✅ Wrote to shared state [{key}]: {value[:50]}..."
        
        @self.agent.tool
        async def get_recent_messages(
            ctx: RunContext[CommunicatingAgentDeps],
            limit: int = 10
        ) -> str:
            """Get recent messages from the message bus.
            
            Args:
                limit: Number of recent messages to retrieve
            """
            history = ctx.deps.message_bus.get_message_history(limit)
            
            if not history:
                return "No recent messages found."
            
            result = f"Recent {len(history)} messages:\n"
            for msg in history:
                timestamp = msg.timestamp.strftime("%H:%M:%S")
                recipient = msg.recipient or "ALL"
                result += f"[{timestamp}] {msg.sender} -> {recipient}: {msg.content.get('text', '')}...\n"
            
            return result
    
    async def start(self):
        """Start the agent's message handling"""
        if self.running:
            return
        
        self.message_queue = await self.message_bus.subscribe(self.agent_id)
        self.running = True
        
        # Start message handler
        self.message_handler_task = asyncio.create_task(
            self._handle_messages(),
            name=f"message_handler_{self.agent_id}"
        )
        
        print(f"🟢 Agent {self.agent_id} started")
    
    async def stop(self):
        """Stop the agent"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel message handler
        if self.message_handler_task:
            self.message_handler_task.cancel()
            try:
                await self.message_handler_task
            except asyncio.CancelledError:
                pass
        
        # Unsubscribe from message bus
        if self.message_queue:
            await self.message_bus.unsubscribe(self.agent_id, self.message_queue)
        
        print(f"🔴 Agent {self.agent_id} stopped")
    
    async def _handle_messages(self):
        """Handle incoming messages"""
        while self.running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                print(f"📨 {self.agent_id} received message from {message.sender}")
                
                # Process message based on type
                if message.message_type == MessageType.REQUEST:
                    await self._handle_request(message)
                elif message.message_type == MessageType.BROADCAST:
                    await self._handle_broadcast(message)
                elif message.message_type == MessageType.STATUS_UPDATE:
                    await self._handle_status_update(message)
                
            except asyncio.TimeoutError:
                continue  # No message received, continue loop
            except Exception as e:
                print(f"❌ {self.agent_id} message handling error: {e}")
    
    async def _handle_request(self, message: AgentMessage):
        """Handle a request message"""
        try:
            deps = CommunicatingAgentDeps(
                agent_id=self.agent_id,
                message_bus=self.message_bus,
                shared_state=self.shared_state,
                message_queue=self.message_queue
            )
            
            # Process the request
            request_content = message.content.get("text", "")
            context = f"Message from {message.sender}: {request_content}"
            
            result = await self.agent.run(context, deps=deps)
            
            # Send response back
            response = AgentMessage(
                id=f"MSG{datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]}",
                sender=self.agent_id,
                recipient=message.sender,
                message_type=MessageType.RESPONSE,
                content={"text": result.output, "request_id": message.id},
                timestamp=datetime.now(),
                correlation_id=message.id
            )
            
            await self.message_bus.publish(response)
            
        except Exception as e:
            print(f"❌ {self.agent_id} error handling request: {e}")
    
    async def _handle_broadcast(self, message: AgentMessage):
        """Handle a broadcast message"""
        # Just log the broadcast for now
        content = message.content.get("text", "")
        print(f"📢 {self.agent_id} received broadcast: {content[:50]}...")
    
    async def _handle_status_update(self, message: AgentMessage):
        """Handle a status update message"""
        content = message.content.get("text", "")
        print(f"📊 {self.agent_id} received status update: {content[:50]}...")
    
    async def send_direct_message(self, recipient: str, content: str) -> bool:
        """Send a direct message to another agent"""
        try:
            deps = CommunicatingAgentDeps(
                agent_id=self.agent_id,
                message_bus=self.message_bus,
                shared_state=self.shared_state
            )
            
            result = await self.agent.run(
                f"Send a message to {recipient}: {content}",
                deps=deps
            )
            
            return True
        except Exception as e:
            print(f"❌ Failed to send message: {e}")
            return False

class MultiAgentSystem:
    """Manages a system of communicating agents"""
    
    def __init__(self):
        self.message_bus = MessageBus()
        self.shared_state = SharedState()
        self.agents: Dict[str, CommunicatingAgent] = {}
    
    def add_agent(self, agent_id: str, agent_config: Dict[str, Any]):
        """Add an agent to the system"""
        agent = Agent(
            agent_config.get('model', 'openai:gpt-4o-mini'),
            deps_type=CommunicatingAgentDeps,
            system_prompt=agent_config.get('system_prompt', f'You are agent {agent_id}'),
            retries=2
        )
        
        communicating_agent = CommunicatingAgent(
            agent_id, agent, self.message_bus, self.shared_state
        )
        
        self.agents[agent_id] = communicating_agent
        print(f"➕ Added agent: {agent_id}")
    
    async def start_all_agents(self):
        """Start all agents"""
        start_tasks = [agent.start() for agent in self.agents.values()]
        await asyncio.gather(*start_tasks)
    
    async def stop_all_agents(self):
        """Stop all agents"""
        stop_tasks = [agent.stop() for agent in self.agents.values()]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
    
    async def demonstrate_communication(self):
        """Demonstrate agent communication patterns"""
        if len(self.agents) < 2:
            print("❌ Need at least 2 agents for communication demo")
            return
        
        agent_ids = list(self.agents.keys())
        
        # Initialize shared state
        await self.shared_state.set("system_status", "operational")
        await self.shared_state.set("active_tickets", 0)
        
        print("\n🎬 Demonstrating Agent Communication Patterns")
        print("="*60)
        
        # Pattern 1: Direct message
        print("\n1️⃣ Direct Message Pattern")
        sender = self.agents[agent_ids[0]]
        recipient_id = agent_ids[1]
        
        success = await sender.send_direct_message(
            recipient_id,
            "Can you help me with customer ticket #12345? The customer is asking about our return policy."
        )
        
        if success:
            print(f"✅ Direct message sent from {sender.agent_id} to {recipient_id}")
        
        await asyncio.sleep(2)  # Allow processing
        
        # Pattern 2: Broadcast
        print("\n2️⃣ Broadcast Pattern")
        broadcaster = self.agents[agent_ids[0]]
        deps = CommunicatingAgentDeps(
            agent_id=broadcaster.agent_id,
            message_bus=self.message_bus,
            shared_state=self.shared_state
        )
        
        await broadcaster.agent.run(
            "Broadcast to all agents: System maintenance will begin in 30 minutes. Please finish current tasks.",
            deps=deps
        )
        
        await asyncio.sleep(2)  # Allow processing
        
        # Pattern 3: Shared state coordination
        print("\n3️⃣ Shared State Coordination")
        
        # Multiple agents updating shared state
        tasks = []
        for i, agent_id in enumerate(agent_ids[:3]):  # Use first 3 agents
            agent = self.agents[agent_id]
            deps = CommunicatingAgentDeps(
                agent_id=agent.agent_id,
                message_bus=self.message_bus,
                shared_state=self.shared_state
            )
            
            task = asyncio.create_task(
                agent.agent.run(
                    f"Update shared data: Set 'agent_{i}_status' to 'working_on_ticket_{i+100}'",
                    deps=deps
                )
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Show final shared state
        print("\# Module 6: Async Multi-Agent Coordination
## Detailed Code Examples

### Learning Objectives
- Design async multi-agent architectures
- Implement concurrent agent execution
- Handle async communication patterns
- Manage shared state and coordination

---

## 6.1 Concurrent Agent Execution

```python
# concurrent_agents.py
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from datetime import datetime
import time

class AnalysisResult(BaseModel):
    """Result from a concurrent analysis agent"""
    agent_type: str
    analysis: str
    confidence: float
    processing_time: float
    recommendations: List[str]
    requires_action: bool

@dataclass
class ConcurrentDeps:
    request_id: str
    priority: str = "normal"
    timeout: int = 30
    start_time: float = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()

class ConcurrentAnalysisSystem:
    """System that runs multiple agents concurrently for comprehensive analysis"""
    
    def __init__(self):
        self.agents = self._initialize_agents()
        self.active_requests: Dict[str, Dict] = {}
    
    def _initialize_agents(self):
        """Initialize specialized analysis agents"""
        
        # Sentiment analysis agent
        sentiment_agent = Agent(
            'openai:gpt-4o-mini',
            output_type=AnalysisResult,
            deps_type=ConcurrentDeps,
            system_prompt="""
            You are a sentiment analysis specialist. Analyze the emotional tone,
            urgency level, and customer satisfaction indicators in customer messages.
            Provide confidence scores and actionable recommendations.
            """,
            retries=2
        )
        
        # Technical complexity agent
        technical_agent = Agent(
            'openai:gpt-4o',
            output_type=AnalysisResult,
            deps_type=ConcurrentDeps,
            system_prompt="""
            You are a technical complexity analyzer. Assess the technical difficulty,
            required expertise level, and potential solutions for technical issues.
            Estimate resolution time and identify required resources.
            """,
            retries=2
        )
        
        # Business impact agent
        business_agent = Agent(
            'openai:gpt-4o-mini',
            output_type=AnalysisResult,
            deps_type=ConcurrentDeps,
            system_prompt="""
            You are a business impact analyzer. Evaluate how customer issues affect
            business operations, revenue, and relationships. Assess priority levels
            and escalation needs based on business impact.
            """,
            retries=2
        )
        
        # Risk assessment agent
        risk_agent = Agent(
            'openai:gpt-4o',
            output_type=AnalysisResult,
            deps_type=ConcurrentDeps,
            system_prompt="""
            You are a risk assessment specialist. Identify potential risks, security
            concerns, legal implications, and compliance issues in customer situations.
            Provide risk mitigation recommendations.
            """,
            retries=2
        )
        
        return {
            "sentiment": sentiment_agent,
            "technical": technical_agent,
            "business": business_agent,
            "risk": risk_agent
        }
    
    async def analyze_customer_issue(
        self, 
        customer_message: str, 
        customer_info: Dict,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Run concurrent analysis of customer issue"""
        
        request_id = f"REQ{datetime.now().strftime('%Y%m%d%H%M%S')}"
        start_time = time.time()
        
        # Create dependencies
        deps = ConcurrentDeps(
            request_id=request_id,
            priority=priority,
            start_time=start_time
        )
        
        # Prepare analysis context
        analysis_context = f"""
        Customer Message: {customer_message}
        
        Customer Information:
        - Name: {customer_info.get('name', 'Unknown')}
        - Tier: {customer_info.get('tier', 'basic')}
        - Account Status: {customer_