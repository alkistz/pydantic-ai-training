# Module 7: Complex Multi-Agent Systems
## Detailed Code Examples

### Learning Objectives
- Design sophisticated agent hierarchies
- Implement agent orchestration patterns
- Handle complex state management
- Build resilient multi-agent systems

---

## 7.1 Agent Hierarchy and Orchestration

```python
# agent_hierarchy.py
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum
import json

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"

class AgentRole(str, Enum):
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    WORKER = "worker"
    SUPERVISOR = "supervisor"

class Task(BaseModel):
    """Represents a task in the hierarchy"""
    id: str
    title: str
    description: str
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = Field(default=1, ge=1, le=5)  # 5 = highest priority
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    parent_task_id: Optional[str] = None
    subtasks: List[str] = Field(default_factory=list)
    estimated_duration: int = 30  # minutes
    actual_duration: Optional[int] = None
    result: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentCapability(BaseModel):
    """Defines what an agent can do"""
    domain: str  # e.g., "technical", "billing", "customer_service"
    skill_level: int = Field(ge=1, le=5)  # 5 = expert
    max_concurrent_tasks: int = 3
    specializations: List[str] = Field(default_factory=list)

@dataclass
class HierarchyDeps:
    agent_id: str
    role: AgentRole
    capabilities: AgentCapability
    task_manager: 'TaskManager'
    agent_registry: 'AgentRegistry'
    current_tasks: List[str] = field(default_factory=list)

class TaskManager:
    """Manages task distribution and tracking"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_assignments: Dict[str, List[str]] = {}  # agent_id -> task_ids
        self.task_queue = asyncio.PriorityQueue()
        self.completion_callbacks: Dict[str, List[callable]] = {}
    
    async def create_task(
        self, 
        title: str, 
        description: str, 
        priority: int = 1,
        parent_task_id: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None
    ) -> str:
        """Create a new task"""
        task_id = f"TASK{datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]}"
        
        task = Task(
            id=task_id,
            title=title,
            description=description,
            priority=priority,
            parent_task_id=parent_task_id,
            metadata={"required_capabilities": required_capabilities or []}
        )
        
        self.tasks[task_id] = task
        
        # Add to queue (negative priority for max-heap behavior)
        await self.task_queue.put((-priority, task_id))
        
        # If this is a subtask, add to parent
        if parent_task_id and parent_task_id in self.tasks:
            self.tasks[parent_task_id].subtasks.append(task_id)
        
        return task_id
    
    async def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to an agent"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        task.assigned_agent = agent_id
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now()
        
        # Track assignment
        if agent_id not in self.task_assignments:
            self.task_assignments[agent_id] = []
        self.task_assignments[agent_id].append(task_id)
        
        return True
    
    async def complete_task(self, task_id: str, result: str, actual_duration: int = None):
        """Mark a task as completed"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.updated_at = datetime.now()
        
        if actual_duration:
            task.actual_duration = actual_duration
        
        # Remove from agent assignments
        if task.assigned_agent and task.assigned_agent in self.task_assignments:
            try:
                self.task_assignments[task.assigned_agent].remove(task_id)
            except ValueError:
                pass
        
        # Check if parent task can be completed
        if task.parent_task_id:
            await self._check_parent_completion(task.parent_task_id)
        
        # Execute completion callbacks
        if task_id in self.completion_callbacks:
            for callback in self.completion_callbacks[task_id]:
                try:
                    await callback(task)
                except Exception as e:
                    print(f"❌ Callback error for task {task_id}: {e}")
        
        return True
    
    async def _check_parent_completion(self, parent_task_id: str):
        """Check if all subtasks are completed"""
        parent_task = self.tasks.get(parent_task_id)
        if not parent_task:
            return
        
        # Check if all subtasks are completed
        all_completed = True
        for subtask_id in parent_task.subtasks:
            subtask = self.tasks.get(subtask_id)
            if not subtask or subtask.status != TaskStatus.COMPLETED:
                all_completed = False
                break
        
        if all_completed:
            # Aggregate results from subtasks
            results = []
            for subtask_id in parent_task.subtasks:
                subtask = self.tasks[subtask_id]
                if subtask.result:
                    results.append(f"{subtask.title}: {subtask.result}")
            
            await self.complete_task(
                parent_task_id,
                f"Completed with subtasks: {'; '.join(results)}"
            )
    
    async def get_next_task(self, agent_capabilities: AgentCapability) -> Optional[str]:
        """Get the next appropriate task for an agent"""
        try:
            # Get highest priority task
            priority, task_id = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
            
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            # Check if agent can handle this task
            required_caps = task.metadata.get("required_capabilities", [])
            if required_caps:
                agent_specs = set(agent_capabilities.specializations)
                if not any(cap in agent_specs for cap in required_caps):
                    # Put task back in queue
                    await self.task_queue.put((priority, task_id))
                    return None
            
            return task_id
            
        except asyncio.TimeoutError:
            return None  # No tasks available
    
    def get_agent_workload(self, agent_id: str) -> int:
        """Get current workload for an agent"""
        return len(self.task_assignments.get(agent_id, []))
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get task details"""
        return self.tasks.get(task_id)

class AgentRegistry:
    """Registry of available agents and their capabilities"""
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.online_agents: set = set()
    
    def register_agent(
        self, 
        agent_id: str, 
        role: AgentRole, 
        capabilities: AgentCapability,
        agent_instance: Any
    ):
        """Register an agent with the system"""
        self.agents[agent_id] = {
            "role": role,
            "capabilities": capabilities,
            "instance": agent_instance,
            "registered_at": datetime.now(),
            "last_seen": datetime.now()
        }
    
    def set_agent_online(self, agent_id: str):
        """Mark agent as online"""
        self.online_agents.add(agent_id)
        if agent_id in self.agents:
            self.agents[agent_id]["last_seen"] = datetime.now()
    
    def set_agent_offline(self, agent_id: str):
        """Mark agent as offline"""
        self.online_agents.discard(agent_id)
    
    def find_best_agent(
        self, 
        required_capabilities: List[str],
        exclude_agents: List[str] = None
    ) -> Optional[str]:
        """Find the best agent for a task"""
        exclude_agents = exclude_agents or []
        best_agent = None
        best_score = 0
        
        for agent_id, agent_info in self.agents.items():
            if agent_id not in self.online_agents or agent_id in exclude_agents:
                continue
            
            capabilities = agent_info["capabilities"]
            
            # Calculate capability match score
            score = 0
            for req_cap in required_capabilities:
                if req_cap in capabilities.specializations:
                    score += capabilities.skill_level
            
            # Consider workload (prefer less busy agents)
            workload_penalty = len(self.task_manager.get_agent_workload(agent_id)) * 0.5
            score -= workload_penalty
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    def get_online_agents_by_role(self, role: AgentRole) -> List[str]:
        """Get all online agents with a specific role"""
        return [
            agent_id for agent_id, agent_info in self.agents.items()
            if agent_id in self.online_agents and agent_info["role"] == role
        ]

class HierarchicalAgent:
    """Agent that operates within a hierarchy"""
    
    def __init__(
        self, 
        agent_id: str, 
        role: AgentRole,
        capabilities: AgentCapability,
        system_prompt: str,
        model: str = "openai:gpt-4o-mini"
    ):
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None
        
        # Create the underlying AI agent
        self.agent = Agent(
            model,
            deps_type=HierarchyDeps,
            system_prompt=system_prompt,
            retries=2
        )
        
        self._add_hierarchy_tools()
    
    def _add_hierarchy_tools(self):
        """Add hierarchy-specific tools to the agent"""
        
        @self.agent.tool
        async def delegate_subtask(
            ctx: RunContext[HierarchyDeps],
            subtask_title: str,
            subtask_description: str,
            required_skills: List[str],
            priority: int = 1
        ) -> str:
            """Delegate a subtask to another agent.
            
            Args:
                subtask_title: Title of the subtask
                subtask_description: Detailed description
                required_skills: Required skills/capabilities
                priority: Priority level (1-5)
            """
            task_manager = ctx.deps.task_manager
            registry = ctx.deps.agent_registry
            
            # Create the subtask
            subtask_id = await task_manager.create_task(
                subtask_title,
                subtask_description,
                priority,
                required_capabilities=required_skills
            )
            
            # Find appropriate agent
            best_agent = registry.find_best_agent(
                required_skills,
                exclude_agents=[ctx.deps.agent_id]  # Don't delegate to self
            )
            
            if best_agent:
                await task_manager.assign_task(subtask_id, best_agent)
                return f"✅ Delegated subtask '{subtask_title}' to {best_agent} (ID: {subtask_id})"
            else:
                return f"⚠️ No suitable agent found for subtask '{subtask_title}' (ID: {subtask_id})"
        
        @self.agent.tool
        async def request_assistance(
            ctx: RunContext[HierarchyDeps],
            assistance_type: str,
            description: str,
            urgent: bool = False
        ) -> str:
            """Request assistance from a supervisor or peer.
            
            Args:
                assistance_type: Type of assistance needed
                description: Description of what help is needed
                urgent: Whether this is urgent
            """
            registry = ctx.deps.agent_registry
            task_manager = ctx.deps.task_manager
            
            # Find supervisors or coordinators
            supervisors = registry.get_online_agents_by_role(AgentRole.SUPERVISOR)
            coordinators = registry.get_online_agents_by_role(AgentRole.COORDINATOR)
            
            potential_helpers = supervisors + coordinators
            
            if potential_helpers:
                helper = potential_helpers[0]  # Use first available
                
                # Create assistance task
                task_id = await task_manager.create_task(
                    f"Assistance Request: {assistance_type}",
                    f"Agent {ctx.deps.agent_id} requests assistance: {description}",
                    priority=4 if urgent else 2
                )
                
                await task_manager.assign_task(task_id, helper)
                
                return f"✅ Assistance requested from {helper} (Task: {task_id})"
            else:
                return "❌ No supervisors or coordinators available for assistance"
        
        @self.agent.tool
        async def get_team_status(ctx: RunContext[HierarchyDeps]) -> str:
            """Get status of the agent team.
            
            Args:
                None
            """
            registry = ctx.deps.agent_registry
            task_manager = ctx.deps.task_manager
            
            status_report = "📊 Team Status Report:\n\n"
            
            # Count agents by role
            role_counts = {}
            for agent_id in registry.online_agents:
                agent_info = registry.agents[agent_id]
                role = agent_info["role"]
                role_counts[role] = role_counts.get(role, 0) + 1
            
            status_report += "Online Agents by Role:\n"
            for role, count in role_counts.items():
                status_report += f"  {role}: {count}\n"
            
            # Task statistics
            total_tasks = len(task_manager.tasks)
            completed_tasks = len([t for t in task_manager.tasks.values() if t.status == TaskStatus.COMPLETED])
            in_progress_tasks = len([t for t in task_manager.tasks.values() if t.status == TaskStatus.IN_PROGRESS])
            
            status_report += f"\nTask Statistics:\n"
            status_report += f"  Total: {total_tasks}\n"
            status_report += f"  Completed: {completed_tasks}\n"
            status_report += f"  In Progress: {in_progress_tasks}\n"
            
            return status_report
        
        @self.agent.tool
        async def escalate_issue(
            ctx: RunContext[HierarchyDeps],
            issue_description: str,
            escalation_reason: str
        ) -> str:
            """Escalate an issue to higher authority.
            
            Args:
                issue_description: Description of the issue
                escalation_reason: Why this needs escalation
            """
            registry = ctx.deps.agent_registry
            task_manager = ctx.deps.task_manager
            
            # Find supervisors
            supervisors = registry.get_online_agents_by_role(AgentRole.SUPERVISOR)
            
            if supervisors:
                supervisor = supervisors[0]
                
                # Create escalation task
                task_id = await task_manager.create_task(
                    f"ESCALATION: {issue_description[:50]}...",
                    f"Escalated by {ctx.deps.agent_id}:\n\nIssue: {issue_description}\n\nReason: {escalation_reason}",
                    priority=5  # Highest priority
                )
                
                await task_manager.assign_task(task_id, supervisor)
                
                return f"🚨 Issue escalated to {supervisor} (Task: {task_id})"
            else:
                return "❌ No supervisors available for escalation"
    
    async def start(self, task_manager: TaskManager, agent_registry: AgentRegistry):
        """Start the agent worker"""
        if self.running:
            return
        
        self.running = True
        
        # Register with the system
        agent_registry.register_agent(self.agent_id, self.role, self.capabilities, self)
        agent_registry.set_agent_online(self.agent_id)
        
        # Start worker loop
        self.worker_task = asyncio.create_task(
            self._worker_loop(task_manager, agent_registry),
            name=f"worker_{self.agent_id}"
        )
        
        print(f"🟢 Agent {self.agent_id} ({self.role}) started")
    
    async def stop(self, agent_registry: AgentRegistry):
        """Stop the agent"""
        if not self.running:
            return
        
        self.running = False
        
        # Mark as offline
        agent_registry.set_agent_offline(self.agent_id)
        
        # Cancel worker
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        print(f"🔴 Agent {self.agent_id} stopped")
    
    async def _worker_loop(self, task_manager: TaskManager, agent_registry: AgentRegistry):
        """Main worker loop for processing tasks"""
        while self.running:
            try:
                # Check if we can take more tasks
                current_workload = task_manager.get_agent_workload(self.agent_id)
                if current_workload >= self.capabilities.max_concurrent_tasks:
                    await asyncio.sleep(1)
                    continue
                
                # Get next task
                task_id = await task_manager.get_next_task(self.capabilities)
                
                if task_id:
                    await self._process_task(task_id, task_manager, agent_registry)
                else:
                    await asyncio.sleep(2)  # No tasks available
                    
            except Exception as e:
                print(f"❌ Worker loop error for {self.agent_id}: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_task(
        self, 
        task_id: str, 
        task_manager: TaskManager, 
        agent_registry: AgentRegistry
    ):
        """Process a single task"""
        task = task_manager.get_task_status(task_id)
        if not task:
            return
        
        print(f"🔄 {self.agent_id} processing task: {task.title}")
        
        start_time = datetime.now()
        
        try:
            # Create dependencies
            deps = HierarchyDeps(
                agent_id=self.agent_id,
                role=self.role,
                capabilities=self.capabilities,
                task_manager=task_manager,
                agent_registry=agent_registry,
                current_tasks=[task_id]
            )
            
            # Process the task
            context = f"""
            Task: {task.title}
            Description: {task.description}
            Priority: {task.priority}
            Estimated Duration: {task.estimated_duration} minutes
            
            Please complete this task efficiently and provide a clear result.
            Use delegation tools if the task requires capabilities outside your specialization.
            """
            
            result = await self.agent.run(context, deps=deps)
            
            # Calculate duration
            duration = int((datetime.now() - start_time).total_seconds() / 60)
            
            # Complete the task
            await task_manager.complete_task(task_id, result.output, duration)
            
            print(f"✅ {self.agent_id} completed task: {task.title}")
            
        except Exception as e:
            print(f"❌ {self.agent_id} failed task {task.title}: {e}")
            # Mark task as failed
            task.status = TaskStatus.FAILED
            task.result = f"Failed: {str(e)}"
            task.updated_at = datetime.now()

class HierarchicalSystem:
    """Manages a hierarchical multi-agent system"""
    
    def __init__(self):
        self.task_manager = TaskManager()
        self.agent_registry = AgentRegistry()
        self.agents: Dict[str, HierarchicalAgent] = {}
    
    def create_agent(
        self, 
        agent_id: str, 
        role: AgentRole,
        domain: str,
        specializations: List[str],
        skill_level: int = 3,
        system_prompt: str = None
    ) -> HierarchicalAgent:
        """Create and add an agent to the hierarchy"""
        
        capabilities = AgentCapability(
            domain=domain,
            skill_level=skill_level,
            specializations=specializations,
            max_concurrent_tasks=3 if role == AgentRole.WORKER else 5
        )
        
        if system_prompt is None:
            system_prompt = f"""
            You are {agent_id}, a {role} agent specializing in {domain}.
            Your specializations: {', '.join(specializations)}
            Skill level: {skill_level}/5
            
            Work efficiently within the agent hierarchy. Use delegation and collaboration
            tools when appropriate. Always provide clear, actionable results.
            """
        
        agent = HierarchicalAgent(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities,
            system_prompt=system_prompt,
            model="openai:gpt-4o" if role in [AgentRole.COORDINATOR, AgentRole.SUPERVISOR] else "openai:gpt-4o-mini"
        )
        
        self.agents[agent_id] = agent
        return agent
    
    async def start_system(self):
        """Start all agents in the system"""
        start_tasks = []
        for agent in self.agents.values():
            start_tasks.append(agent.start(self.task_manager, self.agent_registry))
        
        await asyncio.gather(*start_tasks)
        print(f"✅ Hierarchical system started with {len(self.agents)} agents")
    
    async def stop_system(self):
        """Stop all agents"""
        stop_tasks = []
        for agent in self.agents.values():
            stop_tasks.append(agent.stop(self.agent_registry))
        
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        print("✅ Hierarchical system stopped")
    
    async def submit_work_request(
        self, 
        title: str, 
        description: str, 
        priority: int = 1,
        required_capabilities: List[str] = None
    ) -> str:
        """Submit a work request to the system"""
        
        task_id = await self.task_manager.create_task(
            title, description, priority, required_capabilities=required_capabilities
        )
        
        print(f"📝 Work request submitted: {title} (ID: {task_id})")
        return task_id
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Agent statistics
        agent_stats = {
            "total_agents": len(self.agents),
            "online_agents": len(self.agent_registry.online_agents),
            "agents_by_role": {}
        }
        
        for agent_id in self.agent_registry.online_agents:
            agent_info = self.agent_registry.agents[agent_id]
            role = agent_info["role"]
            agent_stats["agents_by_role"][role] = agent_stats["agents_by_role"].get(role, 0) + 1
        
        # Task statistics
        tasks = list(self.task_manager.tasks.values())
        task_stats = {
            "total_tasks": len(tasks),
            "by_status": {}
        }
        
        for task in tasks:
            status = task.status
            task_stats["by_status"][status] = task_stats["by_status"].get(status, 0) + 1
        
        # Performance metrics
        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED and t.actual_duration]
        
        performance_stats = {
            "average_completion_time": 0,
            "task_success_rate": 0
        }
        
        if completed_tasks:
            total_duration = sum(t.actual_duration for t in completed_tasks)
            performance_stats["average_completion_time"] = total_duration / len(completed_tasks)
            
            successful_tasks = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
            performance_stats["task_success_rate"] = successful_tasks / len(tasks) if tasks else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "agent_statistics": agent_stats,
            "task_statistics": task_stats,
            "performance_metrics": performance_stats
        }

async def demonstrate_hierarchical_system():
    """Demonstrate the hierarchical multi-agent system"""
    
    print("🎬 Hierarchical Multi-Agent System Demo")
    print("="*50)
    
    # Create the system
    system = HierarchicalSystem()
    
    # Create agent hierarchy
    
    # Supervisor level
    system.create_agent(
        "supervisor_001",
        AgentRole.SUPERVISOR,
        "management",
        ["oversight", "decision_making", "escalation_handling"],
        skill_level=5
    )
    
    # Coordinator level
    system.create_agent(
        "coordinator_001",
        AgentRole.COORDINATOR,
        "customer_service",
        ["task_routing", "resource_allocation", "coordination"],
        skill_level=4
    )
    
    # Specialist level
    system.create_agent(
        "tech_specialist_001",
        AgentRole.SPECIALIST,
        "technical",
        ["hardware_troubleshooting", "software_debugging", "network_issues"],
        skill_level=4
    )
    
    system.create_agent(
        "billing_specialist_001",
        AgentRole.SPECIALIST,
        "financial",
        ["billing_disputes", "payment_processing", "account_management"],
        skill_level=4
    )
    
    # Worker level
    system.create_agent(
        "support_worker_001",
        AgentRole.WORKER,
        "general_support",
        ["basic_inquiries", "documentation", "data_entry"],
        skill_level=3
    )
    
    system.create_agent(
        "support_worker_002",
        AgentRole.WORKER,
        "general_support",
        ["basic_inquiries", "customer_communication", "ticket_management"],
        skill_level=3
    )
    
    try:
        # Start the system
        await system.start_system()
        
        # Submit various work requests
        print("\n📋 Submitting work requests...")
        
        # Complex technical issue requiring coordination
        task1 = await system.submit_work_request(
            "Server Outage Investigation",
            "Critical: Customer reporting complete server outage affecting multiple services. Need immediate investigation and resolution with regular status updates.",
            priority=5,
            required_capabilities=["hardware_troubleshooting", "network_issues"]
        )
        
        # Billing dispute
        task2 = await system.submit_work_request(
            "Enterprise Billing Dispute",
            "Enterprise customer disputes $50,000 charge on latest invoice. Requires thorough review and resolution with management approval if needed.",
            priority=4,
            required_capabilities=["billing_disputes", "account_management"]
        )
        
        # General inquiry
        task3 = await system.submit_work_request(
            "Product Information Request",
            "Customer requesting detailed specifications and pricing for latest laptop models.",
            priority=2,
            required_capabilities=["basic_inquiries"]
        )
        
        # Documentation task
        task4 = await system.submit_work_request(
            "Update Support Documentation",
            "Update troubleshooting guides based on recent customer issues and feedback.",
            priority=1,
            required_capabilities=["documentation"]
        )
        
        # Let the system process tasks
        print("\n⏳ Processing tasks...")
        await asyncio.sleep(10)  # Allow time for processing
        
        # Check system status
        print("\n📊 System Status:")
        status = await system.get_system_status()
        print(json.dumps(status, indent=2, default=str))
        
        # Show task progress
        print("\n📝 Task Progress:")
        for task_id in [task1, task2, task3, task4]:
            task = system.task_manager.get_task_status(task_id)
            if task:
                print(f"  {task.title}: {task.status}")
                if task.assigned_agent:
                    print(f"    Assigned to: {task.assigned_agent}")
                if task.result:
                    print(f"    Result: {task.result[:100]}...")
        
        # Wait for more processing
        await asyncio.sleep(15)
        
        # Final status check
        print("\n📈 Final System Status:")
        final_status = await system.get_system_status()
        print(f"Total tasks: {final_status['task_statistics']['total_tasks']}")
        print(f"Completed: {final_status['task_statistics']['by_status'].get('completed', 0)}")
        print(f"In progress: {final_status['task_statistics']['by_status'].get('in_progress', 0)}")
        print(f"Success rate: {final_status['performance_metrics']['task_success_rate']:.1%}")
        
    finally:
        await system.stop_system()

if __name__ == "__main__":
    asyncio.run(demonstrate_hierarchical_system())
```

---

## Practice Exercises for Module 7

### Exercise 1: Dynamic Agent Creation
Build a system that can:
- Create new agents on demand based on workload
- Scale agent pools up and down automatically
- Handle agent failures and replacements
- Optimize agent distribution across tasks

### Exercise 2: Advanced Task Dependencies
Implement a task system with:
- Complex dependency graphs (A depends on B and C)
- Conditional task execution
- Task rollback and retry mechanisms
- Resource conflict resolution

### Exercise 3: Performance Optimization
Create optimizations for:
- Task scheduling algorithms
- Agent load balancing
- Memory and CPU usage optimization
- Network communication efficiency

### Exercise 4: Monitoring and Analytics
Build comprehensive monitoring that tracks:
- Individual agent performance metrics
- System-wide efficiency indicators
- Task completion patterns and bottlenecks
- Cost and resource utilization

### Next Steps
Once you've mastered complex multi-agent systems, you'll be ready for **Module 8: Graph-Based Multi-Agent Systems**, where we'll explore using pydantic-graph for sophisticated workflow management.

**Key Takeaways:**
- Hierarchical organization enables scalable agent management
- Task delegation allows efficient work distribution
- Agent capabilities matching ensures optimal task assignment
- Supervision and escalation provide quality control
- Performance monitoring enables continuous optimization
- Proper lifecycle management ensures system reliability