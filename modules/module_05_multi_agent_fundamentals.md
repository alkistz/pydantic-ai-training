# Module 5: Multi-Agent Fundamentals
## Detailed Code Examples

### Learning Objectives
- Understand different multi-agent patterns
- Implement agent delegation
- Master programmatic agent hand-off
- Handle usage limits across agents

---

## 5.1 Agent Delegation - Specialized Support System

```python
# agent_delegation.py
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from datetime import datetime

class TicketAnalysis(BaseModel):
    """Analysis result from the triage agent"""
    category: str
    complexity: str  # "simple", "moderate", "complex"
    estimated_time: int  # minutes
    recommended_agent: str
    priority: str
    requires_escalation: bool

class TechnicalSolution(BaseModel):
    """Technical solution from specialist agent"""
    solution_steps: List[str]
    estimated_time: int
    success_probability: float
    requires_followup: bool
    additional_resources: List[str]

@dataclass
class AgentDeps:
    ticket_id: str
    customer_tier: str = "basic"
    agent_workload: Dict[str, int] = None
    
    def __post_init__(self):
        if self.agent_workload is None:
            self.agent_workload = {
                "technical": 3,
                "billing": 1,
                "general": 2
            }

# Master coordinator agent
triage_agent = Agent(
    'openai:gpt-4o',
    output_type=TicketAnalysis,
    deps_type=AgentDeps,
    system_prompt="""
    You are a triage specialist who analyzes customer issues and routes them 
    to the appropriate specialist agent.
    
    Categories:
    - technical: Hardware/software issues, troubleshooting
    - billing: Payment, refunds, account issues  
    - general: Product info, policies, basic questions
    
    Complexity levels:
    - simple: Common issues with standard solutions (5-10 min)
    - moderate: Requires investigation or multiple steps (15-30 min)
    - complex: Advanced troubleshooting or escalation needed (30+ min)
    
    Consider customer tier for priority assignment.
    """,
    retries=2
)

# Specialized agents
technical_agent = Agent(
    'openai:gpt-4o',
    output_type=TechnicalSolution,
    deps_type=AgentDeps,
    system_prompt="""
    You are a technical support specialist with expertise in:
    - Hardware troubleshooting
    - Software configuration
    - Network connectivity
    - Performance optimization
    
    Provide detailed, step-by-step solutions.
    Include estimated time and success probability.
    """,
    retries=2
)

billing_agent = Agent(
    'openai:gpt-4o-mini',  # Faster model for billing queries
    deps_type=AgentDeps,
    system_prompt="""
    You are a billing specialist who handles:
    - Payment issues
    - Refund requests
    - Account billing questions
    - Subscription management
    
    Always verify customer identity before discussing account details.
    Be empathetic with billing disputes.
    """,
    retries=2
)

general_agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=AgentDeps,
    system_prompt="""
    You are a general customer service representative handling:
    - Product information
    - Company policies
    - Basic how-to questions
    - Order status inquiries
    
    Be friendly and helpful. Direct complex issues to specialists.
    """,
    retries=2
)

class MultiAgentSupportSystem:
    """Orchestrates multiple specialized agents"""
    
    def __init__(self):
        self.agents = {
            "technical": technical_agent,
            "billing": billing_agent,
            "general": general_agent
        }
        self.triage = triage_agent
        self.session_history: List[Dict] = []
    
    async def handle_customer_issue(self, issue: str, customer_info: Dict) -> Dict:
        """Handle customer issue using appropriate specialist"""
        
        # Create dependencies
        deps = AgentDeps(
            ticket_id=f"TICK{datetime.now().strftime('%Y%m%d%H%M%S')}",
            customer_tier=customer_info.get("tier", "basic")
        )
        
        # Step 1: Triage the issue
        print("🔍 Analyzing issue...")
        triage_result = await self.triage.run(
            f"Customer issue: {issue}\nCustomer tier: {customer_info.get('tier', 'basic')}",
            deps=deps
        )
        
        analysis = triage_result.output
        print(f"📋 Triage complete: {analysis.category} ({analysis.complexity})")
        
        # Step 2: Route to appropriate specialist
        if analysis.requires_escalation:
            return {
                "status": "escalated",
                "message": "This issue requires human intervention. A specialist will contact you within 30 minutes.",
                "ticket_id": deps.ticket_id,
                "analysis": analysis
            }
        
        specialist_agent = self.agents[analysis.category]
        
        # Step 3: Get specialist response
        print(f"🤖 Routing to {analysis.category} specialist...")
        
        if analysis.category == "technical":
            specialist_result = await specialist_agent.run(
                f"Technical issue: {issue}\nCustomer tier: {customer_info.get('tier')}",
                deps=deps,
                usage=triage_result.usage  # Pass usage tracking
            )
            
            solution = specialist_result.output
            response = self._format_technical_response(solution, analysis)
            
        else:
            # For billing and general agents
            specialist_result = await specialist_agent.run(
                f"Customer inquiry: {issue}\nCategory: {analysis.category}",
                deps=deps,
                usage=triage_result.usage
            )
            
            response = {
                "status": "resolved",
                "category": analysis.category,
                "response": specialist_result.output,
                "estimated_time": analysis.estimated_time
            }
        
        # Record session
        self.session_history.append({
            "timestamp": datetime.now().isoformat(),
            "ticket_id": deps.ticket_id,
            "analysis": analysis.dict(),
            "response": response,
            "total_usage": specialist_result.usage().dict()
        })
        
        return response
    
    def _format_technical_response(self, solution: TechnicalSolution, analysis: TicketAnalysis) -> Dict:
        """Format technical solution response"""
        return {
            "status": "resolved" if solution.success_probability > 0.8 else "partial",
            "category": "technical",
            "solution_steps": solution.solution_steps,
            "estimated_time": solution.estimated_time,
            "success_probability": solution.success_probability,
            "requires_followup": solution.requires_followup,
            "additional_resources": solution.additional_resources,
            "complexity": analysis.complexity
        }

# Enhanced coordinator with tool access to specialists
@triage_agent.tool
async def consult_technical_specialist(ctx: RunContext[AgentDeps], technical_question: str) -> str:
    """Consult technical specialist for complex technical questions.
    
    Args:
        technical_question: Specific technical question needing expert input
    """
    result = await technical_agent.run(
        f"Technical consultation: {technical_question}",
        deps=ctx.deps,
        usage=ctx.usage
    )
    
    if isinstance(result.output, TechnicalSolution):
        solution = result.output
        return f"""
        Technical specialist consultation:
        Success probability: {solution.success_probability}
        Estimated time: {solution.estimated_time} minutes
        Key steps: {', '.join(solution.solution_steps[:2])}...
        """
    else:
        return f"Technical specialist response: {result.output}"

@triage_agent.tool
async def check_agent_availability(ctx: RunContext[AgentDeps]) -> str:
    """Check current workload of specialist agents.
    
    Args:
        None
    """
    workload = ctx.deps.agent_workload
    
    availability = []
    for agent_type, current_load in workload.items():
        if current_load < 3:
            status = "Available"
        elif current_load < 6:
            status = "Busy"
        else:
            status = "Overloaded"
        
        availability.append(f"{agent_type}: {status} ({current_load} active tickets)")
    
    return "Current agent availability:\n" + "\n".join(availability)

async def demonstrate_agent_delegation():
    support_system = MultiAgentSupportSystem()
    
    # Test scenarios
    test_cases = [
        {
            "issue": "My laptop won't boot up. The screen stays black and I hear a beeping sound.",
            "customer": {"name": "John Doe", "tier": "premium", "id": "CUST001"}
        },
        {
            "issue": "I was charged twice for my subscription this month. Can you help me get a refund?",
            "customer": {"name": "Jane Smith", "tier": "basic", "id": "CUST002"}
        },
        {
            "issue": "What are the specifications of your latest laptop model?",
            "customer": {"name": "Bob Wilson", "tier": "enterprise", "id": "CUST003"}
        },
        {
            "issue": "My server is experiencing severe performance issues and keeps crashing. This is affecting our entire business operations.",
            "customer": {"name": "Alice Johnson", "tier": "enterprise", "id": "CUST004"}
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {test_case['customer']['name']} ({test_case['customer']['tier']})")
        print(f"Issue: {test_case['issue']}")
        print('='*80)
        
        try:
            result = await support_system.handle_customer_issue(
                test_case['issue'],
                test_case['customer']
            )
            
            print(f"✅ Result: {result['status']}")
            if 'response' in result:
                print(f"Response: {result['response']}")
            elif 'solution_steps' in result:
                print("Solution steps:")
                for step in result['solution_steps']:
                    print(f"  • {step}")
                print(f"Success probability: {result['success_probability']:.1%}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Print session summary
    print(f"\n{'='*80}")
    print("SESSION SUMMARY")
    print('='*80)
    print(f"Total tickets processed: {len(support_system.session_history)}")
    
    for session in support_system.session_history:
        print(f"\nTicket {session['ticket_id']}:")
        print(f"  Category: {session['analysis']['category']}")
        print(f"  Complexity: {session['analysis']['complexity']}")
        print(f"  Status: {session['response']['status']}")

if __name__ == "__main__":
    asyncio.run(demonstrate_agent_delegation())
```

**Key Concepts:**
- **Agent Delegation**: One agent calling another for specialized tasks
- **Triage Pattern**: Central coordinator routing to specialists
- **Usage Tracking**: Passing usage limits between agents
- **Structured Communication**: Using Pydantic models for agent-to-agent data

---

## 5.2 Programmatic Agent Hand-off

```python
# agent_handoff.py
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class HandoffReason(str, Enum):
    COMPLEXITY = "complexity"
    SPECIALIZATION = "specialization"
    ESCALATION = "escalation"
    WORKLOAD = "workload"
    CUSTOMER_REQUEST = "customer_request"

class AgentRole(str, Enum):
    INTAKE = "intake"
    TECHNICAL = "technical"
    BILLING = "billing"
    MANAGER = "manager"
    SPECIALIST = "specialist"

class HandoffContext(BaseModel):
    """Context passed between agents during handoff"""
    previous_agent: str
    handoff_reason: HandoffReason
    conversation_summary: str
    customer_info: Dict[str, Any]
    attempted_solutions: List[str]
    escalation_level: int
    priority: str
    notes: List[str]

@dataclass
class WorkflowDeps:
    conversation_history: List[Dict[str, str]]
    customer_tier: str
    current_agent: str
    escalation_level: int = 0
    
    def add_message(self, role: str, message: str):
        self.conversation_history.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "agent": self.current_agent
        })

class AgentWorkflow:
    """Manages the workflow and handoffs between agents"""
    
    def __init__(self):
        self.agents = self._initialize_agents()
        self.active_workflows: Dict[str, WorkflowDeps] = {}
    
    def _initialize_agents(self):
        """Initialize all agents in the workflow"""
        
        # Intake agent - first point of contact
        intake_agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=WorkflowDeps,
            system_prompt="""
            You are the first-line customer service agent. Your role is to:
            1. Greet customers warmly and gather basic information
            2. Understand the customer's issue clearly
            3. Attempt simple solutions for common problems
            4. Determine if the issue needs to be handed off to a specialist
            
            Use the handoff tools when:
            - Technical issues beyond basic troubleshooting
            - Billing disputes or complex account issues
            - Customer requests escalation
            - Issue requires specialized knowledge
            """,
            retries=2
        )
        
        # Technical specialist
        technical_agent = Agent(
            'openai:gpt-4o',
            deps_type=WorkflowDeps,
            system_prompt="""
            You are a technical specialist. You receive cases from intake agents.
            Focus on:
            1. Reviewing the conversation history and attempted solutions
            2. Providing advanced technical solutions
            3. Escalating to management if critical systems are affected
            4. Following up on complex issues
            
            Always acknowledge the previous agent's work and build upon it.
            """,
            retries=2
        )
        
        # Billing specialist
        billing_agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=WorkflowDeps,
            system_prompt="""
            You are a billing specialist handling financial matters.
            You focus on:
            1. Payment issues and disputes
            2. Refund processing
            3. Account adjustments
            4. Subscription management
            
            Be empathetic with billing concerns and explain policies clearly.
            Escalate to management for large refunds or policy exceptions.
            """,
            retries=2
        )
        
        # Manager for escalations
        manager_agent = Agent(
            'openai:gpt-4o',
            deps_type=WorkflowDeps,
            system_prompt="""
            You are a customer service manager handling escalated cases.
            Your authority includes:
            1. Policy exceptions and overrides
            2. Significant refunds and compensations
            3. Complex dispute resolution
            4. Final decision making
            
            Review the entire conversation history and previous agents' work.
            Provide authoritative solutions and ensure customer satisfaction.
            """,
            retries=2
        )
        
        # Add tools to agents
        self._add_handoff_tools(intake_agent)
        self._add_handoff_tools(technical_agent)
        self._add_handoff_tools(billing_agent)
        self._add_handoff_tools(manager_agent)
        
        return {
            AgentRole.INTAKE: intake_agent,
            AgentRole.TECHNICAL: technical_agent,
            AgentRole.BILLING: billing_agent,
            AgentRole.MANAGER: manager_agent
        }
    
    def _add_handoff_tools(self, agent: Agent):
        """Add handoff tools to an agent"""
        
        @agent.tool
        async def handoff_to_specialist(
            ctx: RunContext[WorkflowDeps],
            target_agent: str,
            reason: str,
            summary: str,
            notes: Optional[str] = None
        ) -> str:
            """Hand off the conversation to a specialist agent.
            
            Args:
                target_agent: Target agent type (technical, billing, manager)
                reason: Reason for handoff (complexity, specialization, escalation, etc.)
                summary: Summary of the conversation and issue
                notes: Additional notes for the receiving agent
            """
            
            # Create handoff context
            handoff_context = HandoffContext(
                previous_agent=ctx.deps.current_agent,
                handoff_reason=HandoffReason(reason),
                conversation_summary=summary,
                customer_info={"tier": ctx.deps.customer_tier},
                attempted_solutions=[],  # Would extract from conversation
                escalation_level=ctx.deps.escalation_level,
                priority="medium",  # Would determine from context
                notes=[notes] if notes else []
            )
            
            # Update workflow state
            ctx.deps.current_agent = target_agent
            if reason == "escalation":
                ctx.deps.escalation_level += 1
            
            # Log handoff
            ctx.deps.add_message(
                "system",
                f"Handed off to {target_agent} agent. Reason: {reason}"
            )
            
            return f"""
            ✅ Successfully handed off to {target_agent} specialist.
            
            Handoff Summary:
            - Reason: {reason}
            - Previous Agent: {handoff_context.previous_agent}
            - Escalation Level: {ctx.deps.escalation_level}
            - Summary: {summary}
            
            The {target_agent} specialist will continue assisting you shortly.
            """
        
        @agent.tool
        async def review_conversation_history(ctx: RunContext[WorkflowDeps]) -> str:
            """Review the full conversation history for context.
            
            Args:
                None
            """
            history = ctx.deps.conversation_history
            
            if not history:
                return "No conversation history available."
            
            summary = f"Conversation History ({len(history)} messages):\n\n"
            
            for msg in history[-10:]:  # Last 10 messages
                timestamp = msg.get('timestamp', 'Unknown time')
                agent = msg.get('agent', 'Unknown agent')
                summary += f"[{timestamp}] {agent}: {msg['message'][:100]}...\n"
            
            return summary
        
        @agent.tool
        async def escalate_to_manager(
            ctx: RunContext[WorkflowDeps],
            escalation_reason: str,
            customer_impact: str
        ) -> str:
            """Escalate the issue to a manager.
            
            Args:
                escalation_reason: Why this needs manager attention
                customer_impact: How this impacts the customer
            """
            ctx.deps.escalation_level += 1
            ctx.deps.current_agent = "manager"
            
            escalation_summary = f"""
            ESCALATION TO MANAGER - Level {ctx.deps.escalation_level}
            
            Reason: {escalation_reason}
            Customer Impact: {customer_impact}
            Customer Tier: {ctx.deps.customer_tier}
            Previous Agent: {ctx.deps.current_agent}
            
            A manager will take over this case immediately.
            """
            
            ctx.deps.add_message("system", escalation_summary)
            
            return escalation_summary
    
    async def start_conversation(self, customer_message: str, customer_info: Dict) -> str:
        """Start a new customer conversation"""
        
        # Create workflow context
        workflow_id = f"WF{datetime.now().strftime('%Y%m%d%H%M%S')}"
        deps = WorkflowDeps(
            conversation_history=[],
            customer_tier=customer_info.get("tier", "basic"),
            current_agent="intake"
        )
        
        self.active_workflows[workflow_id] = deps
        
        # Start with intake agent
        deps.add_message("customer", customer_message)
        
        result = await self.agents[AgentRole.INTAKE].run(
            f"New customer inquiry: {customer_message}",
            deps=deps
        )
        
        deps.add_message("agent", result.output)
        
        return result.output
    
    async def continue_conversation(
        self, 
        workflow_id: str, 
        customer_message: str
    ) -> str:
        """Continue an existing conversation"""
        
        if workflow_id not in self.active_workflows:
            return "Workflow not found. Please start a new conversation."
        
        deps = self.active_workflows[workflow_id]
        deps.add_message("customer", customer_message)
        
        # Get current agent
        current_agent_role = AgentRole(deps.current_agent)
        current_agent = self.agents[current_agent_role]
        
        # Continue conversation with current agent
        context = f"""
        Customer message: {customer_message}
        
        Previous conversation context:
        {self._get_conversation_context(deps)}
        """
        
        result = await current_agent.run(context, deps=deps)
        deps.add_message("agent", result.output)
        
        return result.output
    
    def _get_conversation_context(self, deps: WorkflowDeps) -> str:
        """Get relevant conversation context for the agent"""
        recent_messages = deps.conversation_history[-5:]  # Last 5 messages
        
        context = ""
        for msg in recent_messages:
            role = msg['role']
            message = msg['message'][:200]  # Truncate long messages
            context += f"{role}: {message}\n"
        
        return context

async def demonstrate_agent_handoff():
    workflow = AgentWorkflow()
    
    # Simulate a complex customer journey with handoffs
    print("🎬 Starting Customer Service Workflow Demonstration")
    print("="*60)
    
    # Customer starts with a technical issue
    customer_info = {"name": "Sarah Chen", "tier": "premium", "id": "CUST005"}
    
    print("\n📞 Customer contacts support:")
    initial_message = "Hi, I'm having a serious problem. My business laptop crashed during an important presentation and now it won't start at all. I have critical files on there and a deadline tomorrow. This is really urgent!"
    
    print(f"Customer: {initial_message}")
    
    # Start workflow
    response1 = await workflow.start_conversation(initial_message, customer_info)
    print(f"\n🤖 Intake Agent: {response1}")
    
    # Customer provides more technical details
    print("\n📞 Customer provides more details:")
    followup1 = "The laptop was working fine this morning, but during my presentation it suddenly blue-screened with some error about memory. Now when I press the power button, the lights come on for a second but then it goes completely black. I really need those files recovered."
    
    print(f"Customer: {followup1}")
    
    # Get workflow ID (in real implementation, this would be tracked properly)
    workflow_id = list(workflow.active_workflows.keys())[0]
    
    response2 = await workflow.continue_conversation(workflow_id, followup1)
    print(f"\n🤖 Current Agent: {response2}")
    
    # Customer requests escalation
    print("\n📞 Customer requests escalation:")
    followup2 = "Look, I appreciate your help but this is a critical business issue. I need to speak to someone with more authority who can help me get emergency data recovery. Can you escalate this to a manager?"
    
    print(f"Customer: {followup2}")
    
    response3 = await workflow.continue_conversation(workflow_id, followup2)
    print(f"\n🤖 Current Agent: {response3}")
    
    # Show final workflow state
    deps = workflow.active_workflows[workflow_id]
    print(f"\n📊 Final Workflow State:")
    print(f"Current Agent: {deps.current_agent}")
    print(f"Escalation Level: {deps.escalation_level}")
    print(f"Total Messages: {len(deps.conversation_history)}")
    
    print(f"\n📝 Conversation Summary:")
    for msg in deps.conversation_history:
        role = msg['role'].title()
        message = msg['message'][:100] + "..." if len(msg['message']) > 100 else msg['message']
        print(f"  {role}: {message}")

if __name__ == "__main__":
    asyncio.run(demonstrate_agent_handoff())
```

**Key Concepts:**
- **Programmatic Hand-off**: Systematic transfer of control between agents
- **Context Preservation**: Maintaining full conversation context across handoffs
- **Escalation Paths**: Clear escalation hierarchy and procedures
- **Workflow Management**: Tracking and managing multi-agent conversations

---

## Practice Exercises for Module 5

### Exercise 1: Specialist Agent Network
Create a network of specialist agents for different domains:
- Create agents for specific product categories
- Implement load balancing between similar specialists
- Add fallback mechanisms when specialists are unavailable

### Exercise 2: Advanced Triage System
Build a sophisticated triage system that:
- Uses machine learning for better routing decisions
- Considers agent expertise and availability
- Handles multiple simultaneous issues from one customer

### Exercise 3: Hand-off Quality Control
Implement quality control for hand-offs:
- Validate that handoffs include sufficient context
- Monitor handoff success rates
- Implement feedback loops for improvement

### Exercise 4: Multi-tier Escalation
Create a multi-tier escalation system:
- Level 1: Basic support agents
- Level 2: Senior specialists
- Level 3: Team leads
- Level 4: Management

### Next Steps
Once you've mastered multi-agent fundamentals, you'll be ready for **Module 6: Async Multi-Agent Coordination**, where we'll learn to run multiple agents concurrently and handle complex coordination patterns.

**Key Takeaways:**
- Agent delegation allows specialized expertise within a unified system
- Programmatic handoffs maintain context while transferring control
- Clear escalation paths ensure complex issues get proper attention
- Usage tracking across agents helps manage costs and performance
- Workflow management is crucial for multi-step customer journeys