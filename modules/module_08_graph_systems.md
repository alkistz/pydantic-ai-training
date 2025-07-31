# Module 8: Graph-Based Multi-Agent Systems
## Detailed Code Examples

### Learning Objectives
- Understand pydantic-graph integration
- Design finite state machines for agent control
- Implement complex workflow orchestration
- Master advanced graph patterns

---

## 8.1 Customer Service Workflow Graph

```python
# customer_service_graph.py
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from datetime import datetime
from enum import Enum
import json

# Note: This example assumes pydantic-graph is available
# Install with: pip install pydantic-graph

try:
    from pydantic_graph import BaseNode, End, Graph, GraphRunContext
    GRAPH_AVAILABLE = True
except ImportError:
    # Fallback implementation for demonstration
    GRAPH_AVAILABLE = False
    print("⚠️ pydantic-graph not available, using mock implementation")
    
    class BaseNode:
        pass
    
    class End:
        def __init__(self, data):
            self.data = data
    
    class Graph:
        def __init__(self, nodes):
            pass
        
        async def run(self, start_node, deps=None):
            return type('Result', (), {'output': 'Mock result'})()
    
    class GraphRunContext:
        def __init__(self, deps):
            self.deps = deps

class CustomerIssueType(str, Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"
    ESCALATION = "escalation"

class IssueComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class CustomerIssue(BaseModel):
    """Represents a customer issue flowing through the graph"""
    id: str
    description: str
    customer_tier: str
    issue_type: CustomerIssueType
    complexity: IssueComplexity
    priority: int
    attempted_solutions: List[str] = []
    resolution: Optional[str] = None
    escalation_level: int = 0
    processing_history: List[str] = []

@dataclass
class GraphDeps:
    """Dependencies for the graph execution"""
    customer_db: Dict[str, Any]
    knowledge_base: Dict[str, str]
    escalation_threshold: int = 2
    max_attempts: int = 3

class CustomerServiceGraph:
    """Graph-based customer service workflow"""
    
    def __init__(self):
        self.agents = self._create_agents()
        self.graph = self._build_graph()
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialized agents for different graph nodes"""
        
        # Triage agent - analyzes and routes issues
        triage_agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=GraphDeps,
            system_prompt="""
            You are a triage agent that analyzes customer issues and determines:
            1. Issue type (technical, billing, general, escalation)
            2. Complexity level (simple, moderate, complex)
            3. Priority (1-5, where 5 is urgent)
            
            Be accurate in your assessment as it determines the routing.
            """,
            retries=2
        )
        
        # Technical resolution agent
        technical_agent = Agent(
            'openai:gpt-4o',
            deps_type=GraphDeps,
            system_prompt="""
            You are a technical support specialist. Provide clear, step-by-step
            solutions for technical problems. If you cannot resolve the issue,
            recommend escalation with detailed reasoning.
            """,
            retries=2
        )
        
        # Billing resolution agent
        billing_agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=GraphDeps,
            system_prompt="""
            You are a billing specialist. Handle payment issues, refunds,
            and account problems. Follow company policies strictly and
            escalate when policy exceptions are needed.
            """,
            retries=2
        )
        
        # General support agent
        general_agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=GraphDeps,
            system_prompt="""
            You are a general customer service representative. Handle
            information requests, policy questions, and general inquiries.
            Be helpful and direct customers to appropriate resources.
            """,
            retries=2
        )
        
        # Escalation agent
        escalation_agent = Agent(
            'openai:gpt-4o',
            deps_type=GraphDeps,
            system_prompt="""
            You are an escalation specialist with authority to make exceptions
            and handle complex cases. Provide final resolutions and ensure
            customer satisfaction while protecting company interests.
            """,
            retries=2
        )
        
        return {
            "triage": triage_agent,
            "technical": technical_agent,
            "billing": billing_agent,
            "general": general_agent,
            "escalation": escalation_agent
        }
    
    def _build_graph(self):
        """Build the customer service workflow graph"""
        
        if not GRAPH_AVAILABLE:
            return Graph([])  # Mock graph
        
        # Define graph nodes
        nodes = [
            TriageNode,
            TechnicalResolutionNode,
            BillingResolutionNode,
            GeneralSupportNode,
            EscalationNode,
            QualityCheckNode,
            ResolutionNode
        ]
        
        return Graph(nodes=nodes)
    
    async def process_customer_issue(
        self, 
        issue_description: str,
        customer_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a customer issue through the graph"""
        
        # Create initial issue
        issue = CustomerIssue(
            id=f"ISSUE{datetime.now().strftime('%Y%m%d%H%M%S')}",
            description=issue_description,
            customer_tier=customer_info.get("tier", "basic"),
            issue_type=CustomerIssueType.GENERAL,  # Will be determined by triage
            complexity=IssueComplexity.SIMPLE,    # Will be determined by triage
            priority=1                            # Will be determined by triage
        )
        
        # Create dependencies
        deps = GraphDeps(
            customer_db={"customer": customer_info},
            knowledge_base={
                "return_policy": "30-day return policy",
                "warranty": "1-year warranty on all products",
                "shipping": "Free shipping on orders over $50"
            }
        )
        
        # Start graph execution
        print(f"🎯 Processing issue: {issue.id}")
        
        try:
            result = await self.graph.run(
                TriageNode(issue=issue, agents=self.agents),
                deps=deps
            )
            
            final_issue = result.output
            
            return {
                "issue_id": final_issue.id,
                "resolution": final_issue.resolution,
                "processing_history": final_issue.processing_history,
                "escalation_level": final_issue.escalation_level,
                "final_status": "resolved" if final_issue.resolution else "unresolved"
            }
            
        except Exception as e:
            return {
                "issue_id": issue.id,
                "error": str(e),
                "final_status": "error"
            }

# Graph Node Definitions (requires pydantic-graph)

if GRAPH_AVAILABLE:
    
    @dataclass
    class TriageNode(BaseNode[CustomerIssue, GraphDeps]):
        """Initial triage and routing node"""
        issue: CustomerIssue
        agents: Dict[str, Agent]
        
        async def run(self, ctx: GraphRunContext[CustomerIssue, GraphDeps]) -> Union[
            'TechnicalResolutionNode',
            'BillingResolutionNode', 
            'GeneralSupportNode',
            'EscalationNode'
        ]:
            self.issue.processing_history.append("Triage started")
            
            # Use triage agent to analyze the issue
            triage_agent = self.agents["triage"]
            
            analysis_prompt = f"""
            Analyze this customer issue and provide:
            1. Issue type: technical, billing, general, or escalation
            2. Complexity: simple, moderate, or complex
            3. Priority: 1-5 (5 being most urgent)
            
            Customer tier: {self.issue.customer_tier}
            Issue: {self.issue.description}
            
            Respond in JSON format:
            {{"issue_type": "...", "complexity": "...", "priority": ...}}
            """
            
            result = await triage_agent.run(analysis_prompt, deps=ctx.deps)
            
            try:
                analysis = json.loads(result.output)
                self.issue.issue_type = CustomerIssueType(analysis["issue_type"])
                self.issue.complexity = IssueComplexity(analysis["complexity"])
                self.issue.priority = analysis["priority"]
            except (json.JSONDecodeError, KeyError, ValueError):
                # Fallback to defaults
                self.issue.issue_type = CustomerIssueType.GENERAL
                self.issue.complexity = IssueComplexity.SIMPLE
                self.issue.priority = 2
            
            self.issue.processing_history.append(f"Triaged as {self.issue.issue_type} ({self.issue.complexity})")
            
            # Route based on issue type
            if self.issue.issue_type == CustomerIssueType.TECHNICAL:
                return TechnicalResolutionNode(issue=self.issue, agents=self.agents)
            elif self.issue.issue_type == CustomerIssueType.BILLING:
                return BillingResolutionNode(issue=self.issue, agents=self.agents)
            elif self.issue.issue_type == CustomerIssueType.ESCALATION:
                return EscalationNode(issue=self.issue, agents=self.agents)
            else:
                return GeneralSupportNode(issue=self.issue, agents=self.agents)
    
    @dataclass
    class TechnicalResolutionNode(BaseNode[CustomerIssue, GraphDeps]):
        """Technical issue resolution node"""
        issue: CustomerIssue
        agents: Dict[str, Agent]
        
        async def run(self, ctx: GraphRunContext[CustomerIssue, GraphDeps]) -> Union[
            'QualityCheckNode',
            'EscalationNode',
            'TechnicalResolutionNode'  # For retry
        ]:
            self.issue.processing_history.append("Technical resolution started")
            
            technical_agent = self.agents["technical"]
            
            resolution_prompt = f"""
            Provide a technical solution for this issue:
            
            Issue: {self.issue.description}
            Customer tier: {self.issue.customer_tier}
            Complexity: {self.issue.complexity}
            Previous attempts: {self.issue.attempted_solutions}
            
            If you can resolve it, provide a clear solution.
            If you cannot resolve it, recommend escalation with reasoning.
            """
            
            result = await technical_agent.run(resolution_prompt, deps=ctx.deps)
            
            # Determine if resolution was successful
            solution = result.output.lower()
            
            if "escalate" in solution or "cannot resolve" in solution:
                self.issue.escalation_level += 1
                self.issue.processing_history.append("Technical resolution failed - escalating")
                return EscalationNode(issue=self.issue, agents=self.agents)
            elif len(self.issue.attempted_solutions) >= ctx.deps.max_attempts:
                self.issue.escalation_level += 1
                self.issue.processing_history.append("Max attempts reached - escalating")
                return EscalationNode(issue=self.issue, agents=self.agents)
            else:
                self.issue.resolution = result.output
                self.issue.processing_history.append("Technical resolution provided")
                return QualityCheckNode(issue=self.issue, agents=self.agents)
    
    @dataclass
    class BillingResolutionNode(BaseNode[CustomerIssue, GraphDeps]):
        """Billing issue resolution node"""
        issue: CustomerIssue
        agents: Dict[str, Agent]
        
        async def run(self, ctx: GraphRunContext[CustomerIssue, GraphDeps]) -> Union[
            'QualityCheckNode',
            'EscalationNode'
        ]:
            self.issue.processing_history.append("Billing resolution started")
            
            billing_agent = self.agents["billing"]
            
            resolution_prompt = f"""
            Handle this billing issue:
            
            Issue: {self.issue.description}
            Customer tier: {self.issue.customer_tier}
            
            You can handle:
            - Billing explanations
            - Standard refunds under $100
            - Account adjustments under $50
            
            Escalate if:
            - Refund over $100
            - Policy exception needed
            - Complex dispute
            """
            
            result = await billing_agent.run(resolution_prompt, deps=ctx.deps)
            
            solution = result.output.lower()
            
            if "escalate" in solution or "exception" in solution:
                self.issue.escalation_level += 1
                self.issue.processing_history.append("Billing issue escalated")
                return EscalationNode(issue=self.issue, agents=self.agents)
            else:
                self.issue.resolution = result.output
                self.issue.processing_history.append("Billing resolution provided")
                return QualityCheckNode(issue=self.issue, agents=self.agents)
    
    @dataclass
    class GeneralSupportNode(BaseNode[CustomerIssue, GraphDeps]):
        """General support node"""
        issue: CustomerIssue
        agents: Dict[str, Agent]
        
        async def run(self, ctx: GraphRunContext[CustomerIssue, GraphDeps]) -> Union[
            'QualityCheckNode',
            'EscalationNode'
        ]:
            self.issue.processing_history.append("General support started")
            
            general_agent = self.agents["general"]
            
            resolution_prompt = f"""
            Provide general customer support for:
            
            Issue: {self.issue.description}
            Customer tier: {self.issue.customer_tier}
            
            Available knowledge:
            {json.dumps(ctx.deps.knowledge_base, indent=2)}
            
            Provide helpful information or escalate if needed.
            """
            
            result = await general_agent.run(resolution_prompt, deps=ctx.deps)
            
            self.issue.resolution = result.output
            self.issue.processing_history.append("General support response provided")
            
            return QualityCheckNode(issue=self.issue, agents=self.agents)
    
    @dataclass
    class EscalationNode(BaseNode[CustomerIssue, GraphDeps]):
        """Escalation handling node"""
        issue: CustomerIssue
        agents: Dict[str, Agent]
        
        async def run(self, ctx: GraphRunContext[CustomerIssue, GraphDeps]) -> 'ResolutionNode':
            self.issue.processing_history.append("Escalation handling started")
            
            escalation_agent = self.agents["escalation"]
            
            escalation_prompt = f"""
            Handle this escalated issue with full authority:
            
            Issue: {self.issue.description}
            Customer tier: {self.issue.customer_tier}
            Escalation level: {self.issue.escalation_level}
            Processing history: {self.issue.processing_history}
            
            You have authority to:
            - Make policy exceptions
            - Approve significant refunds
            - Provide compensation
            - Make final decisions
            
            Ensure customer satisfaction while protecting company interests.
            """
            
            result = await escalation_agent.run(escalation_prompt, deps=ctx.deps)
            
            self.issue.resolution = result.output
            self.issue.processing_history.append("Escalation resolution provided")
            
            return ResolutionNode(issue=self.issue)
    
    @dataclass
    class QualityCheckNode(BaseNode[CustomerIssue, GraphDeps]):
        """Quality check before final resolution"""
        issue: CustomerIssue
        agents: Dict[str, Agent]
        
        async def run(self, ctx: GraphRunContext[CustomerIssue, GraphDeps]) -> Union[
            'ResolutionNode',
            'EscalationNode'
        ]:
            self.issue.processing_history.append("Quality check started")
            
            # Simple quality checks
            resolution = self.issue.resolution or ""
            
            quality_issues = []
            
            if len(resolution) < 50:
                quality_issues.append("Resolution too brief")
            
            if "sorry" not in resolution.lower() and self.issue.complexity != IssueComplexity.SIMPLE:
                quality_issues.append("Missing empathy for complex issue")
            
            if self.issue.customer_tier == "enterprise" and "priority" not in resolution.lower():
                quality_issues.append("Missing priority acknowledgment for enterprise customer")
            
            if quality_issues and self.issue.escalation_level == 0:
                # First quality failure - escalate
                self.issue.escalation_level += 1
                self.issue.processing_history.append(f"Quality check failed: {', '.join(quality_issues)}")
                return EscalationNode(issue=self.issue, agents=self.agents)
            else:
                self.issue.processing_history.append("Quality check passed")
                return ResolutionNode(issue=self.issue)
    
    @dataclass
    class ResolutionNode(BaseNode[CustomerIssue, GraphDeps]):
        """Final resolution node"""
        issue: CustomerIssue
        
        async def run(self, ctx: GraphRunContext[CustomerIssue, GraphDeps]) -> End[CustomerIssue]:
            self.issue.processing_history.append("Issue resolved")
            return End(self.issue)

else:
    # Mock implementations when pydantic-graph is not available
    class TriageNode:
        def __init__(self, issue, agents):
            self.issue = issue
            self.agents = agents

async def demonstrate_graph_workflow():
    """Demonstrate the graph-based customer service workflow"""
    
    print("🎬 Graph-Based Customer Service Workflow Demo")
    print("="*55)
    
    if not GRAPH_AVAILABLE:
        print("⚠️ Using mock implementation - install pydantic-graph for full functionality")
    
    # Create the graph system
    service_graph = CustomerServiceGraph()
    
    # Test scenarios
    test_scenarios = [
        {
            "description": "My laptop won't start after the latest update. The screen stays black and I can hear the fan running.",
            "customer": {"tier": "premium", "name": "Alice Johnson", "id": "CUST001"},
            "expected_route": "technical"
        },
        {
            "description": "I was charged $500 for a subscription I cancelled last month. I need this refunded immediately.",
            "customer": {"tier": "basic", "name": "Bob Smith", "id": "CUST002"},
            "expected_route": "billing"
        },
        {
            "description": "What are your return policies for enterprise customers?",
            "customer": {"tier": "enterprise", "name": "Carol Wilson", "id": "CUST003"},
            "expected_route": "general"
        },
        {
            "description": "This is ridiculous! I've been trying to resolve this server outage for 3 hours and nobody can help me. I need to speak to someone in charge RIGHT NOW!",
            "customer": {"tier": "enterprise", "name": "David Brown", "id": "CUST004"},
            "expected_route": "escalation"
        }
    ]
    
    print(f"Processing {len(test_scenarios)} customer issues through the graph...")
    
    # Process each scenario
    results = []
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i}: {scenario['customer']['name']} ({scenario['customer']['tier']})")
        print(f"Issue: {scenario['description'][:80]}...")
        print('='*60)
        
        try:
            result = await service_graph.process_customer_issue(
                scenario["description"],
                scenario["customer"]
            )
            
            results.append(result)
            
            print(f"✅ Issue ID: {result['issue_id']}")
            print(f"Status: {result['final_status']}")
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"🔄 Processing steps: {len(result['processing_history'])}")
                for step in result['processing_history']:
                    print(f"   • {step}")
                
                if result['resolution']:
                    print(f"💡 Resolution: {result['resolution'][:150]}...")
                
                if result['escalation_level'] > 0:
                    print(f"🚨 Escalation level: {result['escalation_level']}")
            
        except Exception as e:
            print(f"❌ Failed to process scenario {i}: {e}")
            results.append({"error": str(e), "final_status": "error"})
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("📊 PROCESSING SUMMARY")
    print('='*60)
    
    successful = len([r for r in results if r.get('final_status') == 'resolved'])
    errors = len([r for r in results if r.get('final_status') == 'error'])
    escalations = len([r for r in results if r.get('escalation_level', 0) > 0])
    
    print(f"Total issues processed: {len(results)}")
    print(f"Successfully resolved: {successful}")
    print(f"Errors: {errors}")
    print(f"Required escalation: {escalations}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    
    # Average processing steps
    avg_steps = sum(len(r.get('processing_history', [])) for r in results) / len(results)
    print(f"Average processing steps: {avg_steps:.1f}")

if __name__ == "__main__":
    asyncio.run(demonstrate_graph_workflow())
```

---

## Practice Exercises for Module 8

### Exercise 1: Complex Graph Workflows
Create advanced graph workflows that include:
- Parallel processing branches
- Conditional routing based on multiple criteria
- Loop detection and prevention
- Dynamic node creation

### Exercise 2: Graph Optimization
Implement optimizations for:
- Path prediction and pre-warming
- Resource allocation across nodes
- Caching of intermediate results
- Performance monitoring and tuning

### Exercise 3: Graph Visualization
Build tools to:
- Visualize graph execution in real-time
- Generate execution reports and analytics
- Debug failed workflows
- Monitor graph performance metrics

### Exercise 4: Advanced Graph Patterns
Implement sophisticated patterns like:
- Saga patterns for distributed transactions
- Circuit breaker patterns in nodes
- Bulkhead isolation between graph sections
- Event sourcing for graph state

### Next Steps
Once you've mastered graph-based systems, you'll be ready for **Module 9: Production Deployment & Monitoring**, where we'll learn to deploy and monitor multi-agent systems in production environments.

**Key Takeaways:**
- Graph-based workflows provide powerful orchestration capabilities
- State machines enable complex decision logic
- Quality gates ensure output standards
- Node composition allows flexible workflow design
- Error handling and escalation paths are crucial
- Graph visualization helps with debugging and optimization