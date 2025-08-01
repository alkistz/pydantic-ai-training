import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.azure import AzureProvider
from settings import settings

model = OpenAIModel(
    "gpt-4o",
    provider=AzureProvider(
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        api_key=settings.azure_openai_api_key,
    ),
)

# agent = Agent(model, retries=3),


@dataclass
class Customer:
    id: str
    name: str
    email: str
    tier: str  # "basic", "premium", "enterprise"
    join_date: datetime
    total_orders: int
    last_order_date: Optional[datetime] = None


@dataclass
class Order:
    id: str
    customer_id: str
    product: str
    status: str  # "processing", "shipped", "delivered", "returned"
    order_date: datetime
    amount: float


# Mock database
class CustomerDatabase:
    def __init__(self):
        self.customers = {
            "CUST001": Customer(
                "CUST001",
                "John Doe",
                "john@email.com",
                "premium",
                datetime(2023, 1, 15),
                5,
                datetime(2024, 10, 1),
            ),
            "CUST002": Customer(
                "CUST002",
                "Jane Smith",
                "jane@email.com",
                "basic",
                datetime(2024, 3, 10),
                2,
                datetime(2024, 9, 15),
            ),
        }

        self.orders = {
            "ORD001": Order(
                "ORD001",
                "CUST001",
                "MacBook Pro",
                "delivered",
                datetime(2024, 10, 1),
                2499.99,
            ),
            "ORD002": Order(
                "ORD002",
                "CUST001",
                "iPhone Case",
                "shipped",
                datetime(2024, 10, 15),
                49.99,
            ),
            "ORD003": Order(
                "ORD003",
                "CUST002",
                "iPad",
                "processing",
                datetime(2024, 10, 20),
                799.99,
            ),
        }

    def get_customer_by_email(self, email: str) -> Optional[Customer]:
        for customer in self.customers.values():
            if customer.email.lower() == email.lower():
                return customer
        return None

    def get_customer_orders(self, customer_id: str) -> list[Order]:
        return [
            order for order in self.orders.values() if order.customer_id == customer_id
        ]


# Dependencies for the agent
@dataclass
class CustomerServiceDeps:
    db: CustomerDatabase


system_prompt = """
    You are a customer service representative with access to customer and order information.
    Always be helpful and use the available tools to look up accurate information.
    Protect customer privacy - only share information with verified customers.
    """

customer_service_agent = Agent(
    model, deps_type=CustomerServiceDeps, system_prompt=system_prompt, retries=2
)


@customer_service_agent.tool
async def lookup_customer(ctx: RunContext[CustomerServiceDeps], email: str) -> str:
    """Look up customer information by email address.

    Args:
        email: Customer's email address
    """
    customer = ctx.deps.db.get_customer_by_email(email)
    if not customer:
        return f"No customer found with email {email}"

    return f"""
    Customer Information:
    - Name: {customer.name}
    - Customer ID: {customer.id}
    - Tier: {customer.tier}
    - Member since: {customer.join_date.strftime("%B %Y")}
    - Total orders: {customer.total_orders}
    - Last order: {customer.last_order_date.strftime("%B %d, %Y") if customer.last_order_date else "None"}
    """


@customer_service_agent.tool
async def get_order_status(
    ctx: RunContext[CustomerServiceDeps], customer_email: str
) -> str:
    """Get all orders for a customer by their email.

    Args:
        customer_email: Customer's email address
    """
    customer = ctx.deps.db.get_customer_by_email(customer_email)
    if not customer:
        return f"No customer found with email: {customer_email}"

    orders = ctx.deps.db.get_customer_orders(customer.id)
    if not orders:
        return f"No orders found for {customer.name}"

    order_info = f"Orders for {customer.name}:\n"
    for order in orders:
        order_info += f"""
        Order #{order.id}:
        - Product: {order.product}
        - Status: {order.status}
        - Date: {order.order_date.strftime("%B %d, %Y")}
        - Amount: ${order.amount:.2f}
        """

    return order_info


async def run_agent(message: str):
    db = CustomerDatabase()
    deps = CustomerServiceDeps(db)
    result = await customer_service_agent.run(message, deps=deps)
    print(result.output)




if __name__ == "__main__":
    asyncio.run(run_agent("What do you know about jane@email.com? What about their orders?"))
