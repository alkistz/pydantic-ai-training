# Module 2: Tools & Function Calling
## Detailed Code Examples

### Learning Objectives
- Understand tool/function calling concepts
- Create custom tools for agents
- Master tool schemas and parameter validation
- Handle tool errors and edge cases

---

## 2.1 Basic Tool Creation - Customer Database Tools

```python
# customer_database_tools.py
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

# Simulated customer database
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
            "CUST001": Customer("CUST001", "John Doe", "john@email.com", "premium", 
                              datetime(2023, 1, 15), 5, datetime(2024, 10, 1)),
            "CUST002": Customer("CUST002", "Jane Smith", "jane@email.com", "basic",
                              datetime(2024, 3, 10), 2, datetime(2024, 9, 15)),
        }
        
        self.orders = {
            "ORD001": Order("ORD001", "CUST001", "MacBook Pro", "delivered", 
                          datetime(2024, 10, 1), 2499.99),
            "ORD002": Order("ORD002", "CUST001", "iPhone Case", "shipped",
                          datetime(2024, 10, 15), 49.99),
            "ORD003": Order("ORD003", "CUST002", "iPad", "processing",
                          datetime(2024, 10, 20), 799.99),
        }
    
    def get_customer_by_email(self, email: str) -> Optional[Customer]:
        for customer in self.customers.values():
            if customer.email.lower() == email.lower():
                return customer
        return None
    
    def get_customer_orders(self, customer_id: str) -> List[Order]:
        return [order for order in self.orders.values() 
                if order.customer_id == customer_id]

# Dependencies for the agent
@dataclass
class CustomerServiceDeps:
    db: CustomerDatabase

# Create agent with customer service tools
customer_service_agent = Agent(
    'openai:gpt-4o',
    deps_type=CustomerServiceDeps,
    system_prompt="""
    You are a customer service representative with access to customer and order information.
    Always be helpful and use the available tools to look up accurate information.
    Protect customer privacy - only share information with verified customers.
    """,
    retries=2
)

@customer_service_agent.tool
async def lookup_customer(ctx: RunContext[CustomerServiceDeps], email: str) -> str:
    """Look up customer information by email address.
    
    Args:
        email: Customer's email address
    """
    customer = ctx.deps.db.get_customer_by_email(email)
    if not customer:
        return f"No customer found with email: {email}"
    
    return f"""
    Customer Information:
    - Name: {customer.name}
    - Customer ID: {customer.id}
    - Tier: {customer.tier}
    - Member since: {customer.join_date.strftime('%B %Y')}
    - Total orders: {customer.total_orders}
    - Last order: {customer.last_order_date.strftime('%B %d, %Y') if customer.last_order_date else 'None'}
    """

@customer_service_agent.tool
async def get_order_status(ctx: RunContext[CustomerServiceDeps], customer_email: str) -> str:
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
        - Date: {order.order_date.strftime('%B %d, %Y')}
        - Amount: ${order.amount:.2f}
        """
    
    return order_info

@customer_service_agent.tool
async def update_order_status(
    ctx: RunContext[CustomerServiceDeps], 
    order_id: str, 
    new_status: str,
    reason: str = ""
) -> str:
    """Update the status of an order.
    
    Args:
        order_id: The order ID to update
        new_status: New status (processing, shipped, delivered, returned)
        reason: Reason for the status change
    """
    valid_statuses = ["processing", "shipped", "delivered", "returned", "cancelled"]
    if new_status not in valid_statuses:
        return f"Invalid status. Valid options: {', '.join(valid_statuses)}"
    
    if order_id in ctx.deps.db.orders:
        old_status = ctx.deps.db.orders[order_id].status
        ctx.deps.db.orders[order_id].status = new_status
        
        update_msg = f"Order {order_id} status updated from '{old_status}' to '{new_status}'"
        if reason:
            update_msg += f"\nReason: {reason}"
        
        return update_msg
    else:
        return f"Order {order_id} not found"

@customer_service_agent.tool
async def calculate_refund(ctx: RunContext[CustomerServiceDeps], order_id: str) -> str:
    """Calculate potential refund amount for an order.
    
    Args:
        order_id: The order ID to calculate refund for
    """
    if order_id not in ctx.deps.db.orders:
        return f"Order {order_id} not found"
    
    order = ctx.deps.db.orders[order_id]
    days_since_order = (datetime.now() - order.order_date).days
    
    # Refund policy logic
    if days_since_order <= 30:
        refund_percentage = 100
    elif days_since_order <= 60:
        refund_percentage = 50
    else:
        refund_percentage = 0
    
    refund_amount = order.amount * (refund_percentage / 100)
    
    return f"""
    Refund calculation for Order {order_id}:
    - Original amount: ${order.amount:.2f}
    - Days since order: {days_since_order}
    - Refund percentage: {refund_percentage}%
    - Refund amount: ${refund_amount:.2f}
    """

async def demonstrate_customer_tools():
    # Setup
    db = CustomerDatabase()
    deps = CustomerServiceDeps(db)
    
    # Simulate customer service interactions
    interactions = [
        "Hi, I'm john@email.com and I want to check my order status",
        "Can you help me return my MacBook Pro order? I'm not satisfied with it.",
        "I'm jane@email.com and my iPad order seems to be stuck in processing. Can you update it to shipped?"
    ]
    
    for interaction in interactions:
        print(f"\n{'='*60}")
        print(f"Customer: {interaction}")
        print('='*60)
        
        result = await customer_service_agent.run(interaction, deps=deps)
        print(f"Agent: {result.output}")
        print(f"Usage: {result.usage()}")

if __name__ == "__main__":
    asyncio.run(demonstrate_customer_tools())
```

**Key Concepts:**
- **Tool Declaration**: Using `@agent.tool` decorator
- **RunContext**: Accessing dependencies within tools
- **Parameter Documentation**: Clear docstrings for better LLM understanding
- **Data Validation**: Checking inputs and providing meaningful errors

---

## 2.2 Advanced Tool Validation and Error Handling

```python
# advanced_tool_validation.py
import asyncio
import re
from typing import Dict, List, Optional
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
from datetime import datetime

class ProductSearchQuery(BaseModel):
    """Structured product search parameters"""
    category: Optional[str] = Field(None, description="Product category")
    price_min: Optional[float] = Field(None, ge=0, description="Minimum price")
    price_max: Optional[float] = Field(None, ge=0, description="Maximum price")
    brand: Optional[str] = Field(None, description="Product brand")
    in_stock: bool = Field(True, description="Only show in-stock items")
    
    @validator('price_max')
    def validate_price_range(cls, v, values):
        if v is not None and 'price_min' in values and values['price_min'] is not None:
            if v <= values['price_min']:
                raise ValueError('price_max must be greater than price_min')
        return v

@dataclass
class Product:
    id: str
    name: str
    category: str
    brand: str
    price: float
    stock: int
    rating: float

@dataclass
class ProductDeps:
    products: List[Product]

# Mock product database
def create_mock_products() -> List[Product]:
    return [
        Product("P001", "MacBook Pro 16\"", "laptops", "Apple", 2499.99, 5, 4.8),
        Product("P002", "Dell XPS 13", "laptops", "Dell", 1299.99, 12, 4.6),
        Product("P003", "iPhone 15 Pro", "phones", "Apple", 999.99, 20, 4.7),
        Product("P004", "Samsung Galaxy S24", "phones", "Samsung", 899.99, 15, 4.5),
        Product("P005", "iPad Air", "tablets", "Apple", 599.99, 8, 4.6),
        Product("P006", "Surface Laptop", "laptops", "Microsoft", 1199.99, 0, 4.4),  # Out of stock
    ]

product_agent = Agent(
    'openai:gpt-4o',
    deps_type=ProductDeps,
    system_prompt="""
    You are a product specialist helping customers find the right products.
    Use the search tools to find products that match customer needs.
    Always mention stock availability and suggest alternatives if needed.
    """,
    retries=2
)

@product_agent.tool
async def search_products(
    ctx: RunContext[ProductDeps], 
    query: ProductSearchQuery
) -> str:
    """Search for products based on specified criteria.
    
    Args:
        query: Structured search parameters including category, price range, brand, etc.
    """
    try:
        products = ctx.deps.products
        results = []
        
        for product in products:
            # Apply filters
            if query.category and product.category != query.category.lower():
                continue
            if query.brand and product.brand.lower() != query.brand.lower():
                continue
            if query.price_min and product.price < query.price_min:
                continue
            if query.price_max and product.price > query.price_max:
                continue
            if query.in_stock and product.stock == 0:
                continue
            
            results.append(product)
        
        if not results:
            return "No products found matching your criteria. Try expanding your search parameters."
        
        # Format results
        response = f"Found {len(results)} product(s):\n\n"
        for product in results[:10]:  # Limit to 10 results
            stock_status = "In Stock" if product.stock > 0 else "Out of Stock"
            response += f"""
            {product.name} ({product.brand})
            - Price: ${product.price:.2f}
            - Rating: {product.rating}/5.0
            - Stock: {stock_status} ({product.stock} available)
            - Category: {product.category}
            ---
            """
        
        return response
        
    except Exception as e:
        raise ModelRetry(f"Error searching products: {str(e)}")

@product_agent.tool
async def get_product_details(ctx: RunContext[ProductDeps], product_id: str) -> str:
    """Get detailed information about a specific product.
    
    Args:
        product_id: The unique product identifier
    """
    if not re.match(r'^P\d{3}$', product_id):
        return "Invalid product ID format. Product IDs should be in format P001, P002, etc."
    
    product = next((p for p in ctx.deps.products if p.id == product_id), None)
    
    if not product:
        available_ids = [p.id for p in ctx.deps.products]
        return f"Product {product_id} not found. Available IDs: {', '.join(available_ids)}"
    
    stock_status = "✅ In Stock" if product.stock > 0 else "❌ Out of Stock"
    
    return f"""
    📱 {product.name}
    
    Brand: {product.brand}
    Category: {product.category.title()}
    Price: ${product.price:.2f}
    Rating: {"⭐" * int(product.rating)} {product.rating}/5.0
    Availability: {stock_status} ({product.stock} units)
    Product ID: {product.id}
    """

@product_agent.tool
async def check_compatibility(
    ctx: RunContext[ProductDeps], 
    product1_id: str, 
    product2_id: str
) -> str:
    """Check if two products are compatible with each other.
    
    Args:
        product1_id: First product ID
        product2_id: Second product ID
    """
    product1 = next((p for p in ctx.deps.products if p.id == product1_id), None)
    product2 = next((p for p in ctx.deps.products if p.id == product2_id), None)
    
    if not product1:
        return f"Product {product1_id} not found"
    if not product2:
        return f"Product {product2_id} not found"
    
    # Simple compatibility logic based on brands and categories
    compatibility_rules = {
        ("Apple", "Apple"): "✅ Fully compatible - Same ecosystem",
        ("Apple", "laptops"): "⚠️ Limited compatibility - May need adapters",
        ("Samsung", "Samsung"): "✅ Fully compatible - Same brand",
    }
    
    # Check brand compatibility
    brand_key = (product1.brand, product2.brand)
    if brand_key in compatibility_rules:
        compatibility = compatibility_rules[brand_key]
    elif product1.brand == product2.brand:
        compatibility = "✅ Likely compatible - Same brand"
    else:
        compatibility = "⚠️ Check specifications - Different brands"
    
    return f"""
    Compatibility Check:
    {product1.name} ({product1.brand}) + {product2.name} ({product2.brand})
    
    Result: {compatibility}
    
    Note: Always verify specific technical requirements for your use case.
    """

async def demonstrate_advanced_tools():
    deps = ProductDeps(create_mock_products())
    
    test_queries = [
        "I'm looking for Apple laptops under $2000",
        "Show me all phones in stock with good ratings",
        "What are the details for product P001?",
        "Are the MacBook Pro and iPhone compatible?",
        "I need a tablet for drawing, what do you recommend?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Customer: {query}")
        print('='*70)
        
        try:
            result = await product_agent.run(query, deps=deps)
            print(result.output)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(demonstrate_advanced_tools())
```

---

## 2.3 Tool Chaining and Complex Workflows

```python
# tool_chaining_workflows.py
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from datetime import datetime, timedelta
import json

class InventoryItem(BaseModel):
    product_id: str
    quantity: int
    location: str
    last_updated: datetime

class ShippingQuote(BaseModel):
    carrier: str
    service: str
    cost: float
    estimated_days: int

@dataclass
class OrderProcessingDeps:
    inventory: Dict[str, InventoryItem]
    shipping_rates: Dict[str, List[ShippingQuote]]
    customer_addresses: Dict[str, str]

# Create comprehensive order processing agent
order_processor = Agent(
    'openai:gpt-4o',
    deps_type=OrderProcessingDeps,
    system_prompt="""
    You are an order processing specialist. Use the available tools to:
    1. Check product availability
    2. Reserve inventory
    3. Calculate shipping options
    4. Process orders end-to-end
    
    Always verify each step before proceeding to the next.
    """,
    retries=2
)

@order_processor.tool
async def check_inventory(
    ctx: RunContext[OrderProcessingDeps], 
    product_id: str, 
    quantity: int
) -> str:
    """Check if sufficient inventory is available for a product.
    
    Args:
        product_id: Product identifier
        quantity: Quantity needed
    """
    if product_id not in ctx.deps.inventory:
        return f"Product {product_id} not found in inventory"
    
    item = ctx.deps.inventory[product_id]
    
    if item.quantity >= quantity:
        return f"""
        ✅ Inventory Available:
        - Product: {product_id}
        - Available: {item.quantity} units
        - Requested: {quantity} units
        - Location: {item.location}
        - Last updated: {item.last_updated.strftime('%Y-%m-%d %H:%M')}
        """
    else:
        return f"""
        ❌ Insufficient Inventory:
        - Product: {product_id}
        - Available: {item.quantity} units
        - Requested: {quantity} units
        - Shortage: {quantity - item.quantity} units
        """

@order_processor.tool
async def reserve_inventory(
    ctx: RunContext[OrderProcessingDeps], 
    product_id: str, 
    quantity: int,
    customer_id: str
) -> str:
    """Reserve inventory for a customer order.
    
    Args:
        product_id: Product identifier
        quantity: Quantity to reserve
        customer_id: Customer making the order
    """
    if product_id not in ctx.deps.inventory:
        return f"Product {product_id} not found in inventory"
    
    item = ctx.deps.inventory[product_id]
    
    if item.quantity < quantity:
        return f"Cannot reserve {quantity} units - only {item.quantity} available"
    
    # Reserve inventory (reduce available quantity)
    item.quantity -= quantity
    item.last_updated = datetime.now()
    
    reservation_id = f"RES_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return f"""
    ✅ Inventory Reserved:
    - Reservation ID: {reservation_id}
    - Product: {product_id}
    - Quantity: {quantity} units
    - Customer: {customer_id}
    - Remaining inventory: {item.quantity} units
    """

@order_processor.tool
async def get_shipping_quotes(
    ctx: RunContext[OrderProcessingDeps], 
    customer_id: str, 
    product_weight: float = 2.0
) -> str:
    """Get shipping quotes for a customer's location.
    
    Args:
        customer_id: Customer identifier
        product_weight: Weight of the package in pounds
    """
    if customer_id not in ctx.deps.customer_addresses:
        return f"Customer {customer_id} address not found"
    
    address = ctx.deps.customer_addresses[customer_id]
    
    # Get shipping quotes based on location
    quotes = ctx.deps.shipping_rates.get(customer_id, [
        ShippingQuote(carrier="UPS", service="Ground", cost=8.99, estimated_days=5),
        ShippingQuote(carrier="FedEx", service="Ground", cost=9.49, estimated_days=4),
        ShippingQuote(carrier="USPS", service="Priority", cost=7.99, estimated_days=3),
    ])
    
    quote_text = f"Shipping quotes for {address}:\n\n"
    for quote in quotes:
        quote_text += f"""
        {quote.carrier} {quote.service}:
        - Cost: ${quote.cost:.2f}
        - Delivery: {quote.estimated_days} business days
        """
    
    return quote_text

@order_processor.tool
async def calculate_order_total(
    ctx: RunContext[OrderProcessingDeps], 
    product_price: float, 
    quantity: int, 
    shipping_cost: float,
    tax_rate: float = 0.08
) -> str:
    """Calculate total order cost including tax and shipping.
    
    Args:
        product_price: Price per unit
        quantity: Number of units
        shipping_cost: Shipping cost
        tax_rate: Tax rate (default 8%)
    """
    subtotal = product_price * quantity
    tax = subtotal * tax_rate
    total = subtotal + tax + shipping_cost
    
    return f"""
    Order Total Calculation:
    - Subtotal: ${subtotal:.2f} ({quantity} × ${product_price:.2f})
    - Tax ({tax_rate*100:.1f}%): ${tax:.2f}
    - Shipping: ${shipping_cost:.2f}
    - TOTAL: ${total:.2f}
    """

@order_processor.tool
async def process_complete_order(
    ctx: RunContext[OrderProcessingDeps], 
    customer_id: str, 
    product_id: str, 
    quantity: int, 
    product_price: float
) -> str:
    """Process a complete order from start to finish using other tools.
    
    Args:
        customer_id: Customer identifier
        product_id: Product identifier
        quantity: Quantity to order
        product_price: Price per unit
    """
    steps_completed = []
    
    # Step 1: Check inventory
    inventory_result = await check_inventory(ctx, product_id, quantity)
    steps_completed.append(f"Inventory Check: {inventory_result.split(':')[0]}")
    
    if "❌" in inventory_result:
        return f"Order failed at inventory check:\n{inventory_result}"
    
    # Step 2: Get shipping quotes
    shipping_result = await get_shipping_quotes(ctx, customer_id)
    steps_completed.append("Shipping Quotes: Retrieved")
    
    # Step 3: Reserve inventory
    reservation_result = await reserve_inventory(ctx, product_id, quantity, customer_id)
    steps_completed.append(f"Inventory Reservation: {reservation_result.split(':')[0]}")
    
    if "❌" in reservation_result:
        return f"Order failed at inventory reservation:\n{reservation_result}"
    
    # Step 4: Calculate total (using cheapest shipping)
    cheapest_shipping = 7.99  # Would parse from shipping_result in real implementation
    total_result = await calculate_order_total(ctx, product_price, quantity, cheapest_shipping)
    steps_completed.append("Order Total: Calculated")
    
    order_id = f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return f"""
    ✅ Order Processed Successfully!
    
    Order ID: {order_id}
    Customer: {customer_id}
    
    Steps Completed:
    {chr(10).join(f"  ✓ {step}" for step in steps_completed)}
    
    Order Details:
    {total_result}
    
    Next Steps:
    - Payment processing
    - Fulfillment preparation
    - Shipping label creation
    """

async def demonstrate_tool_chaining():
    # Setup mock data
    inventory = {
        "LAPTOP001": InventoryItem(
            product_id="LAPTOP001",
            quantity=10,
            location="Warehouse A",
            last_updated=datetime.now() - timedelta(hours=2)
        ),
        "PHONE001": InventoryItem(
            product_id="PHONE001",
            quantity=25,
            location="Warehouse B",
            last_updated=datetime.now() - timedelta(minutes=30)
        )
    }
    
    customer_addresses = {
        "CUST001": "123 Main St, Seattle, WA",
        "CUST002": "456 Oak Ave, Portland, OR"
    }
    
    deps = OrderProcessingDeps(
        inventory=inventory,
        shipping_rates={},
        customer_addresses=customer_addresses
    )
    
    # Test scenarios
    scenarios = [
        "Customer CUST001 wants to order 2 units of LAPTOP001 at $1299.99 each",
        "Check if we have 5 units of PHONE001 available",
        "Get shipping quotes for customer CUST002",
        "Process a complete order: CUST001 ordering 1 LAPTOP001 at $1299.99"
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario}")
        print('='*60)
        
        result = await order_processor.run(scenario, deps=deps)
        print(result.output)

if __name__ == "__main__":
    asyncio.run(demonstrate_tool_chaining())
```

**Key Concepts:**
- **Tool Chaining**: Tools calling other tools to complete complex workflows
- **State Management**: Tracking progress through multi-step processes
- **Error Propagation**: Handling failures at any step in the chain
- **Business Logic**: Implementing real-world business processes

---

## Practice Exercises for Module 2

### Exercise 1: E-commerce Tools
Create tools for an e-commerce system:
- `search_products(query, filters)`
- `add_to_cart(customer_id, product_id, quantity)`
- `apply_discount_code(cart_id, code)`
- `calculate_shipping(cart_id, zip_code)`

### Exercise 2: Error Handling
Implement robust error handling for tools that:
- Validate all inputs before processing
- Return structured error messages
- Use ModelRetry for recoverable errors
- Log errors for debugging

### Exercise 3: Tool Documentation
Create tools with comprehensive documentation that includes:
- Clear parameter descriptions
- Example usage
- Return value specifications
- Error conditions

### Exercise 4: Advanced Validation
Build tools that use Pydantic models for:
- Complex input validation
- Structured output formats
- Custom validation rules
- Type conversion and coercion

### Next Steps
Once you've mastered tools and function calling, you'll be ready for **Module 3: Dependencies & State Management**, where we'll learn how to manage complex state and dependencies across agent interactions.

**Key Takeaways:**
- Tools extend agent capabilities with real-world actions
- Always validate inputs and handle errors gracefully
- Use clear documentation to help the LLM understand tools
- Chain tools together for complex multi-step workflows
- Structure tool outputs for easy parsing and display