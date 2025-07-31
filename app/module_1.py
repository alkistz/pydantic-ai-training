import asyncio

from pydantic_ai import Agent
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

agent = Agent(model)


async def run_agent(message: str):
    result = await agent.run(message)
    print(result)

if __name__ == "__main__":
    asyncio.run(run_agent("Is this working?"))
