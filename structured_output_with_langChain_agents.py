from typing import List, cast

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv(override=True)
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

tavily = TavilyClient()

def main():
    llm = ChatOpenAI()
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools,response_format=AgentResponse)
    result = agent.invoke({"messages": HumanMessage("What's is the whether in Pune")})

    print(result["structured_response"].sources)


    """
     output 

     The current weather in Pune is as follows:
    - Temperature: 28.8°C (83.9°F)
    - Condition: Clear
    - Wind: 7.2 mph from SW
    - Humidity: 39%
    - Pressure: 1012.0 mb
    - Visibility: 10.0 km
    - UV Index: 0.0
    """


# uv run structured_output_with_langChain_agents.py
if __name__ == "__main__":
    main()
