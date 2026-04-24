
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv(override=True)
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch



tavily = TavilyClient()

def main():
    llm = ChatOpenAI()
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools)
    result = agent.invoke({"messages":HumanMessage("What's is the whether in Pune")})

    print(result["messages"][-1].content)

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
#uv run tavily_webseach_with_langchain.py
if __name__ == "__main__":
    main()
