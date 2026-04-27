from http.client import responses

from  dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from langgraph_agents.react import tools, llm
from langgraph_agents.state import AgentState

load_dotenv(override=True)

SYSTEM_MESSAGE = """
You MUST use tools when required.

- Use TavilySearch for real-time data like weather
- Use triple tool for calculations
- Never guess if tool is available
"""
def run_agent_reasoning(state: AgentState) -> AgentState:
    """
    Run the agent reasoning node.
    """
    steps = state.get("steps", 0) + 1
    #response = llm.invoke([{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]])
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_MESSAGE)] + state["messages"]
    )
    #return {"messages": [response]}
    print("LLM response(run_agent_reasoning):", response)
    return {
            "messages": state["messages"] + [response],
            "steps": steps
    }


tool_node = ToolNode(tools)