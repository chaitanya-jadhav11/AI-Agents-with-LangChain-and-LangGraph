from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph,MessagesState
from langgraph_agents.state import AgentState
from langgraph_agents.nodes import tool_node, run_agent_reasoning



load_dotenv(verbose=True)


AGENT_REASON="agent_reason"
ACT= "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    if state.get("steps", 0) >= 5:
        return END

    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


flow = StateGraph(AgentState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT})

flow.add_edge(ACT, AGENT_REASON)


app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

# uv run -m langgraph_agents.main
if __name__ == "__main__":
    print("Hello, LangGraph Agents!")

    initial_state: AgentState = {
        "messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")],
        "steps": 0
    }
    # noinspection PyTypeChecker
    res = app.invoke(initial_state)

    print(f"steps", res["steps"])

    print(
        "Final response:",
        res["messages"][-1].content
    )

