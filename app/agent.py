from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from app.tools import search_papers, compare_papers

# Initialise LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# Bind tools
tools = [search_papers, compare_papers]
llm_with_tools = llm.bind_tools(tools)

# Define nodes
def call_llm(state: MessagesState):
    messages = state["messages"]
    
    # Add system prompt if not already present
    if not any(isinstance(m, SystemMessage) for m in messages):
        system = SystemMessage(content="""You are a research assistant. 
Answer questions ONLY based on the content retrieved from the research papers. 
Do not use your general knowledge. If the retrieved content doesn't answer 
the question, say so clearly.""")
        messages = [system] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# Build graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", tools_condition)
graph_builder.add_edge("tools", "llm")

# Compile
graph = graph_builder.compile()

def run_agent(question: str) -> str:
    """Run the research assistant agent with a question."""
    result = graph.invoke({
        "messages": [HumanMessage(content=question)]
    })
    return result["messages"][-1].content