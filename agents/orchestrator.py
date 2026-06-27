from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.tools import tool
from firebase_admin import firestore
import json
from langgraph.prebuilt import create_react_agent

from tavily import TavilyClient
import os

# Initialize Tavily
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", "tvly-invalid"))

# Initialize tools
@tool
def search_web(query: str) -> str:
    """Use this tool to search the web for real-time information, jobs, or documentation."""
    try:
        response = tavily_client.search(query, search_depth="basic", max_results=5)
        results = response.get("results", [])
        if not results:
            return f"No search results found for '{query}'."
        
        formatted_results = f"Search Results for '{query}':\n"
        for i, res in enumerate(results):
            formatted_results += f"{i+1}. {res.get('title', 'No Title')} - {res.get('url', '')}\nSnippet: {res.get('content', '')}\n\n"
        return formatted_results + "These are all the available results. Do not search again."
    except Exception as e:
        return f"Error performing search: {str(e)}"

@tool
def get_user_profile(user_identifier: str) -> str:
    """Use this tool to retrieve the user's profile, location, skills, and projects from the database."""
    return "User is located in India, knows Next.js, React, and Python. Is interested in career opportunities including government exams."

tools = [search_web, get_user_profile]

# Initialize the model
model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.1)

from langchain_core.messages import SystemMessage, ToolMessage, AIMessage

model_with_tools = model.bind_tools(tools)

# Define the research node that can loop (uses tools)
async def research_node(state):
    messages = state["messages"]
    
    # We want to give it the initial prompt if it's the first message
    if len(messages) == 1: # Only human message
        sys_msg_init = SystemMessage(content=(
            "You are Forge Assistant, an AI that helps users find jobs, government exams, scholarships, and career opportunities. "
            "You can use the search_web tool multiple times to find specific details, criteria, and official links. "
            "Search for up-to-date 2026 information. "
            "CRITICAL: Do NOT search the exact same query twice. If a search fails or doesn't have what you want, try different keywords! "
            "When you have gathered enough information, output a message saying exactly: 'DONE'"
        ))
        response = await model_with_tools.ainvoke([sys_msg_init] + messages)
    else:
        response = await model_with_tools.ainvoke(messages)
        
    return {"messages": [response]}

# Define the synthesis node that generates the Markdown (NO tools bound)
async def synthesize_node(state):
    messages = state["messages"]
    
    sys_msg = SystemMessage(content=(
        "You have gathered enough information from your research. Provide your final summary to the user immediately. "
        "You MUST format your answer as beautiful, conversational Markdown text (like ChatGPT or Gemini). "
        "DO NOT use JSON. "
        "If you found exams or opportunities, use bullet points or headers to make it readable. "
        "Include the exact criteria, dates, and most importantly, provide direct Markdown links (e.g., [Apply Here](URL)) to the official sources so the user can fill out the forms. "
        "Make the tone helpful, professional, and highly organized."
    ))
    
    response = await model.ainvoke([sys_msg] + messages)
    return {"messages": [response]}

async def tools_node(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_responses = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        if tool_name == "search_web":
            result = await search_web.ainvoke(tool_args)
        elif tool_name == "get_user_profile":
            result = await get_user_profile.ainvoke(tool_args)
        else:
            result = "Tool not found."
            
        tool_responses.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_name))
        
    return {"messages": tool_responses}

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check how many tool passes have happened
    tool_pass_count = sum(1 for m in messages if isinstance(m, ToolMessage))
    
    # If the model called tools, and we haven't exceeded 3 tool passes, go to tools
    if last_message.tool_calls and tool_pass_count < 3: 
        return "tools_node"
        
    # If no tool calls, or if it said 'DONE', or if we hit the limit, synthesize!
    return "synthesize_node"

from langgraph.graph import StateGraph, add_messages
from typing import TypedDict, Annotated
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

workflow = StateGraph(AgentState)
workflow.add_node("research_node", research_node)
workflow.add_node("tools_node", tools_node)
workflow.add_node("synthesize_node", synthesize_node)

workflow.set_entry_point("research_node")
workflow.add_conditional_edges("research_node", should_continue)
workflow.add_edge("tools_node", "research_node")
workflow.add_edge("synthesize_node", "__end__")
agent_executor = workflow.compile()
