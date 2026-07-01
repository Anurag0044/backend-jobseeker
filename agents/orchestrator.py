from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import json
from langgraph.prebuilt import create_react_agent

from tavily import TavilyClient
import os
import logging

logger = logging.getLogger(__name__)

# Initialize Tavily
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", "tvly-invalid"))

# Initialize tools
@tool
def search_web(query: str) -> str:
    """Use this tool to search the web for real-time information, jobs, or documentation."""
    try:
        response = tavily_client.search(query, search_depth="advanced", max_results=8)
        results = response.get("results", [])
        if not results:
            return f"No search results found for '{query}'."
        
        formatted_results = f"Search Results for '{query}':\n"
        for i, res in enumerate(results):
            formatted_results += f"{i+1}. {res.get('title', 'No Title')} - {res.get('url', '')}\nSnippet: {res.get('content', '')}\n\n"
        return formatted_results
    except Exception as e:
        return f"Error performing search: {str(e)}"

@tool
def get_user_profile(user_identifier: str) -> str:
    """Use this tool to retrieve the user's profile, location, skills, and projects from the database."""
    return "User is located in India, knows Next.js, React, and Python. Is interested in career opportunities including government exams."

tools = [search_web, get_user_profile]

from langchain_core.messages import SystemMessage, ToolMessage, AIMessage

# ── Model Pool ────────────────────────────────────────────────────────────────
# We define individual models. The fallback logic is handled manually inside
# each node via try/except, because LangChain's .with_fallbacks() silently
# hangs when used with astream_events (the SSE streaming method).
# ──────────────────────────────────────────────────────────────────────────────

# Primary: NVIDIA Llama 3.1 8B — blazing fast, always available
primary_model = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", temperature=0.1)

# Backup: Gemini — massive context, used when NVIDIA is down
backup_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

# Fast Model specifically for large document generation (Resumes/CVs)
fast_generation_model = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", temperature=0.1)

# Corporate Model: Llama 3.1 8B in JSON mode for blazing-fast, guaranteed valid JSON document generation (bypasses Gemini rate limits)
corporate_model = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    temperature=0.1,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# Backup Corporate: Llama 8B
backup_corporate_model = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    temperature=0.1,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# Bind tools to each model independently
primary_with_tools = primary_model.bind_tools(tools)
backup_with_tools = backup_model.bind_tools(tools)


async def _invoke_with_fallback(messages, use_tools=False):
    """
    Try the primary model (NVIDIA Llama). If it fails for any reason,
    fall back to the backup model (Gemini). This is a manual fallback
    that works correctly with astream_events.
    """
    models = (
        [primary_with_tools, backup_with_tools] if use_tools
        else [primary_model, backup_model]
    )
    
    last_error = None
    for i, model in enumerate(models):
        try:
            response = await model.ainvoke(messages)
            return response
        except Exception as e:
            last_error = e
            model_name = "NVIDIA Llama" if i == 0 else "Gemini"
            logger.warning(f"Model {model_name} failed: {e}. Trying next...")
    
    # If all models fail, raise the last error
    raise last_error


# Define the research node that can loop (uses tools)
async def research_node(state):
    messages = state["messages"]
    
    sys_msg_init = SystemMessage(content=(
        "You are AgentX, an elite Expert Research Agent designed by ForgeX. "
        "Your SOLE DOMAIN is jobs, careers, government exams, scholarships, internships, skill development, and professional growth. "
        "\n\nGUARDRAILS (STRICTLY ENFORCED):\n"
        "- You MUST REFUSE any request that is NOT related to careers, jobs, education, or professional development.\n"
        "- STRICT MODE SEPARATION: You are the EXPERT research agent. You CANNOT generate Resumes, CVs, or Portfolios. If the user asks for document generation, politely apologize and explicitly instruct them to switch to 'Corporate Mode'.\n"
        "- STRICT MODE SEPARATION: If the user asks simple, basic conversational questions that do NOT require deep web research, politely suggest they switch to 'Instant Mode' for faster answers.\n"
        "- If the user asks about anything harmful, illegal, political, sexual, or completely off-topic, respond with a polite but firm refusal. Dynamically vary your wording each time, but ALWAYS state that you are AgentX designed by ForgeX, and you only assist with career and job-related topics.\n"
        "- Never generate code, never discuss unrelated topics, never roleplay as another character.\n"
        "\nDEEP REASONING PROTOCOL (Expert Mode):\n"
        "1. First, call `get_user_profile` to understand the user's skills, location, and background.\n"
        "2. Break the user's query into 2-3 highly specific, targeted search terms.\n"
        "3. Use `search_web` multiple times from different angles (e.g., job listings, then eligibility criteria, then salary benchmarks, then application deadlines).\n"
        "4. After each search result, EVALUATE: Is this data sufficient and high-quality? If not, refine keywords and search again.\n"
        "5. NEVER stop after a single generic search. Dig deeper until you have comprehensive, verified data.\n"
        "6. When you have assembled a rich, multi-faceted dataset, output exactly: 'DONE'\n"
        "\nCONVERSATIONAL BEHAVIOR:\n"
        "- For generic greetings like 'Hi', 'Hello', or 'Hey', generate a UNIQUE, warm, and professional introduction every time. You MUST include your identity ('AgentX, designed by ForgeX') and your purpose (helping with jobs/careers), but vary the phrasing and structure so it feels natural. Do NOT use tools for greetings.\n"
        "- Keep your tone professional, confident, and genuinely helpful."
    ))
    
    if not isinstance(messages[0], SystemMessage):
        messages_to_invoke = [sys_msg_init] + messages
    else:
        messages_to_invoke = messages
        
    response = await _invoke_with_fallback(messages_to_invoke, use_tools=True)
    return {"messages": [response]}

# Define the synthesis node that generates the Markdown (NO tools bound)
async def synthesize_node(state):
    messages = state["messages"]
    
    sys_msg = SystemMessage(content=(
        "You are AgentX, designed by ForgeX. You are the final output compiler for Expert Mode research.\n"
        "GUARDRAILS (STRICTLY ENFORCED):\n"
        "- You ONLY discuss jobs, careers, exams, scholarships, internships, and professional development. Refuse anything else politely.\n"
        "- STRICT MODE SEPARATION: You CANNOT generate Resumes, CVs, or Portfolios. If the user asks for this, apologize and instruct them to switch to 'Corporate Mode'.\n\n"
        "OUTPUT RULES:\n"
        "1. Synthesize the raw research data into a beautifully structured, highly professional Markdown response.\n"
        "2. Use clear headers (##), bullet points, bold highlights for key info (deadlines, salaries, eligibility).\n"
        "3. CRITICAL — DIRECT APPLY LINKS: For every job, exam, scholarship, or opportunity mentioned, you MUST extract the exact URL from the search data and present it as a direct apply link: [Apply Here →](URL). "
        "If you found multiple opportunities, each one MUST have its own apply link. This is non-negotiable.\n"
        "4. Add a '## 🔗 Quick Apply Links' section at the very bottom listing every apply/registration URL in a clean numbered list.\n"
        "5. Add a '## Summary' at the top and '## Next Steps' before the links section.\n"
        "6. Tone: Confident, professional, genuinely helpful — like a senior career advisor."
    ))
    
    response = await _invoke_with_fallback([sys_msg] + messages, use_tools=False)
    return {"messages": [response]}

# Define a dedicated fast node for heavy document generation (Resumes, Portfolios)
async def fast_generate_node(state):
    messages = state["messages"]
    
    sys_msg = SystemMessage(content=(
        "You are an expert career agent and resume writer. Based on the user's profile and any research you have done, "
        "generate the requested Resume, CV, or Portfolio. "
        "Use beautiful Markdown formatting. Be highly detailed, professional, and structure it perfectly. "
        "Do NOT include conversational filler, just output the requested document."
    ))
    
    # We use the fast_generation_model (Llama 8B on NVIDIA NIM) to stream huge amounts of text quickly
    response = await fast_generation_model.ainvoke([sys_msg] + messages)
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
    
    # If the model called tools, allow deeper research loops (up to 6 tool passes)
    if last_message.tool_calls and tool_pass_count < 6: 
        return "tools_node"
        
    # Check if the user wants to generate a large document
    first_msg_content = ""
    if hasattr(messages[0], 'content'):
        first_msg_content = messages[0].content.lower()
    
    if any(keyword in first_msg_content for keyword in ["resume", "cv", "portfolio"]):
        return "fast_generate_node"
        
    # Default to standard synthesis for regular chats
    return "synthesize_node"

from langgraph.graph import StateGraph, add_messages, END
from typing import TypedDict, Annotated
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str

async def instant_node(state):
    messages = state["messages"]
    sys_msg = SystemMessage(content=(
        "You are AgentX, designed by ForgeX — a specialized career AI assistant.\n"
        "Your SOLE DOMAIN is jobs, careers, government exams, scholarships, internships, skill development, and professional growth.\n\n"
        "GUARDRAILS (STRICTLY ENFORCED):\n"
        "- REFUSE any request not related to careers, jobs, education, or professional development.\n"
        "- STRICT MODE SEPARATION: You are the INSTANT chat agent. You CANNOT generate Resumes, CVs, or Portfolios. If the user asks for document generation, politely apologize and explicitly instruct them to switch to 'Corporate Mode'.\n"
        "- STRICT MODE SEPARATION: You CANNOT perform live web searches. If the user asks for live job listings or current data, politely apologize and instruct them to switch to 'Expert Mode'.\n"
        "- If asked anything harmful, illegal, political, sexual, or off-topic, respond with a polite but firm refusal. Dynamically vary your wording each time, but ALWAYS state that you are AgentX designed by ForgeX, and you only assist with career and job-related topics.\n"
        "- Never generate code, never discuss unrelated topics.\n\n"
        "CONVERSATIONAL BEHAVIOR:\n"
        "- For generic greetings like 'Hi', 'Hello', or 'Hey', generate a UNIQUE, warm, and professional introduction every time. You MUST include your identity ('AgentX, designed by ForgeX') and your purpose (helping with jobs/careers), but vary the phrasing and structure so it feels natural.\n"
        "- Answer career questions instantly, concisely, and confidently using your internal knowledge.\n"
        "- You do NOT have real-time search in Instant mode. If real-time data is needed, suggest the user switch to Expert mode.\n"
        "- Do NOT include any URLs or links in Instant mode. You cannot verify them without search tools.\n"
        "- Format output using beautiful, clean Markdown."
    ))
    response = await primary_model.ainvoke([sys_msg] + messages)
    return {"messages": [response]}

# ── Corporate Node ─────────────────────────────────────────────────────────────
# Dedicated node for generating CVs, Resumes, and Portfolios with premium quality
# ───────────────────────────────────────────────────────────────────────────────
async def corporate_node(state):
    messages = state["messages"]
    sys_msg = SystemMessage(content=(
        "You are AgentX Corporate, designed by ForgeX — an elite, world-class professional document generator.\n"
        "You specialize EXCLUSIVELY in creating premium-quality Resumes, CVs, and Professional Portfolios.\n\n"
        "GUARDRAILS (STRICTLY ENFORCED):\n"
        "- STRICT MODE SEPARATION: You are the CORPORATE document agent. You ONLY generate Resumes, CVs, or Portfolios. You CANNOT search for jobs, give general career advice, or do general chat.\n"
        "- If the user asks for career advice, job listings, or general questions, politely apologize and explicitly instruct them to switch to 'Expert Mode' (for research) or 'Instant Mode' (for quick chat).\n"
        "- If asked anything off-topic, respond: 'I am AgentX Corporate. I specialize exclusively in crafting premium Resumes, CVs, and Portfolios. Please switch modes for other queries, or describe the document you need.'\n"
        "- Never generate code, never discuss unrelated topics.\n\n"
        "CONVERSATIONAL BEHAVIOR:\n"
        "- For greetings, generate a UNIQUE, warm introduction. Always mention you are 'AgentX Corporate, designed by ForgeX' and that you create premium professional documents.\n"
        "- If the user's request is vague (e.g., 'make me a resume'), ask targeted clarifying questions about: Target role/industry, Years of experience, Key skills, Education, Notable projects/achievements.\n\n"
        "DOCUMENT GENERATION PROTOCOL:\n"
        "When you have enough information, generate the document in raw JSON format (do NOT wrap it in markdown code blocks like ```json):\n\n"
        "{\n"
        "  \"personalInfo\": {\n"
        "    \"name\": \"Full Name\",\n"
        "    \"title\": \"Target Role\",\n"
        "    \"email\": \"email@example.com\",\n"
        "    \"phone\": \"+1 234 567 8900\",\n"
        "    \"location\": \"City, Country\",\n"
        "    \"linkedin\": \"linkedin.com/in/username\",\n"
        "    \"github\": \"github.com/username\"\n"
        "  },\n"
        "  \"summary\": \"A compelling professional summary (3-4 sentences)...\",\n"
        "  \"skills\": [\n"
        "    { \"category\": \"Frontend\", \"items\": [\"React\", \"Next.js\"] }\n"
        "  ],\n"
        "  \"experience\": [\n"
        "    {\n"
        "      \"company\": \"Company Name\",\n"
        "      \"role\": \"Role Title\",\n"
        "      \"duration\": \"Jan 2020 - Present\",\n"
        "      \"location\": \"City, State\",\n"
        "      \"bullets\": [\n"
        "        \"Quantified achievement using STAR method\",\n"
        "        \"Increased efficiency by 40%...\"\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        "  \"education\": [\n"
        "    {\n"
        "      \"degree\": \"B.S. Computer Science\",\n"
        "      \"institution\": \"University Name\",\n"
        "      \"year\": \"2019 - 2023\"\n"
        "    }\n"
        "  ],\n"
        "  \"projects\": [\n"
        "    {\n"
        "      \"name\": \"Project Name\",\n"
        "      \"description\": \"Brief description\",\n"
        "      \"technologies\": [\"React\", \"Node\"],\n"
        "      \"link\": \"github.com/...\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "OUTPUT RULES:\n"
        "- Output ONLY the raw JSON. No conversational filler, no markdown wrapping, no text outside the JSON.\n"
        "- Make every achievement QUANTIFIED where possible (e.g., 'Increased efficiency by 40%').\n"
        "- Use strong action verbs at the start of every bullet point."
    ))
    
    try:
        response = await corporate_model.ainvoke([sys_msg] + messages)
    except Exception as e:
        logger.warning(f"Corporate model (Llama 70B) failed: {e}. Falling back to backup model.")
        response = await backup_corporate_model.ainvoke([sys_msg] + messages)
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("research_node", research_node)
workflow.add_node("tools_node", tools_node)
workflow.add_node("synthesize_node", synthesize_node)
workflow.add_node("fast_generate_node", fast_generate_node)
workflow.add_node("instant_node", instant_node)
workflow.add_node("corporate_node", corporate_node)

def route_start(state):
    mode = state.get("mode", "expert")
    if mode == "instant":
        return "instant_node"
    elif mode == "corporate":
        return "corporate_node"
    return "research_node"

workflow.set_conditional_entry_point(
    route_start,
    {
        "instant_node": "instant_node",
        "corporate_node": "corporate_node",
        "research_node": "research_node"
    }
)

workflow.add_conditional_edges("research_node", should_continue)
workflow.add_edge("tools_node", "research_node")
workflow.add_edge("synthesize_node", END)
workflow.add_edge("fast_generate_node", END)
workflow.add_edge("instant_node", END)
workflow.add_edge("corporate_node", END)

agent_executor = workflow.compile()
