import json
import asyncio
from fastapi import APIRouter, Request, Form, Response, Body
from sse_starlette.sse import EventSourceResponse
from core.logger import logger
from agents.orchestrator import agent_executor
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    phone_number: str = "mock_user"

@router.post("/assistant/stream")
async def stream_assistant(request: ChatRequest):
    """
    Streams LangGraph agent thoughts and actions using Server-Sent Events (SSE).
    """
    async def event_generator():
        try:
            # Yield initial thinking status
            yield {
                "event": "message",
                "data": json.dumps({"type": "thinking", "status": "Starting thought process..."})
            }
            
            initial_state = {
                "messages": [HumanMessage(content=request.message)]
            }
            
            # Use LangGraph's astream_events to get real-time hooks
            logger.info("Starting astream_events loop")
            async for event in agent_executor.astream_events(initial_state, version="v2"):
                kind = event["event"]
                name = event["name"]
                logger.info(f"Yielded event: {kind} | {name}")
                
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    logger.info(f"Chunk content: {chunk.content}, tool_calls: {getattr(chunk, 'tool_calls', None)}")
                    if chunk.content:
                        yield {
                            "event": "message",
                            "data": json.dumps({"type": "token", "content": chunk.content})
                        }
                        
                # Tool execution started
                elif kind == "on_tool_start":
                    inputs = event["data"].get("input", {})
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "tool_start", 
                            "tool": name, 
                            "input": inputs
                        })
                    }
                    
                # Tool execution finished
                elif kind == "on_tool_end":
                    output = event["data"].get("output", "")
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "tool_end", 
                            "tool": name
                        })
                    }
            
            # Thought process complete
            yield {
                "event": "message",
                "data": json.dumps({"type": "done", "status": "Complete"})
            }
            
        except Exception as e:
            logger.error(f"Error in assistant stream: {str(e)}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())
