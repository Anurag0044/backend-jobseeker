
from fastapi import APIRouter, Request, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from agents.whatsapp_agent import process_whatsapp_message
from core.logger import logger

router = APIRouter()

@router.post("/whatsapp")
async def receive_whatsapp_message(
    request: Request,
    From: str = Form(...),
    Body: str = Form(...)
):
    """
    Twilio Webhook for incoming WhatsApp messages.
    - `From` contains the sender's WhatsApp number (e.g., 'whatsapp:+14155238886').
    - `Body` contains the message text.
    """
    logger.info(f"Received WhatsApp message from {From}: {Body}")
    
    # Extract just the phone number (remove the 'whatsapp:' prefix)
    phone_number = From.replace("whatsapp:", "")
    
    # Process the message with our LangChain agent
    agent_response = process_whatsapp_message(Body, phone_number)
    
    logger.info(f"Agent response to {From}: {agent_response}")
    
    # Create the Twilio response
    twiml = MessagingResponse()
    twiml.message(agent_response)
    
    # Return XML as required by Twilio
    return Response(content=twiml.to_xml(), media_type="application/xml")
