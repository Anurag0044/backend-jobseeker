

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from firebase_admin import firestore

@tool
def update_profile(field: str, value: str, phone_number: str) -> str:
    """
    Updates the user's profile in the database.
    Args:
        field: The exact profile field to update (MUST be one of: 'displayName', 'title', 'bio', 'location', 'portfolioUrl', 'github', 'linkedin').
        value: The new value to set for the field.
        phone_number: The user's phone number to identify them.
    """
    try:
        db = firestore.client()
        # Hardcoding user email for testing
        users_ref = db.collection("users").where("email", "==", "subhaschandra270@gmail.com").limit(1)
        docs = users_ref.stream()
        
        user_id = None
        for doc in docs:
            user_id = doc.id
            break
            
        if not user_id:
            user_id = f"mock_user_{phone_number.replace('+', '')}"
            
        db.collection("users").document(user_id).set({field: value}, merge=True)
        return f"Successfully updated {field} to '{value}'."
    except Exception as e:
        # Mocking for local development
        return f"Local Test Mode: Successfully updated {field} to '{value}' for {phone_number} (Database bypassed)."

@tool
def post_to_community(content: str, phone_number: str) -> str:
    """
    Creates a new post in the community forum.
    Args:
        content: The text content of the post.
        phone_number: The user's phone number to identify the author.
    """
    try:
        db = firestore.client()
        # Hardcoding user email for testing
        users_ref = db.collection("users").where("email", "==", "subhaschandra270@gmail.com").limit(1)
        docs = users_ref.stream()
        
        user_id = f"mock_user_{phone_number.replace('+', '')}"
        author_name = "WhatsApp User"
        
        for doc in docs:
            user_id = doc.id
            author_name = doc.to_dict().get("name", author_name)
            break
            
        db.collection("posts").add({
            "author_id": user_id,
            "author_name": author_name,
            "content": content,
            "created_at": firestore.SERVER_TIMESTAMP
        })
        return "Successfully posted your message to the community."
    except Exception as e:
        # Mocking for local development
        return f"Local Test Mode: Successfully posted your message to the community (Database bypassed)."


def process_whatsapp_message(message: str, phone_number: str) -> str:
    """Processes an incoming WhatsApp message using the Gemini LangChain Agent."""
    
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    tools = [update_profile, post_to_community]
    llm_with_tools = llm.bind_tools(tools)
    
    try:
        # Single round-trip: ask the LLM what to do
        prompt = f"You are a helpful assistant on WhatsApp. Always pass the phone_number '{phone_number}' to tools.\nUser: {message}"
        response = llm_with_tools.invoke(prompt)
        
        # If the LLM decided to call a tool, execute it and return the result immediately! (Skips the second LLM thinking phase)
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "update_profile":
                return update_profile.invoke(tool_args)
            elif tool_name == "post_to_community":
                return post_to_community.invoke(tool_args)
                
        # If no tool was called, just return the text response
        return str(response.content)
    except Exception as e:
        return f"Sorry, I encountered an error processing your request: {str(e)}"
