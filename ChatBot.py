from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool

# Initialize FastAPI app
app = FastAPI()

# Set up Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing Gemini API key. Set it as an environment variable.")

# Initialize LangChain models
chat_model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)
llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)  # Free-tier model

# Store chat history per user (for simple cases, use in-memory storage)
user_sessions = {}

# Define a tool
def simple_tool(input_text: str):
    """A simple tool that processes input text and returns a modified response."""
    return f"Processed: {input_text}"

simple_tool_instance = Tool(
    name="SimpleTool",
    func=simple_tool,
    description="A tool that processes input text."
)

# Function to get or create user session
def get_user_session(user_id: str):
    """Retrieves the user session or creates a new one with memory."""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "agent": None
        }
        # Initialize agent with memory
        user_sessions[user_id]["agent"] = initialize_agent(
            tools=[simple_tool_instance],
            llm=llm_model,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=user_sessions[user_id]["memory"],
            verbose=True
        )
    return user_sessions[user_id]

# Request model
class ChatRequest(BaseModel):
    user_id: str  # Unique ID for each user session
    message: str

# API Endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    """Handles user chat requests and maintains conversation memory."""
    try: 
        session = get_user_session(request.user_id)
        response = session["agent"].run(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))