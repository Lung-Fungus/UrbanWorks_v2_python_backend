from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from utils.auth_middleware import firebase_auth
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.settings import ModelSettings
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi.middleware.cors import CORSMiddleware
import re
from utils.config import get_api_keys, get_firebase_credentials, initialize_environment
import logging
import pytz  # Add pytz for timezone handling
from utils.prompts import get_social_media_system_prompt  # Import the social media system prompt

# Initialize environment
initialize_environment()

# Get API keys
api_keys = get_api_keys()
ANTHROPIC_API_KEY = api_keys["ANTHROPIC_API_KEY"]

# Initialize Firebase Admin
try:
    # Since main.py should have already initialized Firebase, we just need to get the client
    db = firestore.client()
except Exception as e:
    # If an error occurs, try initializing Firebase ourselves
    if not firebase_admin._apps:
        try:
            # Initialize Firebase without storage bucket to prevent automatic bucket creation
            cred = credentials.Certificate(get_firebase_credentials())
            firebase_admin.initialize_app(cred, {
                'storageBucket': None  # Disable automatic Storage bucket initialization
            })
            db = firestore.client()
        except Exception as e:
            print(f"Error initializing Firebase in social_media_backend: {e}")
            # Create a placeholder db that will be replaced when Firebase is available
            db = None

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Add console handler
    ]
)
logger = logging.getLogger(__name__)

# Define Central Time Zone
central_tz = pytz.timezone('US/Central')

# ========== Pydantic Models ==========
class SocialMediaPost(BaseModel):
    """Represents a single social media post with platform-specific content"""
    platform: str = Field(..., description="Social media platform (X/Instagram/Facebook/LinkedIn)")
    date: str = Field(..., description="Scheduled date in YYYY-MM-DD format")
    content: str = Field(..., description="Post content with emojis and hashtags")

class SocialMediaResponse(BaseModel):
    """Structured response from the social media AI agent"""
    analysis: str = Field(..., description="Strategic analysis of the request and post effectiveness")
    posts: List[SocialMediaPost] = Field(..., description="List of generated social media posts")

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_display_name: str
    user_id: str

class ChatResponse(BaseModel):
    content: SocialMediaResponse
    conversation_id: Optional[str] = None

# ========== Dependency Injection ==========
class SocialDependencies:
    """Dependency container for social media agent"""
    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id
        self.db = db

# Initialize the Anthropic model
model = AnthropicModel('claude-3-7-sonnet-20250219', api_key=ANTHROPIC_API_KEY)

# ========== AI Agent Setup ==========
social_agent = Agent(
    model,
    deps_type=SocialDependencies,
    result_type=SocialMediaResponse,
    model_settings=ModelSettings(max_tokens=8000),
    system_prompt=get_social_media_system_prompt()
)

# Update the validation function with logging
async def validate_and_parse_result(result: Any) -> SocialMediaResponse:
    """Validates and parses the AI response format"""
    try:
        # logging.info(f"Starting validation of result type: {type(result)}")
        # logging.info(f"Raw result content: {result}")

        if isinstance(result, SocialMediaResponse):
            # logging.info("Result is already a SocialMediaResponse")
            return result

        content = str(result)
        # logging.info(f"Converted content: {content}")

        analysis_match = re.search(r'<analysis>(.*?)</analysis>', content, re.DOTALL)
        posts_match = re.search(r'<posts>(.*?)</posts>', content, re.DOTALL)

        # logging.info(f"Analysis match found: {bool(analysis_match)}")
        # logging.info(f"Posts match found: {bool(posts_match)}")

        if not analysis_match or not posts_match:
            raise ValueError("Response must include both <analysis> and <posts> sections")

        analysis = analysis_match.group(1).strip()
        posts_text = posts_match.group(1).strip()

        # logging.info(f"Extracted analysis: {analysis[:100]}...")
        # logging.info(f"Extracted posts text: {posts_text[:100]}...")

        posts = []
        for i, post_block in enumerate(posts_text.split('---')):
            if not post_block.strip():
                continue

            # logging.info(f"Processing post block {i}: {post_block}")

            platform_match = re.search(r'Platform:\s*(.+)', post_block)
            date_match = re.search(r'Date:\s*(.+)', post_block)
            content_match = re.search(r'Content:\s*(.+)', post_block, re.DOTALL)

            # logging.info(f"Post {i} matches - Platform: {bool(platform_match)}, Date: {bool(date_match)}, Content: {bool(content_match)}")

            if not all([platform_match, date_match, content_match]):
                raise ValueError(f"Post {i} missing required fields")

            post = SocialMediaPost(
                platform=platform_match.group(1).strip(),
                date=date_match.group(1).strip(),
                content=content_match.group(1).strip()
            )
            # logging.info(f"Created post {i}: {post.model_dump_json()}")
            posts.append(post)

        if not posts:
            raise ValueError("At least one post must be generated")

        response = SocialMediaResponse(
            analysis=analysis,
            posts=posts
        )
        # logging.info("Successfully created SocialMediaResponse")
        return response

    except Exception as e:
        # logging.error(f"Validation error: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to validate response: {str(e)}")

@social_agent.tool
async def get_conversation_history(ctx: RunContext[SocialDependencies]) -> List[Dict]:
    """Retrieves conversation history from Firestore"""
    try:
        if not ctx.deps.conversation_id:
            return []

        messages_ref = ctx.deps.db.collection("socialmedia").document(
            ctx.deps.conversation_id).collection("messages")
        docs = messages_ref.order_by("timestamp").stream()

        return [
            {
                "role": doc.get("role"),
                "content": doc.get("content"),
                "timestamp": doc.get("timestamp").isoformat() if doc.get("timestamp") else None
            } for doc in docs
        ]
    except Exception as e:
        # logging.error(f"Error fetching conversation history: {str(e)}")
        return []

# ========== API Endpoints ==========
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, user_data: dict = Depends(firebase_auth)):
    """Main chat endpoint for social media generation"""
    try:
        # Verify user ID matches authenticated user
        if request.user_id != user_data['uid']:
            raise HTTPException(status_code=403, detail="User ID mismatch")

        # Initialize dependencies
        deps = SocialDependencies(request.conversation_id)

        # Create conversation if needed
        if not deps.conversation_id:
            doc_ref = db.collection("socialmedia").document()
            deps.conversation_id = doc_ref.id
            doc_ref.set({
                "createdAt": datetime.now(central_tz),
                "lastMessageTimestamp": datetime.now(central_tz),
                "messageCount": 0,
                "title": f"Social Media Session - {datetime.now(central_tz).strftime('%b %d')}",
                "summary": "New social media strategy session",
                "userid": user_data['uid']  # Use authenticated user ID
            })
        else:
            # Verify the conversation exists and belongs to the user
            doc_ref = db.collection("socialmedia").document(deps.conversation_id)
            doc = doc_ref.get()
            if not doc.exists:
                raise HTTPException(status_code=404, detail="Conversation not found")
            conversation_owner = doc.get("userId") or doc.get("userid")
            if conversation_owner != user_data['uid']:
                raise HTTPException(status_code=403, detail="Not authorized to access this conversation")

        # Run the AI agent with user info included in the message
        formatted_message = f"""User: {request.user_display_name}

Request: {request.message}"""

        # Get the response
        result = await social_agent.run(
            formatted_message,
            deps=deps
        )

        # Validate the result
        validated_result = await validate_and_parse_result(result.data)

        # Save conversation history
        batch = db.batch()
        conv_ref = db.collection("socialmedia").document(deps.conversation_id)

        # Save user message
        user_msg_ref = conv_ref.collection("messages").document()
        batch.set(user_msg_ref, {
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now(central_tz)
        })

        # Save AI response
        ai_msg_ref = conv_ref.collection("messages").document()
        batch.set(ai_msg_ref, {
            "role": "assistant",
            "content": result.data.model_dump_json(),
            "timestamp": datetime.now(central_tz)
        })

        # Update conversation metadata
        batch.update(conv_ref, {
            "lastMessageTimestamp": datetime.now(central_tz),
            "messageCount": firestore.Increment(2)
        })

        batch.commit()

        return {
            "content": validated_result,
            "conversation_id": deps.conversation_id
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Please run main.py to start the server")
