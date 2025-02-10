from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from auth_middleware import firebase_auth
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.settings import ModelSettings
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi.middleware.cors import CORSMiddleware
import re
from config import get_api_keys, get_firebase_credentials, initialize_environment

# Initialize environment
initialize_environment()

# Get API keys
api_keys = get_api_keys()
ANTHROPIC_API_KEY = api_keys["ANTHROPIC_API_KEY"]

# Initialize Firebase Admin
if not firebase_admin._apps:
    cred = credentials.Certificate(get_firebase_credentials())
    firebase_admin.initialize_app(cred)

db = firestore.client()
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update the logging configuration
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

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
model = AnthropicModel('claude-3-5-sonnet-20241022', api_key=ANTHROPIC_API_KEY)

# ========== AI Agent Setup ==========
social_agent = Agent(
    model,
    deps_type=SocialDependencies,
    result_type=SocialMediaResponse,
    model_settings=ModelSettings(max_tokens=8000),
    system_prompt="""You are Clarke, the social media manager for UrbanWorks - an internationally recognized Chicago architectural firm. 
    Create posts that balance technical expertise with community engagement.

    YOU MUST ALWAYS RESPOND IN THIS EXACT FORMAT:
    
    <analysis>
    [Write your strategic analysis here explaining your post strategy]
    </analysis>
    
    <posts>
    Platform: [Must be one of: X, Instagram, Facebook, or LinkedIn]
    Date: [Must be in YYYY-MM-DD format]
    Content: [Write the post content here]
    ---
    [Repeat the above format for each additional post]
    </posts>

    IMPORTANT: Both <analysis> and <posts> sections are REQUIRED in every response.
    Each post MUST include Platform, Date, and Content fields.
    At least one post is required in every response.

    TONE & VOICE:
    - Professional yet accessible: authoritative but warm
    - Proud but not boastful: emphasize collaborative achievements
    - Active voice/present tense for immediacy
    - Technical concepts made accessible

    CONTENT PRIORITIES:
    1. Community Impact:
    - Highlight service to underserved communities
    - Show public input and engagement
    - Social/environmental responsibility
    
    2. Professional Excellence:
    - Share awards/recognitions with humility
    - Acknowledge team and partners
    - Demonstrate urban planning thought leadership
    
    3. Diversity & Inclusion:
    - Reflect MWBE identity
    - Showcase diverse project types
    - Emphasize inclusive design approaches

    KEY THEMES TO INCORPORATE:
    Sustainable Design | Community Engagement | Urban Innovation
    Social Responsibility | Technical Expertise | Collaborative Approach
    Civic Commitment | Cultural Awareness

    WRITING RULES:
    - Start with clear, direct news statements
    - Use specific details (sizes, dates, numbers)
    - Express genuine gratitude for partners/awards
    - Concise sentences with strategic line breaks
    - Professional abbreviations when appropriate

    AVOID:
    - Technical jargon without context
    - Self-congratulatory tone without partners
    - Vague statements about impact
    - Exclusive language
    - Overly casual expressions

 """
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
        # logging.info(f"Received chat request from user: {request.user_display_name}")
        
        # Initialize dependencies
        deps = SocialDependencies(request.conversation_id)
        
        # Create conversation if needed
        if not deps.conversation_id:
            doc_ref = db.collection("socialmedia").document()
            deps.conversation_id = doc_ref.id
            doc_ref.set({
                "createdAt": datetime.now(),
                "lastMessageTimestamp": datetime.now(),
                "messageCount": 0,
                "title": f"Social Media Session - {datetime.now().strftime('%b %d')}",
                "summary": "New social media strategy session",
                "userId": request.user_id
            })
        else:
            # Verify the conversation exists and belongs to the user
            doc_ref = db.collection("socialmedia").document(deps.conversation_id)
            doc = doc_ref.get()
            if not doc.exists:
                raise HTTPException(status_code=404, detail="Conversation not found")
            if doc.get("userId") != request.user_id:
                raise HTTPException(status_code=403, detail="Not authorized to access this conversation")

        # Run the AI agent with user info included in the message
        formatted_message = f"""User: {request.user_display_name}
        
Request: {request.message}"""

        # logging.info("Calling AI agent")
        try:
            # First log the raw message we're sending
            # logging.info(f"Sending message to AI: {formatted_message}")
            
            # Get the response
            result = await social_agent.run(
                formatted_message,
                deps=deps
            )

            # Immediately log the complete raw response
            # logging.info("\n=== RAW AI RESPONSE START ===")
            # logging.info(f"Complete raw response object: {result}")
            # if hasattr(result, 'content'):
            #     logging.info(f"Response content: {result.content}")
            # if hasattr(result, 'messages'):
            #     logging.info(f"Response messages: {result.messages}")
            # if hasattr(result, 'data'):
            #     logging.info(f"Response data: {result.data}")
            # logging.info("=== RAW AI RESPONSE END ===\n")

            # Continue with validation...
            # logging.info("Starting validation")
            validated_result = await validate_and_parse_result(result.data)
            # logging.info("Validation successful")

        except Exception as e:
            # logging.error(f"Error during AI response or validation: {str(e)}")
            # logging.error("Full error:", exc_info=True)
            raise

        # Save conversation history
        batch = db.batch()
        conv_ref = db.collection("socialmedia").document(deps.conversation_id)
        
        # Save user message
        user_msg_ref = conv_ref.collection("messages").document()
        batch.set(user_msg_ref, {
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now()
        })
        
        # Save AI response
        ai_msg_ref = conv_ref.collection("messages").document()
        batch.set(ai_msg_ref, {
            "role": "assistant",
            "content": result.data.model_dump_json(),
            "timestamp": datetime.now()
        })
        
        # Update conversation metadata
        batch.update(conv_ref, {
            "lastMessageTimestamp": datetime.now(),
            "messageCount": firestore.Increment(2)
        })
        
        batch.commit()

        return {
            "content": validated_result,
            "conversation_id": deps.conversation_id
        }

    except ValueError as e:
        # logging.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # logging.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Please run main.py to start the server")
