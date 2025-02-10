# UrbanWorks Backend API

FastAPI-based backend service for UrbanWorks Architecture, providing AI-powered functionality for chat, proposals, social media, and image generation.

## Features

- AI Chat Assistant
- Proposal Generation
- Social Media Post Creation
- Image Generation
- File Upload Management

## Setup

### Local Development

1. Clone the repository
2. Create `.env.local` file with required environment variables
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the server:
```bash
python main.py
```

### Replit Deployment

1. Create new Repl and import code
2. Add required secrets in Replit's Secrets tab:
   - `FIREBASE_CREDENTIALS` (Firebase service account JSON)
   - `ANTHROPIC_API_KEY`
   - `TAVILY_API_KEY`
   - `REPLICATE_API_TOKEN`
   - Firebase config variables (NEXT_PUBLIC_*)
3. Install dependencies from requirements.txt
4. Run main.py

## API Endpoints

- `/chat` - AI chat interface
- `/proposal` - Proposal generation
- `/social` - Social media content creation
- `/image` - Image generation
- `/upload` - File upload management

## Environment Variables

See `.env.example` for required environment variables.

## Tech Stack

- FastAPI
- Firebase Admin SDK
- Anthropic Claude
- Tavily
- Replicate
- LangChain 