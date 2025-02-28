"""
Routes for image generation.
"""
from fastapi import APIRouter, Depends, Form, File, UploadFile, HTTPException, Request, Query
from typing import Optional
from utils.auth_middleware import firebase_auth

from ..services.generation import improve_prompt, generate_image

# Create router
router = APIRouter(tags=["image_generation"])

@router.post("/improve-prompt")
async def improve_prompt_endpoint(prompt: str = Form(...)):
    """
    Improve an image generation prompt using AI.
    
    Args:
        prompt: The original prompt to improve
        
    Returns:
        An improved version of the prompt
    """
    try:
        improved_prompt = await improve_prompt(prompt)
        return {"improved_prompt": improved_prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def generate_image_endpoint(
    request: Request, 
    user_data: dict = Depends(firebase_auth),
    prompt: str = Form(...),
    aspect_ratio: str = Form("1:1"),
    raw: bool = Form(False),
    seed: Optional[int] = Form(None),
    image_prompt: Optional[UploadFile] = File(None),
    image_prompt_strength: float = Form(0.1),
    output_format: str = Form("jpg"),
    folder: str = Form("Uncategorized")
):
    """
    Generate an image using AI.
    
    Args:
        request: The FastAPI request object
        user_data: User data from authentication
        prompt: The text prompt for image generation
        aspect_ratio: Image aspect ratio (default: "1:1")
        raw: Whether to use raw mode (default: False)
        seed: Random seed for reproducibility (default: None)
        image_prompt: Optional image to use as a reference
        image_prompt_strength: Strength of the image prompt (default: 0.1)
        output_format: Output image format (default: "jpg")
        folder: Folder to store the image in (default: "Uncategorized")
        
    Returns:
        Dictionary with image data including URLs and metadata
    """
    try:
        # Extract user_id from user_data and validate
        user_id = user_data.get('uid') or user_data.get('user_id')
        if not user_id:
            raise HTTPException(status_code=401, detail="No valid user id found")
            
        # Extract token for authentication
        token = user_data.get('token') or user_data.get('uid') or user_data.get('user_id')
        if not token:
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

        if not token:
            raise HTTPException(status_code=401, detail="No valid authentication token found")
            
        # Call the service function
        return await generate_image(
            prompt=prompt,
            user_id=user_id,
            token=token,
            aspect_ratio=aspect_ratio,
            raw=raw,
            seed=seed,
            image_prompt=image_prompt,
            image_prompt_strength=image_prompt_strength,
            output_format=output_format,
            folder=folder
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 