"""
Services for image generation.
"""
from typing import Optional, Dict, Any
import base64
import mimetypes
from fastapi import UploadFile, HTTPException
from datetime import datetime
import io
import httpx
import json

from ..config import (
    replicate_client, 
    claude_client, 
    bucket, 
    logger, 
    API_BASE_URL,
    CENTRAL_TZ
)

async def improve_prompt(prompt: str) -> str:
    """
    Improve an image generation prompt using Claude AI.
    
    Args:
        prompt: The original prompt to improve
        
    Returns:
        An improved version of the prompt
    """
    system_prompt = """You are an expert at writing prompts for FLUX image generation. 
    Your task is to improve the given prompt to create more detailed and visually appealing images.
    Focus on adding details about style, lighting, composition, and mood while maintaining the original intent.
    Return only the improved prompt without any explanations."""

    try:
        response = claude_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=200,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Please improve this image generation prompt: {prompt}"
                }
            ]
        )
        
        return response.content[0].text
    except Exception as e:
        logger.error(f"Error improving prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to improve prompt: {str(e)}")

async def generate_image(
    prompt: str,
    user_id: str,
    token: str,
    aspect_ratio: str = "1:1",
    raw: bool = False,
    seed: Optional[int] = None,
    image_prompt: Optional[UploadFile] = None,
    image_prompt_strength: float = 0.1,
    output_format: str = "jpg",
    folder: str = "Uncategorized"
) -> Dict[str, Any]:
    """
    Generate an image using Replicate's Flux model.
    
    Args:
        prompt: The text prompt for image generation
        user_id: The user ID
        token: Authentication token
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
        logger.info(f"Starting image generation with prompt: {prompt}")
        
        # Sanitize folder name
        folder = "".join(c for c in folder if c.isalnum() or c in (' ', '-', '_')).strip()
        if not folder:
            folder = "Uncategorized"
            
        # Handle image prompt upload if provided
        image_prompt_url = None
        image_prompt_data = None
        if image_prompt:
            logger.info(f"Processing image prompt: {image_prompt.filename}")
            # Read the image data directly from the uploaded file
            image_prompt_data = await image_prompt.read()
            logger.info(f"Successfully read image prompt data of size: {len(image_prompt_data)} bytes")

            # Convert image data to base64 data URI
            mime_type = mimetypes.guess_type(image_prompt.filename)[0] or 'image/jpeg'
            base64_image = base64.b64encode(image_prompt_data).decode('utf-8')
            image_prompt_url = f"data:{mime_type};base64,{base64_image}"
            logger.info("Successfully converted image to base64 data URI")

        # Prepare generation parameters
        generation_params = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "raw": raw,
            "safety_tolerance": 6,
            "output_format": output_format
        }

        if seed is not None:
            generation_params["seed"] = seed

        if image_prompt_url:
            generation_params["image_prompt"] = image_prompt_url
            generation_params["image_prompt_strength"] = image_prompt_strength

        logger.info(f"Generation parameters: {generation_params}")

        # Generate image using Replicate with Flux model
        try:
            logger.info("Starting Replicate API call...")
            output = replicate_client.run(
                "black-forest-labs/flux-1.1-pro-ultra",
                input=generation_params
            )

            # Ensure output is a string (URL)
            if isinstance(output, list) and len(output) > 0:
                output_url = output[0]
            else:
                output_url = str(output)
            logger.info(f"Generated image URL from Replicate: {output_url}")

            # Download the generated image
            import requests
            logger.info("Downloading generated image...")
            response = requests.get(output_url)
            logger.info(f"Download response status: {response.status_code}")
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to download generated image")
                
            # Generate a unique filename using timestamp
            timestamp = datetime.now(CENTRAL_TZ).strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{prompt[:30]}.png"
            logger.info(f"Preparing to upload with filename: {filename}")

            # Create file-like object for the image
            files = {
                'file': (
                    filename,
                    io.BytesIO(response.content),
                    f'image/{output_format}'
                )
            }
            logger.info(f"Uploading generated image with content type: image/{output_format}")

            # Upload to Firebase using the upload endpoint
            async with httpx.AsyncClient() as client:
                logger.info("Making upload request to storage endpoint...")

                # Ensure the storage path includes the user's directory
                storage_path = f"users/{user_id}/images/{folder}/{filename}"

                # Add metadata fields
                data = {
                    'prompt': prompt,
                    'userId': user_id,
                    'storage_path': storage_path,
                    'folder': folder,
                    'parameters': json.dumps({
                        "prompt": prompt,
                        "aspect_ratio": aspect_ratio,
                        "raw": raw,
                        "safety_tolerance": 6,
                        "output_format": output_format,
                        "seed": seed if seed is not None else "random",
                        "image_prompt_strength": image_prompt_strength if image_prompt_url else None
                    })
                }

                # Make the request with both files and form data
                upload_response = await client.post(
                    f'{API_BASE_URL}/upload',
                    files=files,
                    data=data,
                    headers={
                        'Authorization': f'Bearer {token}',
                        'X-Skip-Redirect': 'true'
                    },
                    follow_redirects=True,
                    timeout=30.0
                )

                if upload_response.status_code not in [200, 201]:
                    error_text = await upload_response.aread()
                    logger.error(f"Upload failed with status {upload_response.status_code}")
                    logger.error(f"Response headers: {upload_response.headers}")
                    logger.error(f"Response body: {error_text}")
                    raise HTTPException(
                        status_code=upload_response.status_code,
                        detail=f"Failed to upload generated image: {error_text.decode('utf-8')}"
                    )

                try:
                    response_text = await upload_response.aread()
                    upload_data = json.loads(response_text.decode('utf-8'))
                    firebase_url = upload_data['url']
                    storage_path = upload_data['storage_path']
                except Exception as e:
                    logger.error(f"Error parsing upload response: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to parse upload response"
                    )

        except Exception as e:
            logger.error(f"Error during image generation/upload: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

        # Create a clean, serializable version of the parameters
        clean_params = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "raw": raw,
            "safety_tolerance": 6,
            "output_format": output_format,
            "seed": seed if seed is not None else "random",
            "image_prompt_url": image_prompt_url,
            "image_prompt_strength": image_prompt_strength if image_prompt_url else None,
            "storage_path": storage_path
        }

        logger.info("Preparing response data...")
        # Return both the original Replicate URL and the Firebase storage URL
        image_data = {
            "url": firebase_url,  # Use Firebase URL as primary
            "original_url": output_url,  # Keep original URL as backup
            "prompt": prompt,
            "timestamp": timestamp,
            "parameters": clean_params,
            "storage_path": storage_path
        }
        logger.info("Generation process completed successfully")
        return image_data

    except Exception as e:
        logger.error(f"Error in generate_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 