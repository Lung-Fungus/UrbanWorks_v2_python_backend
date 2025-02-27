from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Query, Response
from fastapi.middleware.cors import CORSMiddleware
import replicate
import anthropic
from typing import Optional
import firebase_admin
from firebase_admin import storage
from datetime import datetime, timedelta
import httpx
import json
from config import get_api_keys, initialize_environment, get_firebase_config
from auth_middleware import firebase_auth
from fastapi import Depends
import os
import base64
import mimetypes
import logging
import pytz

# Initialize environment
initialize_environment()

API_BASE_URL = os.getenv('API_BASE_URL', 'https://urbanworks-v2-pythonbackend.replit.app')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://urbanworks-v2.web.app",
        "https://urbanworks-v2.firebaseapp.com",
        "https://urbanworks-v2-pythonbackend.replit.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize Firebase Admin if not already initialized
if not firebase_admin._apps:
    from config import initialize_firebase
    initialize_firebase()

# Get Firebase config for bucket name
firebase_config = get_firebase_config()
bucket = storage.bucket(firebase_config["storageBucket"])  # Explicitly specify bucket name
print(f"Firebase initialized successfully with bucket: {bucket.name}")

# Test bucket existence
if not bucket.exists():
    print("Warning: Bucket does not exist. Please ensure it's created in the Firebase Console")
else:
    print("Bucket exists and is accessible")

# Get API keys
api_keys = get_api_keys()

# Initialize clients
replicate_client = replicate.Client(api_token=api_keys["REPLICATE_API_TOKEN"])
claude_client = anthropic.Anthropic(api_key=api_keys["ANTHROPIC_API_KEY"])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Add console handler
    ]
)
logger = logging.getLogger(__name__)

# Define Central Time Zone
central_tz = pytz.timezone('US/Central')

async def improve_prompt(prompt: str) -> str:
    system_prompt = """You are an expert at writing prompts for FLUX image generation. 
    Your task is to improve the given prompt to create more detailed and visually appealing images.
    Focus on adding details about style, lighting, composition, and mood while maintaining the original intent.
    Return only the improved prompt without any explanations."""

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

@app.post("/improve-prompt")
async def improve_prompt_endpoint(prompt: str = Form(...)):
    try:
        improved_prompt = await improve_prompt(prompt)
        return {"improved_prompt": improved_prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_image(request: Request, user_data: dict = Depends(firebase_auth),
    prompt: str = Form(...),
    aspect_ratio: str = Form("1:1"),
    raw: bool = Form(False),
    seed: Optional[int] = Form(None),
    image_prompt: Optional[UploadFile] = File(None),
    image_prompt_strength: float = Form(0.1),
    output_format: str = Form("jpg"),
    folder: str = Form("Uncategorized")  # Add folder parameter
):
    try:
        print(f"Starting image generation with prompt: {prompt}")

        # Extract user_id from user_data and validate
        user_id = user_data.get('uid') or user_data.get('user_id')
        if not user_id:
            raise HTTPException(status_code=401, detail="No valid user id found")
        print(f"Processing request for user_id: {user_id}")

        # Sanitize folder name
        folder = "".join(c for c in folder if c.isalnum() or c in (' ', '-', '_')).strip()
        if not folder:
            folder = "Uncategorized"

        # Extract token for authentication
        token = user_data.get('token') or user_data.get('uid') or user_data.get('user_id')
        if not token:
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

        if not token:
            raise HTTPException(status_code=401, detail="No valid authentication token found")

        # Handle image prompt upload if provided
        image_prompt_url = None
        image_prompt_data = None
        if image_prompt:
            print(f"Processing image prompt: {image_prompt.filename}")
            # Read the image data directly from the uploaded file
            image_prompt_data = await image_prompt.read()
            print(f"Successfully read image prompt data of size: {len(image_prompt_data)} bytes")

            # Convert image data to base64 data URI
            mime_type = mimetypes.guess_type(image_prompt.filename)[0] or 'image/jpeg'
            base64_image = base64.b64encode(image_prompt_data).decode('utf-8')
            image_prompt_url = f"data:{mime_type};base64,{base64_image}"
            print("Successfully converted image to base64 data URI")

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

        print(f"Generation parameters: {generation_params}")

        # Generate image using Replicate with Flux model
        try:
            print("Starting Replicate API call...")
            output = replicate_client.run(
                "black-forest-labs/flux-1.1-pro-ultra",
                input=generation_params
            )

            # Ensure output is a string (URL)
            if isinstance(output, list) and len(output) > 0:
                output_url = output[0]
            else:
                output_url = str(output)
            print(f"Generated image URL from Replicate: {output_url}")

            # Download the generated image and upload using the upload endpoint
            import requests
            import io
            print("Downloading generated image...")
            response = requests.get(output_url)
            print(f"Download response status: {response.status_code}")
            if response.status_code == 200:
                # Generate a unique filename using timestamp
                timestamp = datetime.now(central_tz).strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{prompt[:30]}.png"
                print(f"Preparing to upload with filename: {filename}")

                # Create file-like object for the image
                files = {
                    'file': (
                        filename,
                        io.BytesIO(response.content),
                        f'image/{output_format}'
                    )
                }
                print(f"Uploading generated image with content type: image/{output_format}")

                # Upload to Firebase using the upload endpoint
                async with httpx.AsyncClient() as client:
                    print("Making upload request to storage endpoint...")

                    # Create multipart form data
                    files = {
                        'file': (
                            filename,
                            io.BytesIO(response.content),
                            f'image/{output_format}'
                        )
                    }

                    # Ensure the storage path includes the user's directory
                    storage_path = f"users/{user_id}/images/{folder}/{filename}"

                    # Add metadata fields
                    data = {
                        'prompt': prompt,
                        'userId': user_id,
                        'storage_path': storage_path,
                        'folder': folder,  # Add folder to metadata
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
                            'X-Skip-Redirect': 'true'  # Add header to prevent redirect
                        },
                        follow_redirects=True,
                        timeout=30.0  # Add timeout
                    )

                    if upload_response.status_code not in [200, 201]:
                        error_text = await upload_response.aread()
                        print(f"Upload failed with status {upload_response.status_code}")
                        print(f"Response headers: {upload_response.headers}")
                        print(f"Response body: {error_text}")
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
                        print(f"Error parsing upload response: {str(e)}")
                        raise HTTPException(
                            status_code=500,
                            detail="Failed to parse upload response"
                        )

        except Exception as e:
            print(f"Error during image generation/upload: {str(e)}")
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

        print("Preparing response data...")
        # Return both the original Replicate URL and the Firebase storage URL
        image_data = {
            "url": firebase_url,  # Use Firebase URL as primary
            "original_url": output_url,  # Keep original URL as backup
            "prompt": prompt,
            "timestamp": timestamp,
            "parameters": clean_params,
            "storage_path": storage_path
        }
        print("Generation process completed successfully")
        return image_data

    except Exception as e:
        print(f"Error in generate_image endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-image")
async def delete_image(storage_path: str):
    try:
        print(f"Attempting to delete image at path: {storage_path}")

        if not storage_path or storage_path.strip() == "":
            raise HTTPException(status_code=400, detail="Invalid storage path: Path cannot be empty")

        # Get the blob from the bucket and delete it directly
        blob = bucket.blob(storage_path)
        print(f"Checking if blob exists at path: {storage_path}")

        if not blob.exists():
            print(f"Blob not found at path: {storage_path}")
            raise HTTPException(status_code=404, detail=f"Image not found at path: {storage_path}")

        # Delete the blob
        print(f"Blob found, proceeding with deletion")
        blob.delete()

        print(f"Image at path {storage_path} deleted successfully")
        return {"success": True, "message": "Image deleted successfully", "path": storage_path}

    except HTTPException as he:
        print(f"HTTP error deleting image: {he.detail}")
        raise he
    except Exception as e:
        error_message = str(e)
        print(f"Error deleting image: {error_message}")
        # Include more context in the error
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete image at {storage_path}: {error_message}"
        )

@app.get("/get-image")
async def get_image(storage_path: str = Query(...), user_data: dict = Depends(firebase_auth)):
    try:
        # Verify user has access to this image
        user_id = user_data.get('uid')
        if not user_id:
            raise HTTPException(status_code=401, detail="No valid user id found")

        # Verify the image belongs to the user
        if not storage_path.startswith(f"users/{user_id}/"):
            raise HTTPException(status_code=403, detail="Access denied")

        # Get the blob
        blob = bucket.blob(storage_path)
        if not blob.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        # Generate a signed URL that expires in 1 hour
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=1),
            method="GET"
        )

        return {
            "url": signed_url,
            "storage_path": storage_path,
            "expires_in": 3600  # seconds
        }
    except Exception as e:
        print(f"Error getting image URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-images")
async def list_images(user_id: str = Query(..., description="User ID to list images for")):
    try:
        print(f"\n=== Starting image listing process for user {user_id} ===")
        bucket = storage.bucket()

        # List all images under the user's directory
        prefix = f"users/{user_id}/images/"
        print(f"Listing images with prefix: {prefix}")
        blobs = list(bucket.list_blobs(prefix=prefix))
        print(f"Found {len(blobs)} blobs in {prefix}")

        images = []
        folders = set(["Uncategorized"])  # Track unique folders
        folder_image_counts = {"Uncategorized": 0}  # Track image counts per folder

        for blob in blobs:
            if blob.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                # Extract folder from path
                path_parts = blob.name.split('/')
                if len(path_parts) >= 5:  # users/user_id/images/folder/filename
                    folder = path_parts[-2]
                    folders.add(folder)
                    # Increment folder image count
                    folder_image_counts[folder] = folder_image_counts.get(folder, 0) + 1
                else:
                    folder = "Uncategorized"
                    folder_image_counts["Uncategorized"] += 1

                # Generate a signed URL that expires in 1 hour
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(hours=1),
                    method="GET"
                )

                # Get metadata
                try:
                    metadata = blob.metadata or {}
                except Exception as e:
                    print(f"Error getting metadata for {blob.name}: {e}")
                    metadata = {}

                image_data = {
                    "url": signed_url,
                    "filename": blob.name.split('/')[-1],
                    "storage_path": blob.name,
                    "created": blob.time_created.isoformat() if blob.time_created else None,
                    "size": blob.size,
                    "content_type": blob.content_type,
                    "prompt": metadata.get('prompt', ''),
                    "parameters": metadata.get('parameters', ''),
                    "folder": metadata.get('folder', folder)  # Use metadata folder or path-derived folder
                }
                images.append(image_data)

        print(f"Found {len(images)} images in {len(folders)} folders")
        print(f"Folder image counts: {folder_image_counts}")

        return {
            "images": images,
            "folders": list(folders),
            "folder_image_counts": folder_image_counts
        }
    except Exception as e:
        print(f"\n=== Error listing images ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/move-image")
async def move_image(
    storage_path: str = Form(...),
    new_folder: str = Form(...),
    user_data: dict = Depends(firebase_auth)
):
    try:
        # Verify user has access to this image
        user_id = user_data.get('uid')
        if not user_id:
            raise HTTPException(status_code=401, detail="No valid user id found")

        # Verify the image belongs to the user
        if not storage_path.startswith(f"users/{user_id}/"):
            raise HTTPException(status_code=403, detail="Access denied")

        # Sanitize folder name
        new_folder = "".join(c for c in new_folder if c.isalnum() or c in (' ', '-', '_')).strip()
        if not new_folder:
            new_folder = "Uncategorized"

        # Get the source blob
        source_blob = bucket.blob(storage_path)
        if not source_blob.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        # Create new storage path
        filename = storage_path.split('/')[-1]
        new_storage_path = f"users/{user_id}/images/{new_folder}/{filename}"

        # Copy to new location
        new_blob = bucket.copy_blob(source_blob, bucket, new_storage_path)

        # Update metadata
        metadata = source_blob.metadata or {}
        metadata['folder'] = new_folder
        new_blob.metadata = metadata
        new_blob.patch()

        # Delete old blob
        source_blob.delete()

        # Generate new signed URL
        signed_url = new_blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=1),
            method="GET"
        )

        return {
            "success": True,
            "new_storage_path": new_storage_path,
            "url": signed_url
        }

    except Exception as e:
        print(f"Error moving image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete-folder")
async def delete_folder(
    folder_name: str = Form(...),
    user_id: str = Form(...),
    force_delete: bool = Form(False),  # Add force_delete parameter with default False
    user_data: dict = Depends(firebase_auth)
):
    try:
        print(f"\n=== Starting folder deletion process for folder '{folder_name}' ===")

        # Verify user has access
        auth_user_id = user_data.get('uid')
        if not auth_user_id:
            raise HTTPException(status_code=401, detail="No valid user id found")

        # Verify the user_id matches the authenticated user
        if auth_user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Sanitize folder name
        folder_name = "".join(c for c in folder_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not folder_name or folder_name == "Uncategorized":
            raise HTTPException(status_code=400, detail="Cannot delete the Uncategorized folder")

        # Check if folder has any images
        bucket = storage.bucket()
        prefix = f"users/{user_id}/images/{folder_name}/"
        print(f"Checking for images in folder: {prefix}")

        # List all blobs in the folder
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        # Count only image files for the error message
        image_blobs = [blob for blob in blobs if blob.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
        non_image_blobs = [blob for blob in blobs if not blob.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
        
        print(f"Found {len(image_blobs)} images and {len(non_image_blobs)} other files in folder")
        
        if len(image_blobs) > 0 and not force_delete:
            print(f"Found {len(image_blobs)} images in folder, cannot delete")
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot delete folder with {len(image_blobs)} images. Please move or delete images first."
            )

        # If force_delete is True, we'll delete all files in the folder
        if len(blobs) > 0 and force_delete:
            print(f"Force deleting folder with {len(blobs)} files")
            
            # Delete all blobs in the folder
            for blob in blobs:
                try:
                    print(f"Deleting file: {blob.name}")
                    blob.delete()
                except Exception as e:
                    print(f"Error deleting file {blob.name}: {str(e)}")
                    # Continue with other files even if one fails

        print(f"Folder is empty or all files deleted, proceeding with deletion")

        # Since Firebase Storage doesn't have actual folders (just prefixes in object paths),
        # we don't need to do anything else to delete the folder.

        return {
            "success": True,
            "message": f"Folder '{folder_name}' deleted successfully"
        }

    except HTTPException as he:
        print(f"HTTP error deleting folder: {he.detail}")
        raise he
    except Exception as e:
        print(f"Error deleting folder: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-folder")
async def create_folder(
    folder_name: str = Form(...),
    user_id: str = Form(...),
    user_data: dict = Depends(firebase_auth)
):
    try:
        print(f"\n=== Starting folder creation process for folder '{folder_name}' ===")

        # Verify user has access
        auth_user_id = user_data.get('uid')
        if not auth_user_id:
            raise HTTPException(status_code=401, detail="No valid user id found")

        # Verify the user_id matches the authenticated user
        if auth_user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Sanitize folder name
        folder_name = "".join(c for c in folder_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not folder_name:
            raise HTTPException(status_code=400, detail="Invalid folder name")
        
        if folder_name == "Uncategorized":
            # Uncategorized folder always exists, no need to create it
            return {
                "success": True,
                "message": "Uncategorized folder already exists"
            }

        # Check if folder already exists
        bucket = storage.bucket()
        prefix = f"users/{user_id}/images/{folder_name}/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        if len(blobs) > 0:
            print(f"Folder '{folder_name}' already exists with {len(blobs)} files")
            return {
                "success": True,
                "message": f"Folder '{folder_name}' already exists"
            }
        
        # Create a placeholder.txt file to ensure the folder exists
        placeholder_path = f"users/{user_id}/images/{folder_name}/placeholder.txt"
        placeholder_blob = bucket.blob(placeholder_path)
        placeholder_blob.upload_from_string(
            f"This is a placeholder file to ensure the folder '{folder_name}' exists. Created at {datetime.now(central_tz).isoformat()}"
        )
        
        print(f"Created folder '{folder_name}' with placeholder file")
        
        return {
            "success": True,
            "message": f"Folder '{folder_name}' created successfully"
        }

    except HTTPException as he:
        print(f"HTTP error creating folder: {he.detail}")
        raise he
    except Exception as e:
        print(f"Error creating folder: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Please run main.py to start the server")
