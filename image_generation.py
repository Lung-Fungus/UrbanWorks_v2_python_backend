from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import replicate
import anthropic
from typing import Optional
import firebase_admin
from firebase_admin import storage
from datetime import datetime
import httpx
import json
from config import get_api_keys, initialize_environment, get_firebase_config
from auth_middleware import firebase_auth
from fastapi import Depends
import os

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

async def improve_prompt(prompt: str) -> str:
    system_prompt = """You are an expert at writing prompts for FLUX image generation. 
    Your task is to improve the given prompt to create more detailed and visually appealing images.
    Focus on adding details about style, lighting, composition, and mood while maintaining the original intent.
    Return only the improved prompt without any explanations."""

    response = claude_client.messages.create(
        model="claude-3-5-sonnet-20240620",
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
    output_format: str = Form("jpg")
):
    try:
        print(f"Starting image generation with prompt: {prompt}")
        #print(f"Headers: {request.headers}")  # Log all headers
        #print(f"User data: {user_data}")  # Log user_data for debugging

        # Extract token from user_data.  Let's be VERY explicit and check multiple places.
        token = None
        if user_data:
            token = user_data.get('token')
            if not token:
                token = user_data.get('uid')  # Sometimes Firebase uses 'uid'
            if not token:
                token = user_data.get('user_id')  # Another possibility

        # NEW: Extract user_id separately from user_data and validate
        user_id = user_data.get('uid') or user_data.get('user_id')
        if not user_id:
            raise HTTPException(status_code=401, detail="No valid user id found")

        # Check Authorization header directly (for debugging)
        auth_header = request.headers.get('Authorization')
        if auth_header:
            #print(f"Authorization header: {auth_header}")
            if auth_header.startswith("Bearer "):
                header_token = auth_header.split(" ")[1]
                #print(f"Token from header: {header_token}")
                if not token:
                    token = header_token # Prioritize header token if found

        if not token:
            raise HTTPException(status_code=401, detail="No valid authentication token found")

        # Handle image prompt upload if provided
        image_prompt_url = None
        if image_prompt:
            print(f"Processing image prompt: {image_prompt.filename}")
            files = {'file': (image_prompt.filename, await image_prompt.read(), image_prompt.content_type)}
            print(f"Uploading reference image with content type: {image_prompt.content_type}")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{API_BASE_URL}/upload',
                    files=files,
                    headers={'Authorization': f'Bearer {token}'},
                    follow_redirects=True,
                )
                print(f"Reference image upload response status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    image_prompt_url = data['url']
                    print(f"Reference image uploaded successfully: {image_prompt_url}")
                else:
                    print(f"Reference image upload failed: {await response.text()}")

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
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"generated_image_{timestamp}.{output_format}"
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

                    # Add metadata fields and include userId so that the file is stored correctly.
                    data = {
                        'prompt': prompt,  # Send prompt directly
                        'userId': user_id,  # NEW: include user id to dictate the folder path in Firebase
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

                    print(f"Uploading with metadata:")
                    print(f"Prompt: {prompt}")
                    print(f"Parameters: {data['parameters']}")

                    # Make the request with both files and form data, following redirects
                    upload_response = await client.post(
                        f'{API_BASE_URL}/upload',
                        files=files,
                        data=data,
                        headers={'Authorization': f'Bearer {token}'},
                        follow_redirects=True,
                    )

                    print(f"Upload response status: {upload_response.status_code}")

                    if upload_response.status_code != 200:
                        error_text = await upload_response.aread()
                        print(f"Upload failed with status {upload_response.status_code}")
                        print(f"Response headers: {upload_response.headers}")
                        print(f"Response body: {error_text}")
                        raise Exception(f"Failed to upload generated image. Status: {upload_response.status_code}, Response: {error_text}")

                    response_text = await upload_response.aread()
                    upload_data = json.loads(response_text.decode('utf-8'))
                    firebase_url = upload_data['url']
                    storage_path = upload_data['storage_path']
                    print(f"Upload successful. Firebase URL: {firebase_url}, Storage path: {storage_path}")

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
        # Use the delete endpoint from firestore_upload
        async with httpx.AsyncClient() as client:
            # The firestore_upload endpoint expects 'path' not 'storage_path'
            response = await client.delete(f'/upload/delete?path={storage_path}')

            if response.status_code != 200:
                response_text = await response.aread()
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to delete image: {response_text.decode('utf-8')}"
                )

            print("Image deleted successfully")
            return {"success": True, "message": "Image deleted successfully"}

    except HTTPException as he:
        print(f"HTTP error deleting image: {he.detail}")
        raise he
    except Exception as e:
        print(f"Error deleting image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-images")
async def list_images(userId: str):
    try:
        print(f"Listing images for user: {userId}")
        all_images = []

        # Only list blobs in the user's specific directory
        user_blobs = bucket.list_blobs(prefix=f"users/{userId}/images/")

        for blob in user_blobs:
            try:
                # Get the metadata
                blob.reload()  # Ensure we have the latest metadata
                metadata = blob.metadata or {}

                # Create image data structure
                image_data = {
                    "storage_path": blob.name,
                    "url": f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{blob.name.replace('/', '%2F')}?alt=media",
                    "prompt": metadata.get("prompt", ""),
                    "created": metadata.get("created", datetime.datetime.now().isoformat()),
                    "parameters": metadata.get("parameters", "{}"),
                }
                all_images.append(image_data)
                print(f"Added image: {blob.name}")
            except Exception as e:
                print(f"Error processing blob {blob.name}: {str(e)}")
                continue

        # Sort by creation time, newest first
        all_images.sort(key=lambda x: x["created"], reverse=True)
        print(f"Total images found: {len(all_images)}")
        return {"images": all_images}

    except Exception as e:
        print(f"Error listing images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Please run main.py to start the server")
