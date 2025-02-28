from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import storage
import os
from datetime import datetime, timedelta
from utils.config import initialize_environment, get_firebase_config
import pytz  # Add pytz for timezone handling

# Initialize environment
initialize_environment()

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
)

# Initialize Firebase Admin
try:
    if not firebase_admin._apps:
        from config import initialize_firebase, get_firebase_config
        initialize_firebase()

    # Use the correct bucket path
    storage_bucket = "urbanworks-v2.firebasestorage.app"
    bucket = storage.bucket(storage_bucket)
    
    # Check bucket existence but don't try to create it
    if not bucket.exists():
        print("Bucket does not exist. Please create the bucket manually in the Firebase Console.")
        print("You must verify site or domain ownership at https://search.google.com/search-console")
except Exception as e:
    print(f"Firebase initialization error: {e}")

# Define Central Time Zone
central_tz = pytz.timezone('US/Central')

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    template_name: str = Form(None),
    category: str = Form(None),
    user_id: str = Form(None),
    prompt: str = Form(None),
    parameters: str = Form(None),
    storage_path: str = Form(None)
):
    try:
        # Only log details if not a redirect (prevents duplicate logging)
        skip_redirect = request.headers.get('X-Skip-Redirect') == 'true'
        if not skip_redirect:
            print(f"\n=== Starting upload process ===")
            print(f"File details:")
            print(f"- Filename: {file.filename}")
            print(f"- Content type: {file.content_type}")
            print(f"- User ID: {user_id}")
            print(f"- Storage path: {storage_path}")

        # Validate file type
        allowed_types = [
            'application/pdf', 
            'application/msword', 
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'image/jpeg',
            'image/jpg',
            'image/png',
            'image/gif',
            'image/webp'
        ]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"File type {file.content_type} not allowed")

        # Read file content
        content = await file.read()

        # Generate a unique filename using timestamp
        timestamp = datetime.now(central_tz).strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{os.path.basename(file.filename)}"

        # Use provided storage path or generate one based on user_id
        if not storage_path:
            if user_id:
                if file.content_type.startswith('image/'):
                    storage_path = f"users/{user_id}/images/{filename}"
                else:
                    storage_path = f"users/{user_id}/templates/{filename}"
            else:
                if file.content_type.startswith('image/'):
                    storage_path = f"Images/{filename}"
                else:
                    storage_path = f"Templates/{filename}"

        try:
            bucket = storage.bucket()
            blob = bucket.blob(storage_path)

            # Set metadata
            metadata = {
                'contentType': file.content_type,
                'created': datetime.now(central_tz).isoformat()
            }

            if user_id:
                metadata['userId'] = user_id

            if prompt:
                metadata['prompt'] = prompt

            if parameters:
                metadata['parameters'] = parameters

            # Set metadata on blob
            blob.metadata = metadata

            # Upload to Firebase Storage
            blob.upload_from_string(
                content,
                content_type=file.content_type
            )

            # Make sure metadata is saved
            blob.patch()

            # Generate the direct Firebase Storage URL
            storage_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{storage_path.replace('/', '%2F')}?alt=media"

            # Return the complete response with metadata
            response_data = {
                "success": True,
                "url": storage_url,
                "filename": filename,
                "content_type": file.content_type,
                "storage_path": storage_path,
                "prompt": prompt,
                "parameters": parameters
            }

            if not skip_redirect:
                print("\n=== Upload process completed successfully ===")

            return response_data

        except Exception as storage_error:
            print(f"\n=== Storage error occurred ===")
            print(f"Error type: {type(storage_error)}")
            print(f"Error message: {str(storage_error)}")
            raise HTTPException(status_code=500, detail=f"Storage error: {str(storage_error)}")

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"\n=== Unexpected error occurred ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.delete("/delete")
async def delete_file(path: str = Query(..., description="Storage path of the file to delete")):
    try:
        bucket = storage.bucket()  # Use default bucket
        blob = bucket.blob(path)

        if not blob.exists():
            raise HTTPException(status_code=404, detail="File not found")

        blob.delete()

        return {"success": True, "message": "File deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-images")
async def list_images(user_id: str = Query(..., description="User ID to list images for")):
    try:
        print(f"\n=== Starting image listing process for user {user_id} ===")
        bucket = storage.bucket()

        # List blobs in the user's specific images directory
        prefix = f"users/{user_id}/images/"
        print(f"Listing images with prefix: {prefix}")
        blobs = list(bucket.list_blobs(prefix=prefix))
        print(f"Found {len(blobs)} blobs in {prefix}")

        images = []
        for blob in blobs:
            print(f"Processing blob: {blob.name}")
            if blob.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                # Generate the direct Firebase Storage URL
                storage_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{blob.name.replace('/', '%2F')}?alt=media"

                # Get metadata
                try:
                    metadata = blob.metadata or {}
                    print(f"Blob metadata: {metadata}")
                except Exception as e:
                    print(f"Error getting metadata for {blob.name}: {e}")
                    metadata = {}

                image_data = {
                    "url": storage_url,
                    "filename": blob.name.split('/')[-1],
                    "storage_path": blob.name,
                    "created": blob.time_created.isoformat() if blob.time_created else None,
                    "size": blob.size,
                    "content_type": blob.content_type,
                    "prompt": metadata.get('prompt', ''),
                    "parameters": metadata.get('parameters', '')
                }
                print(f"Added image: {image_data['storage_path']}")
                images.append(image_data)

        print(f"Found {len(images)} images for user {user_id}")
        return {"images": images}
    except Exception as e:
        print(f"\n=== Error listing images ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/refresh-url")
async def refresh_image_url(storage_path: str = Query(..., description="Storage path of the image to refresh")):
    try:
        print(f"\n=== Refreshing signed URL for image: {storage_path} ===")
        bucket = storage.bucket()
        blob = bucket.blob(storage_path)

        if not blob.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        # Generate a new signed URL that expires in 1 hour
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
        print(f"Error refreshing signed URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Please run main.py to start the server") 