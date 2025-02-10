from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, storage
import os
from datetime import datetime
from config import get_firebase_credentials, initialize_environment, get_firebase_config

# Initialize environment
initialize_environment()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase Admin
try:
    if not firebase_admin._apps:
        from config import initialize_firebase
        initialize_firebase()
    
    # Get default bucket
    bucket = storage.bucket()
    print(f"Firebase initialized successfully with bucket: {bucket.name}")
    
    # Test bucket existence and create if needed
    if not bucket.exists():
        print("Bucket does not exist, attempting to create...")
        try:
            bucket.create()
            print("Bucket created successfully")
            # Set bucket CORS configuration
            bucket.cors = [
                {
                    'origin': ['http://localhost:3000'],
                    'method': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
                    'responseHeader': ['Content-Type'],
                    'maxAgeSeconds': 3600
                }
            ]
            bucket.patch()
            print("Bucket CORS configuration updated")
        except Exception as create_error:
            print(f"Failed to create bucket: {str(create_error)}")
            print("Please create the bucket manually in the Firebase Console")
    else:
        print("Bucket exists and is accessible")
except Exception as e:
    print(f"Firebase initialization error: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    template_name: str = Form(None),
    category: str = Form(None),
    user_id: str = Form(None),
    prompt: str = Form(None),
    parameters: str = Form(None)
):
    try:
        print(f"\n=== Starting upload process ===")
        print(f"File details:")
        print(f"- Filename: {file.filename}")
        print(f"- Content type: {file.content_type}")
        print(f"- Prompt: {prompt}")
        print(f"- Parameters: {parameters}")
        
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
        print(f"Validating file type: {file.content_type}")
        if file.content_type not in allowed_types:
            print(f"File type validation failed: {file.content_type} not in allowed types")
            raise HTTPException(status_code=400, detail=f"File type {file.content_type} not allowed")
        print("File type validation passed")

        # Read file content
        print("Reading file content...")
        content = await file.read()
        print(f"File content read successfully, size: {len(content)} bytes")
        
        # Generate unique filename while preserving original name
        base_name, file_extension = os.path.splitext(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Determine the storage path based on file type
        if file.content_type.startswith('image/'):
            unique_filename = f"Images/{base_name}_{timestamp}{file_extension}"
            print(f"File identified as image, using Images/ directory")
        else:
            unique_filename = f"Templates/{base_name}_{timestamp}{file_extension}"
            print(f"File identified as document, using Templates/ directory")
            
        print(f"Generated unique filename: {unique_filename}")

        try:
            print("\n=== Starting Firebase Storage upload ===")
            bucket = storage.bucket()
            print(f"Got bucket reference: {bucket.name}")
            
            # Create blob
            blob = bucket.blob(unique_filename)
            print(f"Created blob reference: {blob.name}")

            # Set metadata
            metadata = {
                'contentType': file.content_type
            }
            
            # Add prompt if provided
            if prompt:
                print(f"Adding prompt to metadata: {prompt}")
                metadata['prompt'] = prompt
            
            # Add parameters if provided
            if parameters:
                print(f"Adding parameters to metadata: {parameters}")
                metadata['parameters'] = parameters
                
            print(f"Setting complete metadata: {metadata}")
            
            # Set metadata on blob
            blob.metadata = metadata
            
            # Upload to Firebase Storage
            print("Uploading file to Firebase Storage...")
            blob.upload_from_string(
                content,
                content_type=file.content_type
            )
            
            # Make sure metadata is saved
            blob.patch()
            
            print("File uploaded successfully to Firebase Storage")

            # Make the blob publicly accessible
            print("Making blob publicly accessible...")
            blob.make_public()
            print("Blob made public successfully")

            # Get the public URL
            public_url = blob.public_url
            print(f"Generated public URL: {public_url}")

            # Return the complete response with metadata
            response_data = {
                "success": True,
                "url": public_url,
                "filename": file.filename,
                "content_type": file.content_type,
                "storage_path": unique_filename,
                "prompt": prompt,
                "parameters": parameters
            }
            print("\n=== Upload process completed successfully ===")
            print(f"Response data: {response_data}")
            return response_data

        except Exception as storage_error:
            print(f"\n=== Storage error occurred ===")
            print(f"Error type: {type(storage_error)}")
            print(f"Error message: {str(storage_error)}")
            if hasattr(storage_error, '__dict__'):
                print(f"Error details: {storage_error.__dict__}")
            raise HTTPException(status_code=500, detail=f"Storage error: {str(storage_error)}")

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"\n=== Unexpected error occurred ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        if hasattr(e, '__dict__'):
            print(f"Error details: {e.__dict__}")
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
async def list_images():
    try:
        print("\n=== Starting image listing process ===")
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix="Images/")
        
        images = []
        for blob in blobs:
            if blob.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                blob.make_public()  # Ensure the image is publicly accessible
                # Get metadata
                metadata = blob.metadata or {}
                images.append({
                    "url": blob.public_url,
                    "filename": blob.name.split('/')[-1],
                    "storage_path": blob.name,
                    "created": blob.time_created.isoformat() if blob.time_created else None,
                    "size": blob.size,
                    "content_type": blob.content_type,
                    "prompt": metadata.get('prompt', ''),
                    "parameters": metadata.get('parameters', '')
                })
        
        print(f"Found {len(images)} images")
        return {"images": images}
    except Exception as e:
        print(f"\n=== Error listing images ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Please run main.py to start the server") 