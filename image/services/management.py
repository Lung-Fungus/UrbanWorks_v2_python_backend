"""
Services for image management (listing, deleting, moving, etc.).
"""
from typing import Dict, Any, List
from fastapi import HTTPException
from datetime import datetime, timedelta

from ..config import (
    bucket, 
    logger, 
    CENTRAL_TZ,
    SUPPORTED_IMAGE_FORMATS,
    DEFAULT_FOLDER
)

async def delete_image(storage_path: str) -> Dict[str, Any]:
    """
    Delete an image from Firebase Storage.
    
    Args:
        storage_path: Path to the image in Firebase Storage
        
    Returns:
        Dictionary with success status and message
    """
    try:
        logger.info(f"Attempting to delete image at path: {storage_path}")

        if not storage_path or storage_path.strip() == "":
            raise HTTPException(status_code=400, detail="Invalid storage path: Path cannot be empty")

        # Get the blob from the bucket and delete it directly
        blob = bucket.blob(storage_path)
        logger.info(f"Checking if blob exists at path: {storage_path}")

        if not blob.exists():
            logger.info(f"Blob not found at path: {storage_path}")
            raise HTTPException(status_code=404, detail=f"Image not found at path: {storage_path}")

        # Delete the blob
        logger.info(f"Blob found, proceeding with deletion")
        blob.delete()

        logger.info(f"Image at path {storage_path} deleted successfully")
        return {
            "success": True, 
            "message": "Image deleted successfully", 
            "path": storage_path
        }

    except HTTPException as he:
        logger.error(f"HTTP error deleting image: {he.detail}")
        raise he
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error deleting image: {error_message}")
        # Include more context in the error
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete image at {storage_path}: {error_message}"
        )

async def get_image_url(storage_path: str, user_id: str) -> Dict[str, Any]:
    """
    Get a signed URL for an image.
    
    Args:
        storage_path: Path to the image in Firebase Storage
        user_id: User ID for access control
        
    Returns:
        Dictionary with signed URL and expiration time
    """
    try:
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
        logger.error(f"Error getting image URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def list_images(user_id: str) -> Dict[str, Any]:
    """
    List all images for a user.
    
    Args:
        user_id: User ID to list images for
        
    Returns:
        Dictionary with images, folders, and folder image counts
    """
    try:
        logger.info(f"\n=== Starting image listing process for user {user_id} ===")

        # List all images under the user's directory
        prefix = f"users/{user_id}/images/"
        logger.info(f"Listing images with prefix: {prefix}")
        blobs = list(bucket.list_blobs(prefix=prefix))
        logger.info(f"Found {len(blobs)} blobs in {prefix}")

        images = []
        folders = set([DEFAULT_FOLDER])  # Track unique folders
        folder_image_counts = {DEFAULT_FOLDER: 0}  # Track image counts per folder

        for blob in blobs:
            if any(blob.name.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS):
                # Extract folder from path
                path_parts = blob.name.split('/')
                if len(path_parts) >= 5:  # users/user_id/images/folder/filename
                    folder = path_parts[-2]
                    folders.add(folder)
                    # Increment folder image count
                    folder_image_counts[folder] = folder_image_counts.get(folder, 0) + 1
                else:
                    folder = DEFAULT_FOLDER
                    folder_image_counts[DEFAULT_FOLDER] += 1

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
                    logger.error(f"Error getting metadata for {blob.name}: {e}")
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

        logger.info(f"Found {len(images)} images in {len(folders)} folders")
        logger.info(f"Folder image counts: {folder_image_counts}")

        return {
            "images": images,
            "folders": list(folders),
            "folder_image_counts": folder_image_counts
        }
    except Exception as e:
        logger.error(f"\n=== Error listing images ===")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def move_image(storage_path: str, new_folder: str, user_id: str) -> Dict[str, Any]:
    """
    Move an image to a different folder.
    
    Args:
        storage_path: Path to the image in Firebase Storage
        new_folder: Name of the folder to move the image to
        user_id: User ID for access control
        
    Returns:
        Dictionary with success status and new storage path
    """
    try:
        # Verify the image belongs to the user
        if not storage_path.startswith(f"users/{user_id}/"):
            raise HTTPException(status_code=403, detail="Access denied")

        # Sanitize folder name
        new_folder = "".join(c for c in new_folder if c.isalnum() or c in (' ', '-', '_')).strip()
        if not new_folder:
            new_folder = DEFAULT_FOLDER

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
        logger.error(f"Error moving image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def delete_folder(folder_name: str, user_id: str, force_delete: bool = False) -> Dict[str, Any]:
    """
    Delete a folder and optionally all its contents.
    
    Args:
        folder_name: Name of the folder to delete
        user_id: User ID for access control
        force_delete: Whether to force delete the folder even if it contains images
        
    Returns:
        Dictionary with success status and message
    """
    try:
        logger.info(f"\n=== Starting folder deletion process for folder '{folder_name}' ===")

        # Sanitize folder name
        folder_name = "".join(c for c in folder_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not folder_name or folder_name == DEFAULT_FOLDER:
            raise HTTPException(status_code=400, detail=f"Cannot delete the {DEFAULT_FOLDER} folder")

        # Check if folder has any images
        prefix = f"users/{user_id}/images/{folder_name}/"
        logger.info(f"Checking for images in folder: {prefix}")

        # List all blobs in the folder
        blobs = list(bucket.list_blobs(prefix=prefix))

        # Count only image files for the error message
        image_blobs = [blob for blob in blobs if any(blob.name.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS)]
        non_image_blobs = [blob for blob in blobs if not any(blob.name.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS)]

        logger.info(f"Found {len(image_blobs)} images and {len(non_image_blobs)} other files in folder")

        if len(image_blobs) > 0 and not force_delete:
            logger.info(f"Found {len(image_blobs)} images in folder, cannot delete")
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot delete folder with {len(image_blobs)} images. Please move or delete images first."
            )

        # If force_delete is True, we'll delete all files in the folder
        if len(blobs) > 0 and force_delete:
            logger.info(f"Force deleting folder with {len(blobs)} files")

            # Delete all blobs in the folder
            for blob in blobs:
                try:
                    logger.info(f"Deleting file: {blob.name}")
                    blob.delete()
                except Exception as e:
                    logger.error(f"Error deleting file {blob.name}: {str(e)}")
                    # Continue with other files even if one fails

        logger.info(f"Folder is empty or all files deleted, proceeding with deletion")

        # Since Firebase Storage doesn't have actual folders (just prefixes in object paths),
        # we don't need to do anything else to delete the folder.

        return {
            "success": True,
            "message": f"Folder '{folder_name}' deleted successfully"
        }

    except HTTPException as he:
        logger.error(f"HTTP error deleting folder: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error deleting folder: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def create_folder(folder_name: str, user_id: str) -> Dict[str, Any]:
    """
    Create a new folder.
    
    Args:
        folder_name: Name of the folder to create
        user_id: User ID for access control
        
    Returns:
        Dictionary with success status and message
    """
    try:
        logger.info(f"\n=== Starting folder creation process for folder '{folder_name}' ===")

        # Sanitize folder name
        folder_name = "".join(c for c in folder_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not folder_name:
            raise HTTPException(status_code=400, detail="Invalid folder name")

        if folder_name == DEFAULT_FOLDER:
            # Uncategorized folder always exists, no need to create it
            return {
                "success": True,
                "message": f"{DEFAULT_FOLDER} folder already exists"
            }

        # Check if folder already exists
        prefix = f"users/{user_id}/images/{folder_name}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        if len(blobs) > 0:
            logger.info(f"Folder '{folder_name}' already exists with {len(blobs)} files")
            return {
                "success": True,
                "message": f"Folder '{folder_name}' already exists"
            }

        # Create a placeholder.txt file to ensure the folder exists
        placeholder_path = f"users/{user_id}/images/{folder_name}/placeholder.txt"
        placeholder_blob = bucket.blob(placeholder_path)
        placeholder_blob.upload_from_string(
            f"This is a placeholder file to ensure the folder '{folder_name}' exists. Created at {datetime.now(CENTRAL_TZ).isoformat()}"
        )

        logger.info(f"Created folder '{folder_name}' with placeholder file")

        return {
            "success": True,
            "message": f"Folder '{folder_name}' created successfully"
        }

    except HTTPException as he:
        logger.error(f"HTTP error creating folder: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error creating folder: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 