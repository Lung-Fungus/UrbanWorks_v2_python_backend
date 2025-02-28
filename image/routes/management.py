"""
Routes for image management (listing, deleting, moving, etc.).
"""
from fastapi import APIRouter, Depends, Form, HTTPException, Query
from utils.auth_middleware import firebase_auth

from ..services.management import (
    delete_image,
    get_image_url,
    list_images,
    move_image,
    delete_folder,
    create_folder
)

# Create router
router = APIRouter(tags=["image_management"])

@router.delete("/delete-image")
async def delete_image_endpoint(storage_path: str):
    """
    Delete an image from Firebase Storage.
    
    Args:
        storage_path: Path to the image in Firebase Storage
        
    Returns:
        Dictionary with success status and message
    """
    return await delete_image(storage_path)

@router.get("/get-image")
async def get_image_endpoint(
    storage_path: str = Query(...), 
    user_data: dict = Depends(firebase_auth)
):
    """
    Get a signed URL for an image.
    
    Args:
        storage_path: Path to the image in Firebase Storage
        user_data: User data from authentication
        
    Returns:
        Dictionary with signed URL and expiration time
    """
    # Extract user_id from user_data
    user_id = user_data.get('uid')
    if not user_id:
        raise HTTPException(status_code=401, detail="No valid user id found")
        
    return await get_image_url(storage_path, user_id)

@router.get("/list-images")
async def list_images_endpoint(
    user_id: str = Query(..., description="User ID to list images for")
):
    """
    List all images for a user.
    
    Args:
        user_id: User ID to list images for
        
    Returns:
        Dictionary with images, folders, and folder image counts
    """
    return await list_images(user_id)

@router.post("/move-image")
async def move_image_endpoint(
    storage_path: str = Form(...),
    new_folder: str = Form(...),
    user_data: dict = Depends(firebase_auth)
):
    """
    Move an image to a different folder.
    
    Args:
        storage_path: Path to the image in Firebase Storage
        new_folder: Name of the folder to move the image to
        user_data: User data from authentication
        
    Returns:
        Dictionary with success status and new storage path
    """
    # Extract user_id from user_data
    user_id = user_data.get('uid')
    if not user_id:
        raise HTTPException(status_code=401, detail="No valid user id found")
        
    return await move_image(storage_path, new_folder, user_id)

@router.post("/delete-folder")
async def delete_folder_endpoint(
    folder_name: str = Form(...),
    user_id: str = Form(...),
    force_delete: bool = Form(False),
    user_data: dict = Depends(firebase_auth)
):
    """
    Delete a folder and optionally all its contents.
    
    Args:
        folder_name: Name of the folder to delete
        user_id: User ID for access control
        force_delete: Whether to force delete the folder even if it contains images
        user_data: User data from authentication
        
    Returns:
        Dictionary with success status and message
    """
    # Verify the user_id matches the authenticated user
    auth_user_id = user_data.get('uid')
    if not auth_user_id:
        raise HTTPException(status_code=401, detail="No valid user id found")
        
    if auth_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
        
    return await delete_folder(folder_name, user_id, force_delete)

@router.post("/create-folder")
async def create_folder_endpoint(
    folder_name: str = Form(...),
    user_id: str = Form(...),
    user_data: dict = Depends(firebase_auth)
):
    """
    Create a new folder.
    
    Args:
        folder_name: Name of the folder to create
        user_id: User ID for access control
        user_data: User data from authentication
        
    Returns:
        Dictionary with success status and message
    """
    # Verify the user_id matches the authenticated user
    auth_user_id = user_data.get('uid')
    if not auth_user_id:
        raise HTTPException(status_code=401, detail="No valid user id found")
        
    if auth_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
        
    return await create_folder(folder_name, user_id)

@router.get("/ensure-user-bucket")
async def ensure_user_bucket_endpoint(
    user_data: dict = Depends(firebase_auth)
):
    """
    Ensure that the user's bucket and folder structure exists.
    
    Args:
        user_data: User data from authentication
        
    Returns:
        Dictionary with success status and message
    """
    # Extract user_id from user_data
    user_id = user_data.get('uid')
    if not user_id:
        raise HTTPException(status_code=401, detail="No valid user id found")
    
    # We don't need to do anything special here since the bucket is already configured correctly
    # Just return success to let the frontend know it can proceed
    return {
        "success": True,
        "message": "User bucket is accessible"
    } 