from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config import verify_firebase_token

class FirebaseAuthMiddleware(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(FirebaseAuthMiddleware, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        try:
            credentials: HTTPAuthorizationCredentials = await super(FirebaseAuthMiddleware, self).__call__(request)
            if credentials.scheme != "Bearer":
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authentication scheme. Use Bearer token."
                )

            decoded_token = verify_firebase_token(credentials.credentials)
            request.state.user = decoded_token
            return decoded_token

        except Exception:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )

# Create a reusable instance
firebase_auth = FirebaseAuthMiddleware() 