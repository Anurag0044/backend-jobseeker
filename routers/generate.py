from fastapi import APIRouter, Depends
from pydantic import BaseModel

from middleware.firebase_auth import verify_firebase_token

router = APIRouter()


class GenerateRequest(BaseModel):
    resume_text: str
    job_url: str


@router.post("/generate")
def generate(
    payload: GenerateRequest,
    token: dict = Depends(verify_firebase_token),
) -> dict:
    return {
        "status": "success",
        "message": "Phase 1 scaffold verified. AI pipeline coming in Phase 2.",
        "user_uid": token.get("uid"),
        "received": {
            "resume_length": len(payload.resume_text),
            "job_url": payload.job_url,
        },
    }
