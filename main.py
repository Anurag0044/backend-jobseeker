from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from middleware.firebase_auth import _initialize_firebase
from routers import generate


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _initialize_firebase()
    yield


app = FastAPI(
    title="Job Seeker's Concierge API",
    description=(
        "Secure AI-powered backend for resume tailoring, cover letter generation, "
        "and job application assistance. Auth: Firebase. AI: Gemini (Phase 2)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dev mode — will be locked down in Phase 6
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router, prefix="/api/v1")


@app.get("/")
def health_check() -> dict:
    return {"status": "online", "project": "Job Seeker's Concierge API"}
