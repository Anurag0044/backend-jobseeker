# Job Seeker's Concierge - Backend Architecture & Working Document

## Section 1 — Project Overview

**What the project does:** The Job Seeker's Concierge is an AI-powered backend system that automates the creation of customized cover letters and optimized resumes. It scrapes job descriptions from provided URLs, parses the user's uploaded resume, and leverages advanced LLMs to generate highly tailored application materials.

**The problem it solves:** Applying for jobs is time-consuming and tedious. Manually tailoring resumes and writing unique cover letters for every application is a massive bottleneck for job seekers. Many candidates send generic applications that fail to pass ATS (Applicant Tracking Systems) or catch recruiters' eyes.

**The solution it delivers:** It provides a one-click automated pipeline. The user simply provides a target job posting URL and their base resume. The system extracts the job requirements, aligns the user's experience with the job description, and outputs a custom, ATS-friendly resume and a persuasive cover letter tailored perfectly to the role.

**Who uses it and how:** Job seekers use the system via a Next.js web frontend. They authenticate with Firebase, upload their resume (PDF/Text/Docx), paste a job posting URL, and click "Generate". The frontend sends a request to this backend, which orchestrates multiple AI agents to process the request and returns the customized documents.

## Section 2 — Architecture Diagram (ASCII)

```text
+---------------------+        (1) Auth     +-------------------------+
|                     |-------------------->|                         |
| Client (Browser /   |                     | Firebase Authentication |
| Next.js Frontend)   |<--------------------|                         |
|                     |      (2) Token      +-------------------------+
+---------------------+
           |
           | (3) Request (JWT, Resume File, Job URL)
           v
+-----------------------------------------------------------------------------------+
| FastAPI Backend                                                                   |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | Middleware Stack                                                            |  |
|  | 1. Request ID (Assigns UUID)                                                |  |
|  | 2. Security Headers (Sets HSTS, X-Frame-Options, etc.)                      |  |
|  | 3. Rate Limiter (SlowAPI limits per user/IP)                                |  |
|  | 4. Firebase Auth (Validates JWT)                                            |  |
|  +-----------------------------------------------------------------------------+  |
|           |                                                                       |
|           | (4) Validated Request                                                 |
|           v                                                                       |
|  +-----------------------------------------------------------------------------+  |
|  | Route Handler (routers/generate.py)                                         |  |
|  | -> Sanitizes Input (bleach, url validation)                                 |  |
|  | -> Parses Resume File (PyPDF2, text extraction)                             |  |
|  +-----------------------------------------------------------------------------+  |
|           |                                                                       |
|           v                                                                       |
|  +-----------------------------------------------------------------------------+  |
|  | Orchestrator (agents/orchestrator.py)                                       |  |
|  |                                                                             |  |
|  |   [Scraper Agent] ---> (Extracts job details from URL)                      |  |
|  |         |                                                                   |  |
|  |         v                                                                   |  |
|  |   [Resume Agent] ----> (Generates optimized resume)                         |  |
|  |         |                                                                   |  |
|  |         v                                                                   |  |
|  |   [Cover Letter Agent] (Generates tailored cover letter)                    |  |
|  +-----------------------------------------------------------------------------+  |
|           |                                            ^                          |
|           | (5) Prompts                                | (6) Generated Text       |
|           v                                            |                          |
|  +-----------------------------------------------------------------------------+  |
|  | Gemini 2.5 Flash API (Google ADK)                                           |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
           |
           | (7) JSON Response (Resume, Cover Letter, Request ID)
           v
+---------------------+
| Client (Browser)    |
+---------------------+
```

## Section 3 — Complete File Structure

```text
job-concierge-backend/
├── main.py                     # Application entry point; initializes FastAPI, mounts routers, adds middleware
├── requirements.txt            # Python package dependencies with exact versions
├── render.yaml                 # Infrastructure-as-code configuration for Render.com deployment
├── .env                        # Local environment variables (API keys, settings) - Not in version control
├── .env.example                # Template showing required environment variables
├── .gitignore                  # Specifies intentionally untracked files to ignore (e.g., .env, __pycache__)
├── README.md                   # High-level project documentation
├── working.md                  # This file; comprehensive living documentation of the architecture and state
├── routers/
│   └── generate.py             # FastAPI router handling the /api/v1/generate endpoints
├── middleware/
│   ├── firebase_auth.py        # Middleware intercepting requests to validate Firebase JWT tokens
│   ├── request_id.py           # Middleware assigning unique UUIDs to every incoming request
│   └── security_headers.py     # Middleware appending security headers (CORS, HSTS, X-Content-Type-Options)
├── agents/
│   ├── __init__.py             # Makes agents directory a Python module
│   ├── scraper_agent.py        # Agent responsible for fetching and parsing job descriptions from URLs
│   ├── resume_agent.py         # Agent responsible for tailoring the resume to the job description
│   ├── cover_letter_agent.py   # Agent responsible for drafting the customized cover letter
│   └── orchestrator.py         # Coordinates data flow sequentially between scraper, resume, and cover letter agents
├── core/
│   ├── config.py               # Pydantic BaseSettings loading and validating environment variables
│   ├── gemini_client.py        # Wrapper initializing and managing the Google Gemini ADK client
│   ├── file_parser.py          # Utility for extracting text from uploaded resume files (PDF, txt, docx)
│   ├── logger.py               # Centralized Loguru logger configuration with formatting and sinks
│   ├── rate_limiter.py         # SlowAPI rate limiting configuration to prevent abuse
│   ├── error_handlers.py       # Global exception handlers formatting errors into standardized ErrorResponse JSON
│   └── input_sanitizer.py      # Utility functions using bleach to clean inputs and validate URLs
└── tests/
    ├── conftest.py             # Pytest fixtures (e.g., test client, mock data, auth bypasses)
    ├── test_health.py          # Tests for the root health check endpoints
    ├── test_auth.py            # Tests verifying the Firebase auth middleware accepts/rejects tokens
    ├── test_file_parser.py     # Tests for file extraction logic (PDF parsing, bad formats)
    ├── test_pipeline.py        # Integration tests for the orchestrator and agent flow
    ├── test_sanitizer.py       # Tests validating XSS prevention and URL validation
    ├── test_error_handlers.py  # Tests verifying proper HTTP status codes and ErrorResponse formatting
    └── test_hallucination_guard.py # Tests ensuring the AI responses do not include fabricated data
```

## Section 4 — Tech Stack

- **FastAPI**: (v0.110.0) High-performance web framework used for building the API endpoints and routing.
- **Uvicorn**: (v0.29.0) ASGI web server implementation used to run the FastAPI application.
- **Firebase Admin SDK**: (firebase-admin v6.4.0) Verifies client authentication JWT tokens securely.
- **Google ADK / google-genai**: (google-genai v0.3.0) The official Google GenAI SDK to communicate with Gemini.
- **Gemini 2.5 Flash**: The LLM model utilized for fast, high-quality, and cost-effective text generation.
- **loguru**: (v0.7.2) Replaces standard logging for colorful, asynchronous, and easy-to-configure structured logging.
- **slowapi**: (v0.1.9) Rate-limiting extension for FastAPI to protect endpoints from spam/DDoS.
- **tenacity**: (v8.2.3) Implements automatic retry logic with exponential backoff for external API calls (Gemini/Scraping).
- **bleach**: (v6.1.0) HTML sanitization library to strip malicious tags from user inputs.
- **PyPDF2**: (v3.0.1) Library used to extract raw text content from uploaded PDF resumes.
- **httpx**: (v0.27.0) Async HTTP client for fetching job description HTML and making external API requests.
- **BeautifulSoup4**: (beautifulsoup4 v4.12.3) HTML parser used by the Scraper Agent to extract clean text from job sites.
- **pydantic**: (v2.6.4) Data validation and settings management using Python type annotations.
- **python-dotenv**: (v1.0.1) Loads environment variables from a `.env` file into `os.environ`.
- **python-multipart**: (v0.0.9) Required by FastAPI to process `multipart/form-data` for file uploads.
- **pytest + pytest-asyncio**: (v8.1.1, v0.23.6) Testing framework and async plugin for writing and running the test suite.

## Section 5 — Environment Variables

| Variable Name | Source/Where to get it | Example Format | What breaks if missing |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | Google AI Studio | `AIzaSy...` | The entire AI generation pipeline fails; agents cannot call the model. |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Firebase Console -> Project Settings -> Service Accounts (Stringified JSON) | `{"type": "service_account", "project_id": "..."}` | Auth middleware fails; no secure requests can be authenticated. |
| `ENVIRONMENT` | Manual Configuration | `development`, `production`, `test` | May expose stack traces in prod if not set correctly. Controls logging verbosity. |
| `ALLOWED_ORIGINS` | Manual Configuration | `http://localhost:3000,https://myapp.netlify.app` | CORS errors block the frontend from accessing the backend API. |
| `API_KEY_WEB` | Firebase Console (For testing only) | `AIzaSy...` | Test scripts cannot generate custom test tokens. Not needed in production. |

## Section 6 — API Endpoints

### `GET /`
- **Auth Required**: No
- **Request Format**: None
- **Success Response Schema**: `{"status": "ok", "message": "Job Concierge API is running."}`
- **Error Codes**: `500 Internal Server Error` (Server down or misconfigured)

### `GET /health/deep`
- **Auth Required**: No
- **Request Format**: None
- **Success Response Schema**: `{"status": "ok", "services": {"gemini": "connected", "firebase": "configured"}}`
- **Error Codes**: `503 Service Unavailable` (If Gemini API or Firebase is unreachable)

### `POST /api/v1/generate`
- **Auth Required**: Yes (Bearer Token in Header)
- **Request Format**: `multipart/form-data`
  - `Headers`: `Authorization: Bearer <Firebase_ID_Token>`
  - `Body`:
    - `resume_file`: File upload (application/pdf, text/plain)
    - `job_url`: String (Valid HTTP/HTTPS URL)
- **Success Response Schema**:
  ```json
  {
    "request_id": "uuid-string",
    "resume": "Markdown formatted tailored resume...",
    "cover_letter": "Markdown formatted cover letter...",
    "job_title_detected": "Software Engineer"
  }
  ```
- **Error Codes**:
  - `400 Bad Request`: Invalid file format, invalid URL, or missing fields.
  - `401 Unauthorized`: Missing, expired, or invalid Firebase token.
  - `403 Forbidden`: Token is valid but user lacks permissions (if roles are implemented).
  - `422 Unprocessable Entity`: Validation error from Pydantic (e.g., malformed URL string).
  - `429 Too Many Requests`: Rate limit exceeded (SlowAPI).
  - `500 Internal Server Error`: Pipeline failure, Gemini API error, scraping failure.

## Section 7 — Middleware Stack

**Execution Order:**
1. **Request ID Middleware (`request_id.py`)**: Runs *first*. Injects a `X-Request-ID` UUID into the request state. **Why:** So that every subsequent log and error response can be tagged with this ID for tracing. If out of order, errors thrown in earlier middlewares won't have an ID.
2. **Security Headers Middleware (`security_headers.py`)**: Adds `Strict-Transport-Security`, `X-Content-Type-Options`, etc. to the response. **Why:** Protects against basic web vulnerabilities early in the cycle.
3. **Rate Limiter (SlowAPI)**: Checks if the IP/User has exceeded the request quota. **Why:** Prevents spam and resource exhaustion before expensive auth or file parsing happens.
4. **Firebase Auth Middleware (`firebase_auth.py`)**: Extracts the Bearer token and verifies it against Firebase Admin. **Why:** Must happen before route handling to protect the core business logic. If placed after the route handler, unauthenticated users could trigger expensive LLM generation.

## Section 8 — Agent Pipeline

### 1. Scraper Agent
- **Input parameters**: `job_url` (string)
- **Step by step**: Validates URL -> Fetches HTML via `httpx` -> Parses text with BeautifulSoup -> Cleans text.
- **Gemini prompt**: (Optional/Fallback) "Extract the core job title, requirements, and responsibilities from the following raw text: {text}"
- **Output format**: `{"title": "...", "requirements": [...], "company": "...", "raw_context": "..."}`
- **Error handling**: Retries via `tenacity` on timeout. Returns an empty structured response or fails gracefully with a 400 if the URL is blocked/unreachable.
- **Guardrails**: Blocked domains list (e.g., localhost, internal IPs). Sanitization of the URL.

### 2. Resume Agent
- **Input parameters**: `parsed_resume_text` (string), `scraper_output` (dict)
- **Step by step**: Reads base resume -> Identifies matching skills from scraper output -> Rewrites bullet points to highlight matches -> Formats in ATS-friendly Markdown.
- **Gemini prompt**: "You are an expert ATS-resume writer. Take the user's resume below and tailor it to the provided job description. Emphasize matching skills. Do not invent experience. Output pure Markdown."
- **Output format**: Markdown string.
- **Error handling**: Catches GenAI API errors. Falls back to original resume text if generation fails entirely.
- **Guardrails**: Anti-hallucination validation—ensures no new companies or degrees were invented that weren't in the base resume.

### 3. Cover Letter Agent
- **Input parameters**: `parsed_resume_text` (string), `scraper_output` (dict)
- **Step by step**: Analyzes user's tone from resume -> Drafts a professional, persuasive cover letter targeting the specific company and role.
- **Gemini prompt**: "Write a professional, compelling 3-paragraph cover letter for the role of {title} at {company} using the following user resume. Focus on the value the user brings."
- **Output format**: Markdown string.
- **Error handling**: Retries on generation failure.
- **Guardrails**: Length limits, formatting checks to ensure no placeholder tags (e.g., "[Insert Name Here]") are left behind.

## Section 9 — Error Handling System

- **ErrorResponse Schema**:
  ```json
  {
    "error": {
      "code": 401,
      "type": "Unauthorized",
      "message": "Invalid authentication credentials.",
      "request_id": "123e4567-e89b-12d3-a456-426614174000",
      "timestamp": "2026-06-22T22:03:28Z"
    }
  }
  ```
- **HTTP Status Codes Triggered**:
  - `400`: File too large, unreadable PDF, invalid URL.
  - `401`: Missing Token, Expired Token.
  - `422`: FastAPI validation failure (missing form fields).
  - `429`: SlowAPI limits hit.
  - `500`: Unhandled exceptions, Gemini timeouts.
- **Threaded `request_id`**: The Request ID middleware attaches a UUID to `request.state.request_id`. Global exception handlers extract this from the `request` object and inject it into the `ErrorResponse`.
- **Stack Trace Prevention**: Custom exception handlers catch global `Exception`. In production (`ENVIRONMENT!=development`), the handler logs the stack trace internally via Loguru but returns a generic "Internal Server Error" message to the client.

## Section 10 — Security Features

- **Firebase token verification**: Ensures every request to protected routes comes from a logged-in user with a cryptographically signed JWT.
- **Rate limiting**: `slowapi` enforces limits (e.g., 5 requests/minute per user/IP) to prevent abuse of the expensive Gemini API.
- **Input sanitization**: `bleach` is used to strip HTML/scripts from resume text and URL inputs. URL validation ensures the scheme is exactly `http` or `https`.
- **Blocked domains list**: Hardcoded list preventing the Scraper Agent from requesting internal networks (SSRF prevention) or malicious sites.
- **HTML sanitization with bleach**: Specifically configured to strip all tags from incoming text, preventing stored XSS in the generated documents.
- **Security response headers**: Added via middleware:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `X-XSS-Protection: 1; mode=block`
  - `Strict-Transport-Security: max-age=31536000; includeSubDomains`
  - `Content-Security-Policy: default-src 'self'`
  - `Referrer-Policy: strict-origin-when-cross-origin`
  - `Permissions-Policy: geolocation=(), microphone=()`
- **Anti-hallucination validation**: Post-processing step on the LLM output verifying that critical facts (dates, degrees, past employers) align with the source resume.

## Section 11 — Complete Phase History

Phase 1 — FastAPI Scaffold
  Status: ✅ COMPLETE
  What was built: Project structure, basic endpoints, config loading, and logging setup.
  Key files created: `main.py`, `core/config.py`, `core/logger.py`, `routers/generate.py`.
  How to verify: `python3 -m uvicorn main:app --reload` -> visit `http://localhost:8000/`.

Phase 2 — ADK Agent Pipeline
  Status: ✅ COMPLETE
  What was built: The core AI logic with Scraper, Resume, and Cover Letter agents using Gemini 2.5 Flash.
  Key files created: `agents/orchestrator.py`, `agents/scraper_agent.py`, `core/gemini_client.py`.
  How to verify: Ensure `GEMINI_API_KEY` is set and run `pytest tests/test_pipeline.py`.

Phase 3 — End-to-End Testing
  Status: ✅ COMPLETE
  What was verified: Full pipeline execution from route to orchestrator to LLM to response.

Phase 4 — Hardening
  Status: ✅ COMPLETE
  What was built: Middleware stack, rate limiting, and input sanitization.
  Key files created: `middleware/security_headers.py`, `core/rate_limiter.py`, `core/input_sanitizer.py`.
  How to verify: Issue a curl request and check headers using `curl -I http://localhost:8000/`.

Phase 5 — Error Handling & Security
  Status: ✅ COMPLETE
  What was built: Firebase auth middleware, unified ErrorResponse schemas, and Request ID tracing.
  Key files created: `middleware/firebase_auth.py`, `middleware/request_id.py`, `core/error_handlers.py`.
  How to verify: Send unauthenticated POST to `/api/v1/generate` and observe 401 with full schema.

Phase 6 — Render.com Deployment
  Status: ⬅ NOT STARTED
  What needs to be done:
  1. Commit and push current branch to GitHub.
  2. Create Web Service in Render.com linked to repository.
  3. Define start command (`uvicorn main:app --host 0.0.0.0 --port 10000`).
  4. Inject `GEMINI_API_KEY` and `FIREBASE_SERVICE_ACCOUNT_JSON` via Render dashboard.
  5. Deploy and verify live health check.

## Section 12 — Known Issues & Fixes Applied

Issue 1 — Deprecated Gemini model
  Error: `google.api_core.exceptions.NotFound: 404 models/gemini-1.5-flash not found`
  Root cause: Using legacy model identifiers in the new ADK.
  Fix applied: Updated `model_name` from `gemini-1.5-flash` to `gemini-2.5-flash` across all agent modules.
  Verify fix: `grep -r "gemini-1.5" . --include="*.py"` (should return nothing).

Issue 2 — .env not loading
  Error: `pydantic_core._core_utils.ValidationError: missing GEMINI_API_KEY`
  Root cause: Python dotenv wasn't automatically finding the `.env` file from the working directory in scripts.
  Fix applied: Explicitly passed `dotenv_path='/media/spidey/program/capstone/jobseeker_backend/.env'` to `load_dotenv()`.
  Verify fix: Run test python script printing `os.getenv('GEMINI_API_KEY')`.

Issue 3 — curl 000 status codes
  Error: curl returned `HTTP 000` with connection refused.
  Root cause: The backend Uvicorn server was not running while the test curl commands were executed.
  Fix applied: Started Uvicorn server in a separate terminal before executing curl commands.

Issue 4 — load_dotenv AssertionError in heredoc
  Error: `AssertionError` raised when attempting to run a multi-line python script via zsh heredoc because `load_dotenv` failed.
  Root cause: Subshells and heredocs in zsh handled absolute paths differently or had missing imports.
  Fix applied: Used a standard `python3 -c "..."` string with exact imports and explicit absolute `dotenv_path`.

## Section 13 — How to Run Everything

#### Start the server
```bash
python3 -m uvicorn main:app --reload --port 8000
```

#### Run the test suite
```bash
python3 -m pytest tests/ -v --asyncio-mode=auto
```

#### Install packages
```bash
pip install -r requirements.txt --break-system-packages
```

#### Get a Firebase test token
```python
import requests
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='/media/spidey/program/capstone/jobseeker_backend/.env')
API_KEY = os.getenv("API_KEY_WEB")
payload = {"returnSecureToken": True}
url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}"

response = requests.post(url, json=payload)
print(response.json().get("idToken"))
```

#### Exchange custom token for ID token
```bash
curl -s -X POST "https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key=YOUR_API_KEY_WEB" \
  -H "Content-Type: application/json" \
  -d '{"token":"<GENERATED_CUSTOM_TOKEN>","returnSecureToken":true}'
```

#### Run all curl tests
```bash
# Basic Health
curl -s http://localhost:8000/ | python3 -m json.tool

# Unauthorized Pipeline Check
curl -s -X POST http://localhost:8000/api/v1/generate \
  -F "resume_file=@/tmp/test_resume.txt;type=text/plain" \
  -F "job_url=https://boards.greenhouse.io/anthropic" \
  | python3 -m json.tool
```

## Section 14 — What We Just Did (Session Summary)

- Server was started successfully.
- Backend tested with curl.
- 401 response confirmed working with full ErrorResponse schema.
- `request_id` confirmed present in error response.
- `timestamp` confirmed in ISO format.
- Firebase token generation attempted.
- `load_dotenv` AssertionError encountered in zsh heredoc.
- Fixed by passing explicit `dotenv_path`.
- All Phase 1 through Phase 5 backend features confirmed as implemented and functioning.

## Section 15 — Exact Next Steps

Step 1 — Generate Firebase custom token using fixed script
- **Why:** To authenticate our curl requests against the secure endpoints.
- **Command:** Run the Python script from Section 13 using the `API_KEY_WEB` from `.env`.
- **Success:** Output of a long JWT string.

Step 2 — Exchange custom token for real ID token
- **Why:** Firebase requires an ID token (not just a custom token) for the Bearer auth header.
- **Command:** Run the `signInWithCustomToken` curl command from Section 13.
- **Success:** JSON response containing `idToken`.

Step 3 — Export FIREBASE_TOKEN as shell variable
- **Why:** To easily pass the token in subsequent curl commands without pasting a massive string.
- **Command:** `export FIREBASE_TOKEN="your_idToken_here"`
- **Success:** `echo $FIREBASE_TOKEN` prints the token.

Step 4 — Run full curl test suite with real token
- **Why:** Verify the complete pipeline end-to-end with authentication bypassed.
- **Command:** Run the "Full pipeline test" from Section 16.
- **Success:** Returns a 200 OK with `resume` and `cover_letter` in JSON.

Step 5 — Run pytest suite and confirm all tests pass
- **Why:** Ensure no regressions were introduced during hardening.
- **Command:** `python3 -m pytest tests/ -v --asyncio-mode=auto`
- **Success:** All tests show green `PASSED`.

Step 6 — Begin Phase 6 Render.com deployment
- **Why:** Move the backend from local environment to the cloud.
- **Command:** `git add . && git commit -m "Complete local backend" && git push`
- **Success:** Code is visible on GitHub.

Step 7 — Push backend to GitHub
- **Why:** Required for Render CI/CD.
- **Command:** `git push origin alpha`
- **Success:** Branch `alpha` is updated.

Step 8 — Connect GitHub to Render.com
- **Why:** To deploy the web service automatically.
- **Command:** Use Render Dashboard UI -> New Web Service -> Select Repo.
- **Success:** Render begins the build process.

Step 9 — Set environment variables in Render dashboard
- **Why:** The cloud service needs API keys since `.env` is ignored.
- **Command:** Copy keys from local `.env` to Render Environment Variables settings.
- **Success:** Build and start processes succeed.

Step 10 — Lock CORS to Netlify frontend URL
- **Why:** Security measure to prevent other websites from using our backend.
- **Command:** Update `ALLOWED_ORIGINS` in Render config to point to the frontend URL.
- **Success:** Backend only accepts requests from the specified origin.

Step 11 — Test live API endpoint
- **Why:** Verify the cloud deployment works exactly like local.
- **Command:** `curl -s https://<your-render-url>/`
- **Success:** Returns the health check JSON.

Step 12 — Begin frontend Phase 4 dashboard UI
- **Why:** Connect the Next.js client to the newly deployed backend.
- **Command:** Change directories to the frontend and start development.
- **Success:** Frontend UI can hit the backend endpoints successfully.

## Section 16 — Quick Reference Card

**Server commands:**
  Start:   `python3 -m uvicorn main:app --reload --port 8000`
  Stop:    `CTRL+C`
  Test:    `python3 -m pytest tests/ -v --asyncio-mode=auto`

**Health checks:**
  Basic:   `curl -s http://localhost:8000/ | python3 -m json.tool`
  Deep:    `curl -s http://localhost:8000/health/deep | python3 -m json.tool`
  Headers: `curl -s -I http://localhost:8000/`

**Auth guard test (no token — must return 401):**
```bash
curl -s -X POST http://localhost:8000/api/v1/generate \
  -F "resume_file=@/tmp/test_resume.txt;type=text/plain" \
  -F "job_url=https://boards.greenhouse.io/anthropic" \
  | python3 -m json.tool
```

**Full pipeline test (requires FIREBASE_TOKEN set):**
```bash
curl -s -X POST http://localhost:8000/api/v1/generate \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -F "resume_file=@/tmp/test_resume.txt;type=text/plain" \
  -F "job_url=https://boards.greenhouse.io/anthropic" \
  | python3 -m json.tool
```

**Package install:**
  `pip install -r requirements.txt --break-system-packages`

**Grep for deprecated model (must return nothing):**
  `grep -r "gemini-1.5" . --include="*.py"`

**Check .env loads:**
```python
python3 -c "
from dotenv import load_dotenv
load_dotenv(dotenv_path='/media/spidey/program/capstone/jobseeker_backend/.env')
import os
print('GEMINI:', bool(os.getenv('GEMINI_API_KEY')))
print('FIREBASE:', bool(os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')))
"
```
