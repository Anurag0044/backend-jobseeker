<div align="center">

# Forge-x: Autonomous Career Concierge (Backend API)

**Your intelligent, autonomous career advocate that automates and optimizes your job search.**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.138.1-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-1.3.11-326CE5.svg?style=flat)](https://www.langchain.com/)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7.svg?style=flat&logo=render&logoColor=white)](https://render.com/)

*A Kaggle AI Agents Intensive Vibe Coding Capstone Project (Concierge Agents Track)*
**Live Demo:** [Forge-x on Netlify](https://forge-x.netlify.app/) | **Backend API:** [Forge-x API on Render](https://one-forge-x.onrender.com)

</div>

---

## 🌟 The Vision: For Everyone

**What is Forge-x?**
Imagine having a highly skilled, 24/7 personal career assistant. Forge-x reads your resume, understands your unique career trajectory, and autonomously searches the live internet to find the absolute best job opportunities tailored specifically to you. 

**The Problem It Solves**
The modern job hunt is exhausting. Traditional job boards rely on basic keyword matching, often missing the nuance of your actual experience. You spend hours filtering through irrelevant roles.

**The Forge-x Solution**
Instead of you searching for jobs, Forge-x searches for you. By leveraging advanced Artificial Intelligence, it doesn't just look at keywords; it performs **semantic reasoning**. It understands the *meaning* behind your experience and the *expectations* of a job posting, generating a realistic fit score and explaining exactly *why* you are a great match for a recommended role. And it does all this while keeping your personal data completely secure.

---

## 💻 Under the Hood: For Developers

Forge-x is powered by a robust, modern backend architecture designed for performance, scalability, and complex AI orchestration.

### 🏗️ Technical Architecture & Features

*   **Agentic Workflow (LangGraph):** The core of Forge-x isn't just a simple LLM call. It uses an autonomous agent architecture that can reason, plan, and execute multi-step tool calls to gather data, parse profiles, and search the web.
*   **Real-Time Streaming (SSE):** The backend utilizes Server-Sent Events (SSE) to stream the AI agent's internal "thought process," reasoning stages, and tool execution directly to the frontend. This provides the user with transparent, real-time insight into what the AI is doing.
*   **Secure Profile Parsing:** We safely parse user profiles and resumes (using PyPDF2 and custom logic) without leaking sensitive data to public search indices. User data is securely managed via Firebase Admin SDK.
*   **Live Web Intelligence:** Integrated with Tavily for autonomous web search, ensuring the agent retrieves the most current and highest-quality job postings from across the internet.
*   **High-Performance API:** Built on FastAPI and Uvicorn, offering asynchronous request handling, automatic OpenAPI documentation, and robust data validation via Pydantic.

### 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Web Framework** | Python 3.10+, FastAPI, Uvicorn, Pydantic |
| **AI Orchestration**| LangChain, LangGraph |
| **LLMs** | Google Gemini, NVIDIA AI Endpoints |
| **Data & Auth** | Firebase Admin SDK, Firestore |
| **Tools** | Tavily (Web Search), PyPDF2 (Resume Parsing) |
| **Deployment** | Docker, Render |

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have the following installed:
*   [Python 3.10+](https://www.python.org/downloads/)
*   [Docker](https://www.docker.com/get-started) (Optional, for containerized deployment)

**Required API Keys:**
*   Google Gemini API Key
*   NVIDIA AI API Key
*   Tavily API Key
*   Firebase Admin Service Account JSON file

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anurag0044/backend-jobseeker.git
   cd backend-jobseeker
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables:**
   Create a `.env` file in the root directory and add the required API keys and service account JSON file path.

   ```   
## ⚙️ Usage

### Running Locally

Start the FastAPI development server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*(Note: If your entry point is different, adjust `app.main:app` accordingly, e.g., `main:app`)*

### Interactive API Documentation

Once the server is running, FastAPI automatically generates interactive documentation. Navigate to:
*   **Swagger UI:** `http://localhost:8000/docs`
*   **ReDoc:** `http://localhost:8000/redoc`

### Docker Deployment

To build and run the application using Docker, ensuring environment parity:

```bash
# Build the image
docker build -t job-concierge-api .

# Run the container (make sure your .env file is configured)
docker run -p 8000:8000 --env-file .env job-concierge-api
```

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
