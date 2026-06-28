# Use official lightweight Python image
FROM python:3.12-slim

# Set the directory inside the container
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install all the packages from your newly generated file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your local project files into the container
COPY . .

# Expose a default port for local development
EXPOSE 8000

# Launch the FastAPI app matching your environment settings.
# We use the shell form of CMD to dynamically read Render's $PORT variable (defaults to 8000 locally).
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
