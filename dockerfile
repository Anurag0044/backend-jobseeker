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

# Expose the local host port
EXPOSE 8000

# Launch the FastAPI app matching your local environment settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
