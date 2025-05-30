# Use an official lightweight Python image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt ./

# Install dependencies (prefer Poetry, fallback to pip)
RUN pip install --upgrade pip 
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
