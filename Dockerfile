FROM python:3.10-slim

# 1. Use a neutral working directory for the project
WORKDIR /code

# Install system dependencies like ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Copy and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 2. Copy your application code into a subdirectory
COPY ./app /code/app

# Expose the port the app runs on
EXPOSE 8000

# This command now runs from /code, so it can correctly find the 'app' module
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]