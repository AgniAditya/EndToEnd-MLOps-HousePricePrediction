# Dockerfile for House Price Prediction API
# This file creates a containerized environment for running the FastAPI application
# It ensures consistent deployment across different environments

# Use official Python runtime as a parent image
# Python 3.11-slim provides a good balance between size and functionality
FROM python:3.11-slim

# Set working directory inside the container
# All subsequent commands will be executed from this directory
WORKDIR /app

# Install system dependencies for downloading and extracting files
# These packages are needed to download the model files during build
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
# This step is done before copying the rest of the app for better Docker layer caching
COPY requirements.txt .

# Upgrade pip to the latest version for better package compatibility
RUN pip install --upgrade pip

# Install all Python dependencies listed in requirements.txt
# --no-cache-dir reduces the final image size by not caching downloaded packages
RUN pip install --no-cache-dir -r requirements.txt

# Download and extract model files during build
# This ensures the trained model is available when the container starts
RUN mkdir -p models && \
    curl --fail -L -o models.zip https://github.com/AgniAditya/EndToEnd-MLOps-HousePricePrediction/releases/download/v1.0/models.zip && \
    unzip -j models.zip -d models && \
    rm models.zip

# Copy the rest of the application files
# This includes the FastAPI app, pipeline code, and other necessary files
COPY . .

# Expose the port that FastAPI will run on
# This tells Docker which port the application will use
EXPOSE 8000

# Command to run the FastAPI app with uvicorn
# --host 0.0.0.0 allows external connections to the container
# --port 8000 specifies the port to run the application on
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]