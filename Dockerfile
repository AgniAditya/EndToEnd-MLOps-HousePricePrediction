# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Download and extract model files during build
RUN mkdir -p models && \
    curl --fail -L -o models.zip https://github.com/AgniAditya/EndToEnd-MLOps-HousePricePrediction/releases/download/v1.0/models.zip && \
    unzip models.zip -d models && \
    rm models.zip

# Copy the rest of the app files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]