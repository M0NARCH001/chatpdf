# Stage 1: Build & Dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies (build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Final Image
FROM python:3.11-slim

WORKDIR /app

# Install any required OS libraries for the running app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install dependencies from wheels
RUN pip install --no-cache /wheels/*

# Copy project code
COPY . .

# Expose FastAPI and Gradio ports
EXPOSE 8000
EXPOSE 7860

# Environment variables
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0

# Start script
# We'll use a bash script or custom app.py to launch both 
# Or default to FastAPI if no command is passed
CMD ["python", "app.py"]
