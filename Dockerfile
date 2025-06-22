FROM python:3.10-slim

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages for LightFM
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY r.txt .
RUN pip install --no-cache-dir -r r.txt

# Copy only necessary code
COPY app.py .
COPY training/ training/
COPY recommendation/ recommendation/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
