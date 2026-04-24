FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV PORT=7860

# Set the working directory in the container
WORKDIR /app

# Minimal system library needed by some scientific Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
	libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.runtime.txt .

# Install the dependencies, then override torch with a CUDA-enabled wheel.
RUN python -m pip install --upgrade pip && \
	grep -v '^torch$' requirements.runtime.txt > /tmp/requirements.no-torch.txt && \
    pip install accelerate && \
	pip install -r /tmp/requirements.no-torch.txt && \
	pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0

# Copy the rest of the application code into the container
COPY faiss_yelp/ faiss_yelp/
COPY prompt.py .
COPY rag_retrival.py .
COPY app.py .

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port that the app will run on
EXPOSE 7860
# Command to run the application
CMD ["python", "app.py"]