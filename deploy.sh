#!/bin/bash
# Deploy docker container to Google CLoud Run

# set variables
IMAGE="chadhucey/yelp-rag-app:latest"
SERVICE_NAME="yelp-rag-service"
REGION="us-east4"
PORT="7860"

# Deploy to Google Cloud Run
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --execution-environment gen2 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --cpu 4 \
  --memory 16Gi \
  --concurrency 1 \
  --no-gpu-zonal-redundancy \
  --cpu-boost \
  --allow-unauthenticated \
  --port "$PORT"

# Get the deployed service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
  --platform managed \
  --region "$REGION" \
  --format="value(status.url)")

echo "Deployment complete!"
echo "Service name:  $SERVICE_NAME"
echo "Region:        $REGION"
echo "Public URL:    $SERVICE_URL"
