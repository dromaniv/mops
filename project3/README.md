# Project 3: Dockerized Gradio Application

## Overview
This project provides a **Dockerized Gradio application** for the League of Legends win prediction model trained in Project 1. The application is containerized for easy deployment and hosting.

## Features
- 🐳 **Docker containerization** for consistent deployment
- 🚀 **Docker Compose** for easy orchestration
- 🌐 **Web UI** with Gradio for interactive predictions
- 📦 **Self-contained** - includes model and all dependencies
- ♻️ **Auto-restart** capabilities with health checks

## Project Structure
```
project3/
├── app.py                  # Gradio application
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
├── .dockerignore         # Docker ignore patterns
├── README.md             # This file
└── model/                # Model artifacts
    ├── lol_model.pt
    ├── model_metadata.pkl
    ├── scaler.pkl
    ├── feature_names.pkl
    └── sample_data.pkl
```

## Requirements
- Docker 20.10+
- Docker Compose 2.0+

## Quick Start

### Option 1: Using Docker Compose (Recommended)
```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Option 2: Using Docker directly
```bash
# Build the image
docker build -t lol-prediction-gradio .

# Run the container
docker run -d -p 7860:7860 --name lol-prediction-app lol-prediction-gradio

# View logs
docker logs -f lol-prediction-app

# Stop the container
docker stop lol-prediction-app
docker rm lol-prediction-app
```

## Access the Application
Once running, access the Gradio interface at:
- Local: http://localhost:7860
- Network: http://YOUR_IP:7860

## Model Information
- **Model Type:** Multi-layer Perceptron (MLP)
- **Framework:** PyTorch + PyTorch Lightning
- **Input:** 38 early-game features (first 10 minutes)
- **Output:** Binary classification (Blue team wins/loses)
- **Accuracy:** ~73% on test set

## Usage

### Making Predictions
1. Enter game statistics for both Blue and Red teams
2. The app automatically calculates derived metrics (Gold Diff, XP Diff, CS/min, Gold/min)
3. Click "Predict" to get win probability for each team

### Sample Input
Default values are provided in the interface representing a real game scenario:
- Blue Team: 9 kills, 1 dragon, 17,210 gold
- Red Team: 6 kills, 0 dragons, 16,567 gold

## Docker Configuration

### Environment Variables
- `GRADIO_SERVER_NAME`: Server bind address (default: 0.0.0.0)
- `GRADIO_SERVER_PORT`: Server port (default: 7860)

### Ports
- `7860`: Gradio web interface

### Health Check
The container includes a health check that monitors the application status:
- Interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3
- Start period: 40 seconds

## Development

### Local Testing (without Docker)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### Rebuilding the Image
```bash
# Rebuild after code changes
docker-compose up -d --build

# Or with Docker
docker build -t lol-prediction-gradio . --no-cache
```

## Deployment

### Cloud Platforms
This Docker container can be deployed on:
- **Google Cloud Run**
- **AWS ECS/Fargate**
- **Azure Container Instances**
- **DigitalOcean App Platform**
- **Heroku Container Registry**
- **Railway**
- **Render**

### Example: Deploy to Cloud Run
```bash
# Build for Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/lol-prediction

# Deploy
gcloud run deploy lol-prediction \
  --image gcr.io/PROJECT_ID/lol-prediction \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 7860
```

### Example: Deploy to Railway
1. Push to GitHub repository
2. Create new project on Railway
3. Connect GitHub repository
4. Railway auto-detects Dockerfile
5. Deploy automatically

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs

# Verify port availability
lsof -i :7860
```

### Model loading errors
```bash
# Verify model files exist
docker exec lol-prediction-gradio ls -la /app/model/
```

### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8080:7860"  # Use 8080 instead
```

## Performance Optimization

### Image Size Reduction
The current image uses `python:3.10-slim` for a smaller footprint:
- Base image: ~150MB
- Final image: ~1.5GB (includes PyTorch)

### Memory Requirements
- Minimum: 512MB RAM
- Recommended: 1GB+ RAM

## Security

### Production Considerations
- Change default ports if needed
- Use environment variables for sensitive data
- Enable HTTPS with reverse proxy (nginx/traefik)
- Implement rate limiting
- Add authentication if required

### Example nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring

### Container Stats
```bash
# View resource usage
docker stats lol-prediction-gradio

# View detailed info
docker inspect lol-prediction-gradio
```

### Logs
```bash
# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t lol-prediction .
      - name: Push to registry
        run: |
          docker tag lol-prediction ${{ secrets.REGISTRY }}/lol-prediction
          docker push ${{ secrets.REGISTRY }}/lol-prediction
```

## Related Projects
- **Project 1:** Model training with PyTorch Lightning
- **Project 2:** BentoML API service and Python client

---

**Authors:** Dmytro Romaniv 151958 & Patryk Maciejewski 151960

