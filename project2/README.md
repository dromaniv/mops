# What This Is

This project takes the LoL win prediction model from Project 1 and serves it as a REST API using BentoML.

**Model Stats:**

- Architecture: MLP (Multi-Layer Perceptron)
- Input: 38 features from first 10 minutes of game
- Test Accuracy: ~73%

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Export model from notebook

Run the last cell in `../project1/lol_win_prediction.ipynb` to export the model files.

### 3. Start service

```bash
bentoml serve service:LolWinPredictionService
```

API runs at: `http://localhost:3000`

## API Endpoints

- `POST /health` - Check if service is running
- `POST /predict` - Predict single match
- `POST /predict_batch` - Predict multiple matches

## Usage Examples

### Health Check

**Python:**

```python
import requests
response = requests.post("http://localhost:3000/health", json={})
print(response.json())
```

### Single Prediction

**Python:**

```python
import requests

data = {
    "blueWardsPlaced": 28,
    "blueKills": 9,
    "blueGoldDiff": 643,
    # ... (38 features total)
}

response = requests.post("http://localhost:3000/predict", json=data)
result = response.json()

print(f"Winner: {'Blue' if result['prediction'] == 1 else 'Red'}")
print(f"Probability: {result['probability']}")
```

### Run Example Clients

```bash
python python_client.py
```

## Web UI (Optional)

Start Gradio interface:

```bash
python gradio_app.py
```

Opens at: `http://localhost:7860`
