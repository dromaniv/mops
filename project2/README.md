# What This Is

This project takes the LoL win prediction model from Project 1 and serves it as a REST API using BentoML.

**Model Stats:**

- Architecture: MLP (Multi-Layer Perceptron)
- Input: 38 features from first 10 minutes of game
- Test Accuracy: ~73%

## Files

```
project2/
├── service.py              # BentoML API service
├── gradio_app.py           # Web UI for testing
├── requirements.txt        # Dependencies
├── bentofile.yaml          # BentoML config
├── test_setup.py           # Setup checker
├── quickstart.ps1          # Setup helper
├── model/                  # Model files (exported from notebook)
│   ├── lol_model.pt
│   ├── model_metadata.pkl
│   ├── feature_names.pkl
│   └── scaler.pkl
└── client_examples/        # API client examples
    ├── python_client.py
    └── powershell_client.ps1
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Export model from notebook

Run the last cell in `../project1/lol_win_prediction.ipynb` to export the model files.

### 3. Test setup

```bash
python test_setup.py
```

### 4. Start service

```bash
bentoml serve service:LolWinPredictionService
```

API runs at: `http://localhost:3000`

## API Endpoints

- `GET /health` - Check if service is running
- `POST /predict` - Predict single match
- `POST /predict_batch` - Predict multiple matches

## Usage Examples

### Health Check

**Python:**

```python
import requests
response = requests.get("http://localhost:3000/health")
print(response.json())
```

**PowerShell:**

```powershell
Invoke-RestMethod -Uri "http://localhost:3000/health"
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

**PowerShell:**

```powershell
$data = @{ blueKills = 9; redKills = 6; ... }
$result = Invoke-RestMethod -Uri "http://localhost:3000/predict" `
    -Method Post -Body ($data | ConvertTo-Json) `
    -ContentType "application/json"
```

### Run Example Clients

```bash
# Python
python client_examples/python_client.py

# PowerShell
.\client_examples\powershell_client.ps1
```

## Web UI (Optional)

Start Gradio interface:

```bash
python gradio_app.py
```

Opens at: `http://localhost:7860`
