#  Model Training

This project trains a binary classification model to predict League of Legends match outcomes using PyTorch Lightning.

**Model Details:**

- Architecture: MLP (Multi-Layer Perceptron)
- Dataset: 9,879 Diamond-ranked matches
- Features: 38 early-game statistics from first 10 minutes
- Target: Binary (blue wins: 0 or 1)
- Tools: PyTorch Lightning, Weights & Biases, Optuna

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset

1. Visit: https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min
2. Download `high_diamond_ranked_10min.csv` to `data/` folder

### 3. Set up WandB (optional)

```bash
wandb login
```

For offline mode:

```python
import os
os.environ['WANDB_MODE'] = 'offline'
```

### 4. Run notebook

Open `lol_win_prediction.ipynb` and run all cells.
