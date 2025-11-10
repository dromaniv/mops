# League of Legends Win Prediction with PyTorch Lightning

This project implements a binary classification model to predict League of Legends match outcomes using PyTorch Lightning, with experiment tracking via Weights & Biases and hyperparameter optimization using Optuna.

## 🎯 Project Overview

**Problem:** Predict whether the blue team wins based on early-game statistics (first 10 minutes)

**Approach:**

- Multi-layer perceptron (MLP) for tabular data
- PyTorch Lightning for clean training code
- Weights & Biases for experiment tracking
- Optuna for automated hyperparameter optimization

**Dataset:** 9,879 Diamond-ranked League of Legends matches  
**Features:** 38 early-game statistics (gold, kills, objectives, vision, etc.)  
**Target:** Binary classification (blue wins: 0 or 1)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2. Download Dataset

1. Visit: https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min
2. Download and extract `high_diamond_ranked_10min.csv` to `data/` folder

### 3. Set Up WandB (Optional but Recommended)

```bash
wandb login
# Enter your API key from https://wandb.ai/authorize
```

For offline mode, add this to notebook:

```python
import os
os.environ['WANDB_MODE'] = 'offline'
```

### 4. Run the Notebook

Open `lol_win_prediction.ipynb` in Jupyter or VS Code and run all cells.
