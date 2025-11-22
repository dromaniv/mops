"""
BentoML service for LoL win prediction.
"""

import bentoml
import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field


# Model architecture
class LolWinPredictorModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()


# Load model files
model_dir = Path(__file__).parent / 'model'
with open(model_dir / 'model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
with open(model_dir / 'feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
with open(model_dir / 'scaler.pkl', 'rb') as f:
    scaler_params = pickle.load(f)


# Input/output models
class MatchFeatures(BaseModel):
    blueWardsPlaced: int
    blueWardsDestroyed: int
    blueFirstBlood: int
    blueKills: int
    blueDeaths: int
    blueAssists: int
    blueEliteMonsters: int
    blueDragons: int
    blueHeralds: int
    blueTowersDestroyed: int
    blueTotalGold: int
    blueAvgLevel: float
    blueTotalExperience: int
    blueTotalMinionsKilled: int
    blueTotalJungleMinionsKilled: int
    blueGoldDiff: int
    blueExperienceDiff: int
    blueCSPerMin: float
    blueGoldPerMin: float
    redWardsPlaced: int
    redWardsDestroyed: int
    redFirstBlood: int
    redKills: int
    redDeaths: int
    redAssists: int
    redEliteMonsters: int
    redDragons: int
    redHeralds: int
    redTowersDestroyed: int
    redTotalGold: int
    redAvgLevel: float
    redTotalExperience: int
    redTotalMinionsKilled: int
    redTotalJungleMinionsKilled: int
    redGoldDiff: int
    redExperienceDiff: int
    redCSPerMin: float
    redGoldPerMin: float


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1=Blue wins, 0=Red wins")
    probability: float
    confidence: str


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class LolWinPredictionService:
    
    def __init__(self):
        model_path = model_dir / 'lol_model.pt'
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        self.feature_names = feature_names
        self.input_dim = metadata['input_dim']
        self.scaler_mean = scaler_params['mean']
        self.scaler_scale = scaler_params['scale']
        
        print(f"Model loaded: {self.input_dim} features")
    
    @bentoml.api
    def predict(self, features: MatchFeatures) -> PredictionResponse:
        """Predict match outcome"""
        
        # Convert to array
        feature_dict = features.dict()
        feature_array = np.array([feature_dict[name] for name in self.feature_names], dtype=np.float32)
        
        # Normalize
        feature_array = (feature_array - self.scaler_mean) / self.scaler_scale
        
        # Predict
        input_tensor = torch.FloatTensor(feature_array).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probability = torch.sigmoid(logits).item()
        
        # Get prediction and confidence
        prediction = 1 if probability > 0.5 else 0
        if probability > 0.7 or probability < 0.3:
            confidence = "High"
        elif probability > 0.6 or probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            confidence=confidence
        )
    
    @bentoml.api
    def predict_batch(self, features_list: List[MatchFeatures]) -> List[PredictionResponse]:
        """Predict multiple matches"""
        return [self.predict(features) for features in features_list]
    
    @bentoml.api
    def health(self) -> Dict[str, Any]:
        """Health check"""
        return {
            "status": "healthy",
            "model_loaded": True,
            "features": self.input_dim
        }
