import requests


SERVICE_URL = "http://localhost:3000"


def check_health():
    """Check if service is running"""
    response = requests.post(f"{SERVICE_URL}/health", json={})
    response.raise_for_status()
    return response.json()


def predict_single(features):
    """Predict one match"""
    response = requests.post(
        f"{SERVICE_URL}/predict",
        json={"features": features},
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    return response.json()


def predict_batch(features_list):
    """Predict multiple matches"""
    response = requests.post(
        f"{SERVICE_URL}/predict_batch",
        json={"matches": features_list},
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    
    print("="*70)
    print("LoL Win Prediction API - Python Client")
    print("="*70)
    
    # Health check
    print("\n1. Health Check:")
    try:
        health = check_health()
        print(f"   Status: {health.get('status')}")
        print(f"   Model loaded: {health.get('model_loaded')}")
    except Exception as e:
        print(f"   Error: {e}")
        print("\n   Make sure service is running:")
        print("   bentoml serve service:LolWinPredictionService")
        exit(1)
    
    # Example match (Blue team ahead)
    match_data = {
        "blueWardsPlaced": 28,
        "blueWardsDestroyed": 2,
        "blueFirstBlood": 1,
        "blueKills": 9,
        "blueDeaths": 6,
        "blueAssists": 11,
        "blueEliteMonsters": 1,
        "blueDragons": 1,
        "blueHeralds": 0,
        "blueTowersDestroyed": 0,
        "blueTotalGold": 17210,
        "blueAvgLevel": 6.8,
        "blueTotalExperience": 17039,
        "blueTotalMinionsKilled": 197,
        "blueTotalJungleMinionsKilled": 30,
        "blueGoldDiff": 643,
        "blueExperienceDiff": 8,
        "blueCSPerMin": 19.7,
        "blueGoldPerMin": 1721.0,
        "redWardsPlaced": 15,
        "redWardsDestroyed": 0,
        "redFirstBlood": 0,
        "redKills": 6,
        "redDeaths": 9,
        "redAssists": 8,
        "redEliteMonsters": 0,
        "redDragons": 0,
        "redHeralds": 0,
        "redTowersDestroyed": 0,
        "redTotalGold": 16567,
        "redAvgLevel": 6.6,
        "redTotalExperience": 17031,
        "redTotalMinionsKilled": 240,
        "redTotalJungleMinionsKilled": 28,
        "redGoldDiff": -643,
        "redExperienceDiff": -8,
        "redCSPerMin": 24.0,
        "redGoldPerMin": 1656.7
    }
    
    # Single prediction
    print("\n2. Single Prediction:")
    print(f"   Blue: {match_data['blueKills']} kills, {match_data['blueTotalGold']} gold")
    print(f"   Red:  {match_data['redKills']} kills, {match_data['redTotalGold']} gold")
    
    try:
        result = predict_single(match_data)
        winner = 'Blue' if result['prediction'] == 1 else 'Red'
        print(f"\n   Prediction: {winner} wins")
        print(f"   Probability: {result['probability']:.4f}")
        print(f"   Confidence: {result['confidence']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Batch prediction
    print("\n3. Batch Prediction:")
        
    match_data_flip = {**match_data}

    # Dragon goes to red
    match_data_flip["blueDragons"] = 0
    match_data_flip["redDragons"] = 1
    match_data_flip["blueEliteMonsters"] = match_data_flip["blueDragons"] + match_data_flip["blueHeralds"]
    match_data_flip["redEliteMonsters"]  = match_data_flip["redDragons"] + match_data_flip["redHeralds"]

    # Red wins a couple kills in the fight
    match_data_flip["redKills"] += 2
    match_data_flip["blueDeaths"] += 2

    dragon_swing = 2000
    match_data_flip["redTotalGold"] += dragon_swing
    match_data_flip["blueGoldDiff"] = match_data_flip["blueTotalGold"] - match_data_flip["redTotalGold"]
    match_data_flip["redGoldDiff"]  = -match_data_flip["blueGoldDiff"]
            
    try:
        results = predict_batch([match_data, match_data_flip])
        for i, result in enumerate(results, 1):
            winner = 'Blue' if result['prediction'] == 1 else 'Red'
            print(f"\n   Match {i}:")
            print(f"     Winner: {winner}")
            print(f"     Probability: {result['probability']:.4f}")
            print(f"     Confidence: {result['confidence']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "="*70)
