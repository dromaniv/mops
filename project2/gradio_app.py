import gradio as gr
import torch
import numpy as np
import pickle
from pathlib import Path


# Load model
model_dir = Path(__file__).parent / 'model'

with open(model_dir / 'model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

with open(model_dir / 'scaler.pkl', 'rb') as f:
    scaler_params = pickle.load(f)

model = torch.jit.load(model_dir / 'lol_model.pt')
model.eval()

print("Model loaded")


def predict_match_outcome(
    blue_wards_placed, blue_wards_destroyed, blue_first_blood,
    blue_kills, blue_deaths, blue_assists,
    blue_elite_monsters, blue_dragons, blue_heralds, blue_towers,
    blue_total_gold, blue_avg_level, blue_total_exp,
    blue_minions, blue_jungle_minions,
    blue_gold_diff, blue_exp_diff,
    blue_cs_per_min, blue_gold_per_min,
    red_wards_placed, red_wards_destroyed, red_first_blood,
    red_kills, red_deaths, red_assists,
    red_elite_monsters, red_dragons, red_heralds, red_towers,
    red_total_gold, red_avg_level, red_total_exp,
    red_minions, red_jungle_minions,
    red_gold_diff, red_exp_diff,
    red_cs_per_min, red_gold_per_min
):
    """Make prediction"""
    
    # Build feature array
    features = np.array([
        blue_wards_placed, blue_wards_destroyed, blue_first_blood,
        blue_kills, blue_deaths, blue_assists,
        blue_elite_monsters, blue_dragons, blue_heralds, blue_towers,
        blue_total_gold, blue_avg_level, blue_total_exp,
        blue_minions, blue_jungle_minions,
        blue_gold_diff, blue_exp_diff,
        blue_cs_per_min, blue_gold_per_min,
        red_wards_placed, red_wards_destroyed, red_first_blood,
        red_kills, red_deaths, red_assists,
        red_elite_monsters, red_dragons, red_heralds, red_towers,
        red_total_gold, red_avg_level, red_total_exp,
        red_minions, red_jungle_minions,
        red_gold_diff, red_exp_diff,
        red_cs_per_min, red_gold_per_min
    ], dtype=np.float32)
    
    # Normalize
    features = (features - scaler_params['mean']) / scaler_params['scale']
    
    # Predict
    input_tensor = torch.FloatTensor(features).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()

    return {"Blue wins": probability, "Red wins": 1 - probability}


# Create UI

with gr.Blocks(title="LoL Win Prediction") as demo:
    gr.Markdown("""
    <h1 style='text-align:center;'>League of Legends Win Prediction</h1>
    """)

    with gr.Row():
        with gr.Column(elem_id="blue-col"):
            gr.Markdown("### Blue Team")
            gr.Markdown("**Vision**")
            blue_wards_placed = gr.Number(label="Wards Placed", value=28)
            blue_wards_destroyed = gr.Number(label="Wards Destroyed", value=2)
            gr.Markdown("**Combat**")
            blue_first_blood = gr.Number(label="First Blood (1/0)", value=1, minimum=0, maximum=1)
            blue_kills = gr.Number(label="Kills", value=9)
            blue_deaths = gr.Number(label="Deaths", value=6)
            blue_assists = gr.Number(label="Assists", value=11)
            gr.Markdown("**Objectives**")
            blue_elite_monsters = gr.Number(label="Elite Monsters", value=1)
            blue_dragons = gr.Number(label="Dragons", value=1)
            blue_heralds = gr.Number(label="Heralds", value=0)
            blue_towers = gr.Number(label="Towers", value=0)
            gr.Markdown("**Economy**")
            blue_total_gold = gr.Number(label="Total Gold", value=17210)
            blue_avg_level = gr.Number(label="Avg Level", value=6.8)
            blue_total_exp = gr.Number(label="Total XP", value=17039)
            blue_minions = gr.Number(label="Minions", value=197)
            blue_jungle_minions = gr.Number(label="Jungle Minions", value=30)
            gr.Markdown("**Diff**")
            blue_gold_diff = gr.Number(label="Gold Diff", value=643)
            blue_exp_diff = gr.Number(label="XP Diff", value=8)
            blue_cs_per_min = gr.Number(label="CS/min", value=19.7)
            blue_gold_per_min = gr.Number(label="Gold/min", value=1721.0)
        with gr.Column(elem_id="red-col"):
            gr.Markdown("### Red Team")
            gr.Markdown("**Vision**")
            red_wards_placed = gr.Number(label="Wards Placed", value=15)
            red_wards_destroyed = gr.Number(label="Wards Destroyed", value=0)
            gr.Markdown("**Combat**")
            red_first_blood = gr.Number(label="First Blood (1/0)", value=0, minimum=0, maximum=1)
            red_kills = gr.Number(label="Kills", value=6)
            red_deaths = gr.Number(label="Deaths", value=9)
            red_assists = gr.Number(label="Assists", value=8)
            gr.Markdown("**Objectives**")
            red_elite_monsters = gr.Number(label="Elite Monsters", value=0)
            red_dragons = gr.Number(label="Dragons", value=0)
            red_heralds = gr.Number(label="Heralds", value=0)
            red_towers = gr.Number(label="Towers", value=0)
            gr.Markdown("**Economy**")
            red_total_gold = gr.Number(label="Total Gold", value=16567)
            red_avg_level = gr.Number(label="Avg Level", value=6.6)
            red_total_exp = gr.Number(label="Total XP", value=17031)
            red_minions = gr.Number(label="Minions", value=240)
            red_jungle_minions = gr.Number(label="Jungle Minions", value=28)
            gr.Markdown("**Diff**")
            red_gold_diff = gr.Number(label="Gold Diff", value=-643)
            red_exp_diff = gr.Number(label="XP Diff", value=-8)
            red_cs_per_min = gr.Number(label="CS/min", value=24.0)
            red_gold_per_min = gr.Number(label="Gold/min", value=1656.7)


    with gr.Row():
        predict_btn = gr.Button("Predict", variant="primary", size="lg")

    with gr.Row():
        probs = gr.Label(label="Probabilities", num_top_classes=2, elem_id="probs-fullwidth")

    predict_btn.click(
        fn=predict_match_outcome,
        inputs=[
            blue_wards_placed, blue_wards_destroyed, blue_first_blood,
            blue_kills, blue_deaths, blue_assists,
            blue_elite_monsters, blue_dragons, blue_heralds, blue_towers,
            blue_total_gold, blue_avg_level, blue_total_exp,
            blue_minions, blue_jungle_minions,
            blue_gold_diff, blue_exp_diff, blue_cs_per_min, blue_gold_per_min,
            red_wards_placed, red_wards_destroyed, red_first_blood,
            red_kills, red_deaths, red_assists,
            red_elite_monsters, red_dragons, red_heralds, red_towers,
            red_total_gold, red_avg_level, red_total_exp,
            red_minions, red_jungle_minions,
            red_gold_diff, red_exp_diff, red_cs_per_min, red_gold_per_min
        ],
        outputs=[probs]
    )

    # Add custom CSS for colored backgrounds
    gr.HTML("""
    <style>
    #blue-col { background: #0d47a1; border-radius: 8px; padding: 8px 12px 8px 12px; }
    #red-col  { background: #b71c1c; border-radius: 8px; padding: 8px 12px 8px 12px; }
    #blue-header { margin-bottom: 0.5em; }
    #red-header { margin-bottom: 0.5em; }
    #probs-fullwidth { width: 100% !important; }
    </style>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
