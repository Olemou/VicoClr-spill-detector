import torch
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from pathlib import Path
import numpy as np
from datetime import datetime
from PIL import Image
import io

# ====================== CONFIG ======================
SAVE_FOLDER = Path("tsne_visualizations")
SAVE_FOLDER.mkdir(exist_ok=True)

VARIANTS = [
    {"name": "full",          "title": "All Components"},
    {"name": "no_uncertainty", "title": "Hard-Negative Weighting "},
    {"name": "no_weighting",      "title": "Original SupCon"},
    {"name": "uncertainty_curriculum_lr", "title": "Uncertainty Curriculum Learning"},
    {"name": "uncertainty_only", "title": "Only Uncertainty"}
]

BASE_DIR = Path("./VicoClr_new_version/benchmarking/tsne_output")

# ====================== LOAD DATA ======================
def load_tsne_data():
    dfs = []
    for variant in VARIANTS:
        candidates = [
            BASE_DIR / f"tsne_features_{variant['name']}.pt",
            BASE_DIR / f"tsne_{variant['name']}.pt",
            BASE_DIR / "tsne_features.pt",
        ]
        loaded = False
        for pt_path in candidates:
            if pt_path.exists():
                data_dict = torch.load(pt_path, map_location='cpu')
                data = data_dict["features_2d"].numpy()
                labels = data_dict["labels"].numpy()
                labels = labels.astype(int)

                mask = (labels == 0) | (labels == 1)

                data = data[mask]
                labels = labels[mask]

                df = pd.DataFrame({
                    't-SNE 1': data[:, 0],
                    't-SNE 2': data[:, 1],
                    'Class': labels.astype(str)
                })
                dfs.append(df)
                print(f"Loaded {variant['title']} from {pt_path.name}")
                loaded = True
                break
        if not loaded:
            print(f"Using dummy data for {variant['title']}")
            data = np.random.randn(1000, 2)
            labels = np.random.randint(0, 2, 1000)
            df = pd.DataFrame({'t-SNE 1': data[:, 0], 't-SNE 2': data[:, 1], 'Class': labels.astype(str)})
            dfs.append(df)
    return dfs

dfs = load_tsne_data()

# ====================== DASH APP ======================
app = Dash(__name__)

app.layout = html.Div([
    html.H1("t-SNE Visualization - Model Variants Comparison",
            style={'textAlign': 'center', 'margin': '30px', 'color': '#2c3e50'}),
    
    html.Button("Save All 5 Plots in ONE Image",
                id='save-button',
                style={'fontSize': '18px', 'padding': '12px 24px', 'margin': '20px auto', 'display': 'block'}),
    
    html.Div(id='save-status', style={'textAlign': 'center', 'margin': '15px', 'color': 'green', 'fontWeight': 'bold'}),
    
    # Grid Layout (for viewing)
    html.Div([
        html.Div([dcc.Graph(id='graph-0', style={'height': '520px'})], 
                 style={'width': '48%', 'display': 'inline-block', 'padding': '8px'}),
        html.Div([dcc.Graph(id='graph-1', style={'height': '520px'})], 
                 style={'width': '48%', 'display': 'inline-block', 'padding': '8px'}),
        html.Div([dcc.Graph(id='graph-2', style={'height': '520px'})], 
                 style={'width': '32%', 'display': 'inline-block', 'padding': '8px'}),
        html.Div([dcc.Graph(id='graph-3', style={'height': '520px'})], 
                 style={'width': '32%', 'display': 'inline-block', 'padding': '8px'}),
        html.Div([dcc.Graph(id='graph-4', style={'height': '520px'})], 
                 style={'width': '32%', 'display': 'inline-block', 'padding': '8px'}),
    ], style={'padding': '20px', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})
])

@app.callback(
    [Output(f'graph-{i}', 'figure') for i in range(5)],
    Input('save-button', 'n_clicks')
)
def update_graphs(_):
    return [
        px.scatter(dfs[i], x='t-SNE 1', y='t-SNE 2', color='Class',
                   title=f"{VARIANTS[i]['title']}", opacity=0.85, height=520,
                   color_discrete_sequence=px.colors.qualitative.Set2)
        .update_layout(template="plotly_white", margin=dict(l=30,r=30,t=60,b=30))
        for i in range(5)
    ]

# ====================== SAVE AS SINGLE IMAGE ======================
@app.callback(
    Output('save-status', 'children'),
    Input('save-button', 'n_clicks'),
    prevent_initial_call=True
)
def save_combined_image(n_clicks):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = SAVE_FOLDER / f"tsne_all_variants_{timestamp}.png"

        # Create figures with consistent size
        figures = []
        for i in range(5):
            fig = px.scatter(
                dfs[i], x='t-SNE 1', y='t-SNE 2', color='Class',
                title=f"{VARIANTS[i]['title']}",
                opacity=0.85, height=650,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=50, r=50, t=100, b=50),
                title_font_size=18
            )
            figures.append(fig)

        # Save each figure to image
        images = []
        for fig in figures:
            img_bytes = fig.to_image(format="png", scale=2)   # scale=2 for good quality
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)

        # === FIXED LAYOUT: 3 columns wide, 2 rows ===
        single_width = images[0].width
        single_height = images[0].height
        
        canvas_width = single_width * 3
        canvas_height = single_height * 2
        
        combined = Image.new('RGB', (canvas_width, canvas_height), color=(255, 255, 255))

        # Top row: 3 plots (0, 1, 2)
        combined.paste(images[0], (0, 0))
        combined.paste(images[1], (single_width, 0))
        combined.paste(images[2], (single_width * 2, 0))

        # Bottom row: 2 plots (3, 4) — centered
        offset_x = (canvas_width - single_width * 2) // 2
        combined.paste(images[3], (offset_x, single_height))
        combined.paste(images[4], (offset_x + single_width, single_height))

        # Save with high DPI
        combined.save(filename, dpi=(300, 300))
        
        return f"Successfully saved **one combined image**!<br>" \
               f"File: <b>{filename.name}</b><br>" \
               f"Path: {SAVE_FOLDER.absolute()}"

    except Exception as e:
        return f"Error: {str(e)}<br>Make sure `kaleido` and `pillow` are installed (`pip install kaleido pillow`)"
# ====================== RUN ======================
if __name__ == '__main__':
    print("\ Starting t-SNE Dashboard...")
    print(f"Combined images will be saved in: {SAVE_FOLDER.absolute()}")
    print("Open: http://127.0.0.1:8050")
    print(" (Ctrl + C to stop)\n")
    
    app.run(debug=False, port=8050, host='127.0.0.1')