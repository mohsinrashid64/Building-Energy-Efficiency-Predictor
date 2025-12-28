# Building Energy Efficiency Predictor - Gradio App
# A professional ML demo for portfolio showcase

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# LOAD MODELS AND METADATA
# ============================================================

def load_models():
    """Load all trained models and metadata."""
    metadata = joblib.load('models/all_models_metadata.joblib')

    models = {}
    for m in metadata['models']:
        model_key = f"{m['model_name']} (Degree {m['degree']})"
        models[model_key] = {
            'model': joblib.load(m['filename']),
            'r2_score': m['r2_score'],
            'name': m['model_name'],
            'degree': m['degree']
        }

    return models, metadata

# Load models at startup
MODELS, METADATA = load_models()

# Feature ranges (from dataset analysis)
FEATURE_CONFIG = {
    'Relative Compactness': {'min': 0.62, 'max': 0.98, 'default': 0.76, 'step': 0.01,
        'description': 'Volume-to-surface area ratio (higher = more compact)'},
    'Surface Area': {'min': 514.5, 'max': 808.5, 'default': 671.7, 'step': 1.0,
        'description': 'Total exterior surface area (m²)'},
    'Wall Area': {'min': 245.0, 'max': 416.5, 'default': 318.5, 'step': 1.0,
        'description': 'Total wall area (m²)'},
    'Roof Area': {'min': 110.25, 'max': 220.5, 'default': 176.6, 'step': 0.25,
        'description': 'Roof area (m²)'},
    'Overall Height': {'min': 3.5, 'max': 7.0, 'default': 5.25, 'step': 0.5,
        'description': 'Building height (m) - typically 3.5m or 7m'},
    'Orientation': {'min': 2, 'max': 5, 'default': 3, 'step': 1,
        'description': '2=North, 3=East, 4=South, 5=West'},
    'Glazing Area': {'min': 0.0, 'max': 0.4, 'default': 0.25, 'step': 0.05,
        'description': 'Window-to-floor area ratio (0-40%)'},
    'Glazing Area Distribution': {'min': 0, 'max': 5, 'default': 3, 'step': 1,
        'description': '0=None, 1-4=Uniform on facade, 5=Mixed'}
}

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_single(model_choice, rel_compact, surface_area, wall_area,
                   roof_area, height, orientation, glazing_area, glazing_dist):
    """Make prediction with selected model."""

    input_data = pd.DataFrame([{
        'Relative Compactness': rel_compact,
        'Surface Area': surface_area,
        'Wall Area': wall_area,
        'Roof Area': roof_area,
        'Overall Height': height,
        'Orientation': int(orientation),
        'Glazing Area': glazing_area,
        'Glazing Area Distribution': int(glazing_dist)
    }])

    model_info = MODELS[model_choice]
    prediction = model_info['model'].predict(input_data)[0]

    # Create result card
    result_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px; border-radius: 15px; text-align: center; color: white;">
        <h2 style="margin: 0; font-size: 18px; opacity: 0.9;">Predicted Heating Load</h2>
        <p style="font-size: 48px; font-weight: bold; margin: 15px 0;">{prediction:.2f}</p>
        <p style="margin: 0; font-size: 14px; opacity: 0.8;">kWh/m² per year</p>
        <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.3); margin: 20px 0;">
        <p style="margin: 0; font-size: 12px;">
            Model: {model_info['name']} | Polynomial Degree: {model_info['degree']} | R² Score: {model_info['r2_score']}
        </p>
    </div>
    """

    # Energy efficiency interpretation
    if prediction < 15:
        efficiency = "Excellent - Very Low Energy Consumption"
        color = "#10b981"
    elif prediction < 25:
        efficiency = "Good - Moderate Energy Consumption"
        color = "#3b82f6"
    elif prediction < 35:
        efficiency = "Average - Typical Energy Consumption"
        color = "#f59e0b"
    else:
        efficiency = "Poor - High Energy Consumption"
        color = "#ef4444"

    interpretation_html = f"""
    <div style="background: {color}22; border-left: 4px solid {color};
                padding: 15px; border-radius: 8px; margin-top: 15px;">
        <p style="margin: 0; color: {color}; font-weight: 600;">Energy Efficiency Rating</p>
        <p style="margin: 5px 0 0 0; color: #374151;">{efficiency}</p>
    </div>
    """

    return result_html + interpretation_html


def compare_all_models(rel_compact, surface_area, wall_area, roof_area,
                       height, orientation, glazing_area, glazing_dist):
    """Compare predictions across all models."""

    input_data = pd.DataFrame([{
        'Relative Compactness': rel_compact,
        'Surface Area': surface_area,
        'Wall Area': wall_area,
        'Roof Area': roof_area,
        'Overall Height': height,
        'Orientation': int(orientation),
        'Glazing Area': glazing_area,
        'Glazing Area Distribution': int(glazing_dist)
    }])

    results = []
    for model_key, model_info in MODELS.items():
        pred = model_info['model'].predict(input_data)[0]
        results.append({
            'Model': model_info['name'],
            'Degree': model_info['degree'],
            'R² Score': model_info['r2_score'],
            'Prediction (kWh/m²)': round(pred, 2)
        })

    df = pd.DataFrame(results)
    df = df.sort_values('R² Score', ascending=False)

    return df


def get_model_performance():
    """Get performance metrics for all models."""
    data = []
    for model_key, model_info in MODELS.items():
        data.append({
            'Model': model_info['name'],
            'Polynomial Degree': model_info['degree'],
            'R² Score': model_info['r2_score'],
            'Accuracy (%)': round(model_info['r2_score'] * 100, 2)
        })

    df = pd.DataFrame(data)
    df = df.sort_values('R² Score', ascending=False)
    return df


# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """
/* Main container styling */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header styling */
.header-text {
    text-align: center;
    padding: 20px;
}

/* Slider labels */
.slider-label {
    font-weight: 600;
    color: #374151;
}

/* Tab styling */
.tabs {
    border-radius: 10px;
}

/* Button styling */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}

/* Info boxes */
.info-box {
    background: #f3f4f6;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
"""

# ============================================================
# BUILD GRADIO INTERFACE
# ============================================================

def create_app():
    with gr.Blocks(css=custom_css, title="Building Energy Efficiency Predictor", theme=gr.themes.Soft()) as app:

        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 30px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 15px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 32px;">Building Energy Efficiency Predictor</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 16px;">
                Predict heating load requirements based on building characteristics
            </p>
        </div>
        """)

        with gr.Tabs():
            # ==================== TAB 1: PREDICT ====================
            with gr.TabItem("Predict", id=1):
                gr.HTML("""
                <div style="background: #eff6ff; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #3b82f6;">
                    <p style="margin: 0; color: #1e40af;">
                        <strong>Instructions:</strong> Adjust the building parameters below and click "Predict" to estimate the heating load.
                    </p>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        # Model Selection
                        model_dropdown = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="Lasso (Degree 3)",
                            label="Select Model",
                            info="Lasso Degree 3 has the highest accuracy (R² = 0.99)"
                        )

                        gr.HTML("<h3 style='margin: 20px 0 10px 0; color: #374151;'>Building Geometry</h3>")

                        with gr.Row():
                            rel_compact = gr.Slider(
                                minimum=FEATURE_CONFIG['Relative Compactness']['min'],
                                maximum=FEATURE_CONFIG['Relative Compactness']['max'],
                                value=FEATURE_CONFIG['Relative Compactness']['default'],
                                step=FEATURE_CONFIG['Relative Compactness']['step'],
                                label="Relative Compactness",
                                info=FEATURE_CONFIG['Relative Compactness']['description']
                            )
                            surface_area = gr.Slider(
                                minimum=FEATURE_CONFIG['Surface Area']['min'],
                                maximum=FEATURE_CONFIG['Surface Area']['max'],
                                value=FEATURE_CONFIG['Surface Area']['default'],
                                step=FEATURE_CONFIG['Surface Area']['step'],
                                label="Surface Area (m²)",
                                info=FEATURE_CONFIG['Surface Area']['description']
                            )

                        with gr.Row():
                            wall_area = gr.Slider(
                                minimum=FEATURE_CONFIG['Wall Area']['min'],
                                maximum=FEATURE_CONFIG['Wall Area']['max'],
                                value=FEATURE_CONFIG['Wall Area']['default'],
                                step=FEATURE_CONFIG['Wall Area']['step'],
                                label="Wall Area (m²)",
                                info=FEATURE_CONFIG['Wall Area']['description']
                            )
                            roof_area = gr.Slider(
                                minimum=FEATURE_CONFIG['Roof Area']['min'],
                                maximum=FEATURE_CONFIG['Roof Area']['max'],
                                value=FEATURE_CONFIG['Roof Area']['default'],
                                step=FEATURE_CONFIG['Roof Area']['step'],
                                label="Roof Area (m²)",
                                info=FEATURE_CONFIG['Roof Area']['description']
                            )

                        with gr.Row():
                            height = gr.Slider(
                                minimum=FEATURE_CONFIG['Overall Height']['min'],
                                maximum=FEATURE_CONFIG['Overall Height']['max'],
                                value=FEATURE_CONFIG['Overall Height']['default'],
                                step=FEATURE_CONFIG['Overall Height']['step'],
                                label="Overall Height (m)",
                                info=FEATURE_CONFIG['Overall Height']['description']
                            )
                            orientation = gr.Slider(
                                minimum=FEATURE_CONFIG['Orientation']['min'],
                                maximum=FEATURE_CONFIG['Orientation']['max'],
                                value=FEATURE_CONFIG['Orientation']['default'],
                                step=FEATURE_CONFIG['Orientation']['step'],
                                label="Orientation",
                                info=FEATURE_CONFIG['Orientation']['description']
                            )

                        gr.HTML("<h3 style='margin: 20px 0 10px 0; color: #374151;'>Glazing Properties</h3>")

                        with gr.Row():
                            glazing_area = gr.Slider(
                                minimum=FEATURE_CONFIG['Glazing Area']['min'],
                                maximum=FEATURE_CONFIG['Glazing Area']['max'],
                                value=FEATURE_CONFIG['Glazing Area']['default'],
                                step=FEATURE_CONFIG['Glazing Area']['step'],
                                label="Glazing Area Ratio",
                                info=FEATURE_CONFIG['Glazing Area']['description']
                            )
                            glazing_dist = gr.Slider(
                                minimum=FEATURE_CONFIG['Glazing Area Distribution']['min'],
                                maximum=FEATURE_CONFIG['Glazing Area Distribution']['max'],
                                value=FEATURE_CONFIG['Glazing Area Distribution']['default'],
                                step=FEATURE_CONFIG['Glazing Area Distribution']['step'],
                                label="Glazing Distribution",
                                info=FEATURE_CONFIG['Glazing Area Distribution']['description']
                            )

                        predict_btn = gr.Button("Predict Heating Load", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        prediction_output = gr.HTML(
                            value="""
                            <div style="background: #f3f4f6; padding: 40px; border-radius: 15px;
                                        text-align: center; color: #6b7280; height: 200px;
                                        display: flex; align-items: center; justify-content: center;">
                                <p>Adjust parameters and click "Predict" to see results</p>
                            </div>
                            """
                        )

                predict_btn.click(
                    fn=predict_single,
                    inputs=[model_dropdown, rel_compact, surface_area, wall_area,
                            roof_area, height, orientation, glazing_area, glazing_dist],
                    outputs=prediction_output
                )

            # ==================== TAB 2: COMPARE MODELS ====================
            with gr.TabItem("Compare Models", id=2):
                gr.HTML("""
                <div style="background: #fef3c7; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #f59e0b;">
                    <p style="margin: 0; color: #92400e;">
                        <strong>Model Comparison:</strong> See how different models predict for the same building parameters.
                    </p>
                </div>
                """)

                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h3 style='color: #374151;'>Set Building Parameters</h3>")

                        with gr.Row():
                            cmp_rel_compact = gr.Slider(0.62, 0.98, 0.76, step=0.01, label="Relative Compactness")
                            cmp_surface = gr.Slider(514.5, 808.5, 671.7, step=1.0, label="Surface Area (m²)")
                        with gr.Row():
                            cmp_wall = gr.Slider(245.0, 416.5, 318.5, step=1.0, label="Wall Area (m²)")
                            cmp_roof = gr.Slider(110.25, 220.5, 176.6, step=0.25, label="Roof Area (m²)")
                        with gr.Row():
                            cmp_height = gr.Slider(3.5, 7.0, 5.25, step=0.5, label="Overall Height (m)")
                            cmp_orient = gr.Slider(2, 5, 3, step=1, label="Orientation")
                        with gr.Row():
                            cmp_glaze = gr.Slider(0.0, 0.4, 0.25, step=0.05, label="Glazing Area Ratio")
                            cmp_glaze_dist = gr.Slider(0, 5, 3, step=1, label="Glazing Distribution")

                        compare_btn = gr.Button("Compare All Models", variant="primary", size="lg")

                comparison_table = gr.DataFrame(
                    headers=["Model", "Degree", "R² Score", "Prediction (kWh/m²)"],
                    label="Model Comparison Results",
                    interactive=False
                )

                compare_btn.click(
                    fn=compare_all_models,
                    inputs=[cmp_rel_compact, cmp_surface, cmp_wall, cmp_roof,
                            cmp_height, cmp_orient, cmp_glaze, cmp_glaze_dist],
                    outputs=comparison_table
                )

            # ==================== TAB 3: MODEL INFO ====================
            with gr.TabItem("Model Performance", id=3):
                gr.HTML("""
                <div style="background: #ecfdf5; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #10b981;">
                    <p style="margin: 0; color: #065f46;">
                        <strong>Performance Overview:</strong> All trained models ranked by R² score (coefficient of determination).
                    </p>
                </div>
                """)

                performance_table = gr.DataFrame(
                    value=get_model_performance(),
                    headers=["Model", "Polynomial Degree", "R² Score", "Accuracy (%)"],
                    label="Model Performance Metrics",
                    interactive=False
                )

                gr.HTML("""
                <div style="margin-top: 30px;">
                    <h3 style="color: #374151;">Understanding the Models</h3>

                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-top: 20px;">
                        <div style="background: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;">
                            <h4 style="color: #7c3aed; margin: 0 0 10px 0;">Ridge Regression</h4>
                            <p style="color: #64748b; margin: 0; font-size: 14px;">
                                L2 regularization prevents overfitting by penalizing large coefficients.
                                Good for handling multicollinearity in features.
                            </p>
                        </div>

                        <div style="background: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;">
                            <h4 style="color: #059669; margin: 0 0 10px 0;">Lasso Regression</h4>
                            <p style="color: #64748b; margin: 0; font-size: 14px;">
                                L1 regularization performs feature selection by shrinking some coefficients to zero.
                                Best performer in this project.
                            </p>
                        </div>

                        <div style="background: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;">
                            <h4 style="color: #0891b2; margin: 0 0 10px 0;">ElasticNet</h4>
                            <p style="color: #64748b; margin: 0; font-size: 14px;">
                                Combines L1 and L2 regularization. Useful when there are correlated features
                                and you want some feature selection.
                            </p>
                        </div>
                    </div>

                    <div style="background: #faf5ff; padding: 20px; border-radius: 12px; margin-top: 20px; border: 1px solid #e9d5ff;">
                        <h4 style="color: #7c3aed; margin: 0 0 10px 0;">Polynomial Features</h4>
                        <p style="color: #64748b; margin: 0; font-size: 14px;">
                            Higher polynomial degrees capture non-linear relationships between features.
                            Degree 3 achieved the best results, indicating significant non-linear patterns in building energy consumption.
                        </p>
                    </div>
                </div>
                """)

            # ==================== TAB 4: ABOUT ====================
            with gr.TabItem("About", id=4):
                gr.HTML("""
                <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #374151; text-align: center;">About This Project</h2>

                    <div style="background: #f8fafc; padding: 25px; border-radius: 15px; margin: 20px 0;">
                        <h3 style="color: #1e40af; margin-top: 0;">Project Overview</h3>
                        <p style="color: #475569; line-height: 1.7;">
                            This machine learning application predicts the <strong>heating load</strong> of buildings
                            based on their architectural characteristics. It helps architects and engineers
                            estimate energy requirements during the design phase, enabling more energy-efficient
                            building designs.
                        </p>
                    </div>

                    <div style="background: #f8fafc; padding: 25px; border-radius: 15px; margin: 20px 0;">
                        <h3 style="color: #1e40af; margin-top: 0;">Dataset</h3>
                        <p style="color: #475569; line-height: 1.7;">
                            <strong>Source:</strong> UCI Machine Learning Repository (ENB2012)<br>
                            <strong>Samples:</strong> 768 building simulations<br>
                            <strong>Features:</strong> 8 architectural parameters<br>
                            <strong>Target:</strong> Heating Load (kWh/m² per year)
                        </p>
                    </div>

                    <div style="background: #f8fafc; padding: 25px; border-radius: 15px; margin: 20px 0;">
                        <h3 style="color: #1e40af; margin-top: 0;">Technical Approach</h3>
                        <ul style="color: #475569; line-height: 1.8;">
                            <li>Polynomial feature engineering (degrees 1-3)</li>
                            <li>StandardScaler for numerical features</li>
                            <li>OneHotEncoder for categorical features</li>
                            <li>5-Fold Cross-Validation for robust evaluation</li>
                            <li>Regularized regression models (Ridge, Lasso, ElasticNet)</li>
                        </ul>
                    </div>

                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding: 25px; border-radius: 15px; margin: 20px 0; color: white;">
                        <h3 style="margin-top: 0;">Best Model Performance</h3>
                        <div style="display: flex; justify-content: space-around; text-align: center; flex-wrap: wrap;">
                            <div style="padding: 10px;">
                                <p style="font-size: 36px; font-weight: bold; margin: 0;">99.08%</p>
                                <p style="opacity: 0.9; margin: 5px 0 0 0;">R² Score</p>
                            </div>
                            <div style="padding: 10px;">
                                <p style="font-size: 36px; font-weight: bold; margin: 0;">Lasso</p>
                                <p style="opacity: 0.9; margin: 5px 0 0 0;">Best Model</p>
                            </div>
                            <div style="padding: 10px;">
                                <p style="font-size: 36px; font-weight: bold; margin: 0;">Degree 3</p>
                                <p style="opacity: 0.9; margin: 5px 0 0 0;">Polynomial</p>
                            </div>
                        </div>
                    </div>

                    <div style="text-align: center; margin-top: 30px; padding: 20px;">
                        <p style="color: #6b7280;">
                            Built with scikit-learn, pandas, and Gradio<br>
                            <a href="https://github.com" style="color: #667eea;">View on GitHub</a>
                        </p>
                    </div>
                </div>
                """)

        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #e5e7eb;">
            <p style="color: #9ca3af; font-size: 14px; margin: 0;">
                Building Energy Efficiency Predictor | Machine Learning Portfolio Project
            </p>
        </div>
        """)

    return app

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    app = create_app()
    app.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",
        server_port=7860
    )
