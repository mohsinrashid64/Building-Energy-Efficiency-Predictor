# Building Energy Efficiency Predictor

A machine learning application that predicts the **heating load** of buildings based on their architectural characteristics. This tool helps architects, engineers, and energy consultants estimate energy requirements during the building design phase.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Dataset](#dataset)
- [Models & Performance](#models--performance)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Using the Web App](#using-the-web-app)
- [Running the Jupyter Notebook](#running-the-jupyter-notebook)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This project uses regression models with polynomial feature engineering to predict building heating loads. The best performing model achieves an **R² score of 0.99**, meaning it explains 99% of the variance in heating load predictions.

### Why This Matters

- **Energy Efficiency**: Helps optimize building designs for lower energy consumption
- **Cost Savings**: Accurate predictions enable better HVAC system sizing
- **Sustainability**: Supports green building initiatives by identifying energy-efficient designs

---

## Features

- **Interactive Web Interface**: User-friendly Gradio app with sliders for all building parameters
- **Multiple Models**: Compare predictions across 9 different model configurations
- **Model Selection**: Choose between Ridge, Lasso, and ElasticNet regression
- **Polynomial Features**: Models trained with degrees 1, 2, and 3 for capturing non-linear relationships
- **Energy Efficiency Rating**: Automatic classification of predicted heating load
- **Performance Metrics**: View R² scores and compare model accuracy

---

## Demo

[Live Demo on Hugging Face Spaces](#) *(Coming Soon)*

---

## Dataset

**Source**: [UCI Machine Learning Repository - Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)

| Property | Value |
|----------|-------|
| Samples | 768 buildings |
| Features | 8 input variables |
| Target | Heating Load (kWh/m² per year) |
| Missing Values | None |

### Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| Relative Compactness | Volume-to-surface area ratio | 0.62 - 0.98 |
| Surface Area | Total exterior surface area (m²) | 514.5 - 808.5 |
| Wall Area | Total wall area (m²) | 245.0 - 416.5 |
| Roof Area | Roof area (m²) | 110.25 - 220.5 |
| Overall Height | Building height (m) | 3.5 - 7.0 |
| Orientation | Building orientation (2=N, 3=E, 4=S, 5=W) | 2 - 5 |
| Glazing Area | Window-to-floor area ratio | 0.0 - 0.4 |
| Glazing Area Distribution | Window distribution across facades | 0 - 5 |

### Target Variable

| Variable | Description | Range |
|----------|-------------|-------|
| Heating Load | Annual heating energy requirement (kWh/m²) | 6.01 - 43.10 |

---

## Models & Performance

All models were evaluated using **5-Fold Cross-Validation**.

| Model | Polynomial Degree | R² Score | Accuracy |
|-------|-------------------|----------|----------|
| **Lasso** | **3** | **0.9921** | **99.21%** |
| Ridge | 3 | 0.9931 | 99.31% |
| ElasticNet | 3 | 0.9753 | 97.53% |
| Lasso | 2 | 0.9608 | 96.08% |
| Ridge | 2 | 0.9620 | 96.20% |
| ElasticNet | 2 | 0.9517 | 95.17% |
| Lasso | 1 | 0.9237 | 92.37% |
| Ridge | 1 | 0.9240 | 92.40% |
| ElasticNet | 1 | 0.9230 | 92.30% |

**Best Model**: Lasso Regression with Polynomial Degree 3

---

## Project Structure

```
Building Energy Efficiency Predictor/
│
├── app.py                          # Gradio web application
├── save_model.py                   # Script to train and save all models
├── load_model.py                   # Example script for loading saved models
├── building_energy_efficiency_predictor.ipynb  # Jupyter notebook with full analysis
├── ENB2012_data.csv                # Dataset
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/                         # Saved trained models
│   ├── ridge_degree1.joblib
│   ├── ridge_degree2.joblib
│   ├── ridge_degree3.joblib
│   ├── lasso_degree1.joblib
│   ├── lasso_degree2.joblib
│   ├── lasso_degree3.joblib
│   ├── elasticnet_degree1.joblib
│   ├── elasticnet_degree2.joblib
│   ├── elasticnet_degree3.joblib
│   └── all_models_metadata.joblib
│
└── .venv/                          # Virtual environment (not in repo)
```

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** - Python package installer (comes with Python)
- **Git** - For cloning the repository - [Download Git](https://git-scm.com/downloads)

### Verify Installation

Open a terminal/command prompt and run:

```bash
python --version
# Should output: Python 3.8.x or higher

pip --version
# Should output: pip 21.x.x or higher
```

---

## Installation

### Step 1: Download the Repository

Download the ZIP file from GitHub and extract it, or clone using Git:

```bash
git clone <repository-url>
cd building-energy-efficiency-predictor
```

### Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment keeps project dependencies isolated.

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn
- joblib
- gradio

### Step 4: Train and Save Models

If the `models/` folder is empty or missing, you need to train the models:

```bash
python save_model.py
```

Expected output:
```
Training and saving all models...
==================================================
[OK] Ridge (Degree 1) - R2: 0.9240
[OK] Lasso (Degree 1) - R2: 0.9237
[OK] ElasticNet (Degree 1) - R2: 0.9230
[OK] Ridge (Degree 2) - R2: 0.9620
[OK] Lasso (Degree 2) - R2: 0.9608
[OK] ElasticNet (Degree 2) - R2: 0.9517
[OK] Ridge (Degree 3) - R2: 0.9931
[OK] Lasso (Degree 3) - R2: 0.9921
[OK] ElasticNet (Degree 3) - R2: 0.9753
==================================================
[OK] All 9 models saved to models/ directory
[OK] Metadata saved to models/all_models_metadata.joblib
```

---

## Running the Application

### Start the Web App

```bash
python app.py
```

Expected output:
```
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

### Access the App

Open your web browser and go to:

```
http://127.0.0.1:7860
```

or

```
http://localhost:7860
```

### Stop the App

Press `Ctrl + C` in the terminal to stop the server.

---

## Using the Web App

The app has 4 tabs:

### Tab 1: Predict

1. **Select a Model** from the dropdown (default: Lasso Degree 3)
2. **Adjust Building Parameters** using the sliders:
   - Relative Compactness
   - Surface Area
   - Wall Area
   - Roof Area
   - Overall Height
   - Orientation
   - Glazing Area
   - Glazing Area Distribution
3. **Click "Predict Heating Load"**
4. View the prediction result and energy efficiency rating

### Tab 2: Compare Models

1. Set your building parameters
2. Click "Compare All Models"
3. See predictions from all 9 models side-by-side
4. Compare R² scores to understand model reliability

### Tab 3: Model Performance

- View all models ranked by R² score
- Read explanations of Ridge, Lasso, and ElasticNet
- Understand polynomial feature engineering

### Tab 4: About

- Project overview
- Dataset information
- Technical approach
- Best model statistics

---

## Running the Jupyter Notebook

The notebook contains the complete analysis including:
- Exploratory Data Analysis (EDA)
- Data visualization
- Model training and evaluation
- Cross-validation results

### Start Jupyter

```bash
jupyter notebook building_energy_efficiency_predictor.ipynb
```

Or use JupyterLab:

```bash
jupyter lab
```

### Run All Cells

In Jupyter:
1. Click **Kernel** → **Restart & Run All**
2. Wait for all cells to execute
3. View visualizations and results

---

## Technical Details

### Data Preprocessing

1. **Column Cleaning**: Removed trailing whitespace from column names
2. **Feature Scaling**: StandardScaler for numerical features
3. **Encoding**: OneHotEncoder for categorical features (Orientation, Glazing Distribution)
4. **Feature Engineering**: PolynomialFeatures for degrees 1, 2, and 3

### Model Pipeline

```
Input Data → Preprocessing → Polynomial Features → Regularized Regression → Prediction
```

### Regularization Techniques

| Model | Regularization | Key Property |
|-------|----------------|--------------|
| Ridge | L2 (squared coefficients) | Shrinks all coefficients |
| Lasso | L1 (absolute coefficients) | Can set coefficients to zero (feature selection) |
| ElasticNet | L1 + L2 combined | Balance of both approaches |

### Cross-Validation

- **Method**: 5-Fold Cross-Validation
- **Metric**: R² Score (Coefficient of Determination)
- **Purpose**: Robust evaluation, prevents overfitting

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'xxx'"

**Solution**: Install the missing package:
```bash
pip install xxx
```

Or reinstall all requirements:
```bash
pip install -r requirements.txt
```

### Issue: "sklearn version mismatch" warning

**Solution**: Retrain the models with your current sklearn version:
```bash
python save_model.py
```

### Issue: "Port 7860 already in use"

**Solution**: Either:
1. Stop the other process using port 7860
2. Or modify `app.py` to use a different port:
   ```python
   app.launch(server_port=7861)
   ```

### Issue: App runs but shows blank page

**Solution**:
1. Clear browser cache
2. Try a different browser
3. Check terminal for error messages

### Issue: Virtual environment not activating

**Windows Solution**:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate
```

**macOS/Linux Solution**:
```bash
source .venv/bin/activate
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**If you find this project helpful, please give it a star on GitHub!**
