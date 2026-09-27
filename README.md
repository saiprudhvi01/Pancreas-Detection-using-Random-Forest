# PancreasCare AI

Machine Learning-Based Pancreatic Health Risk Assessment and Personalized Guidance System

## Overview

PancreasCare AI is a research-oriented application providing a machine learning-based pancreatic health risk assessment. It utilizes a Random Forest classifier trained on a dataset of 50,000 pancreatic cancer patient records.

The system features:
- **Real ML Predictions**: Powered by a trained Random Forest model.
- **Risk Assessment**: Classifies risk into Low, Moderate, and High based on pre-diagnosis factors (mapped from diagnosis stages to prevent data leakage).
- **Explainability**: Shows feature importance derived directly from the trained model.
- **Personalized Guidance**: Generates contextual health guidance based on the actual inputs and prediction.
- **Pure Python Backend**: Entirely built on Flask with Jinja templates (no npm, no React).

## Project Structure

- `app.py`: Flask backend and API routing.
- `ml/`: Saved ML artifacts including `model.joblib`, `feature_metadata.json`, and training metrics.
- `templates/`: HTML Jinja2 templates (styled with Tailwind CSS via CDN).

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

## Disclaimer

**This system is intended for health awareness and educational purposes only. It is not a substitute for professional medical diagnosis, treatment, or medical advice. Consult a qualified healthcare professional for medical concerns.**
