# Drinking Water Quality Prediction

A Streamlit machine-learning application that predicts whether uploaded water samples are safe for drinking.

## Features

- Upload a CSV file containing water-quality measurements
- Predict safe or unsafe water samples
- Display prediction confidence for each row
- Summarize safe and unsafe samples
- Download prediction results as CSV
- Show model and dataset visualizations

## Project Files

- `app.py`: Streamlit application.
- `water_model.pkl`: Trained machine-learning model.
- `water_potability.csv`: Source water-potability dataset.
- `water_potability_cleaned.csv`: Cleaned dataset.
- `graphs/`: Accuracy, class distribution, correlation, confusion matrix, feature importance, and missing-value charts.

## Run Locally

Install the dependencies:

```bash
pip install streamlit pandas scikit-learn
```

Start the application:

```bash
streamlit run app.py
```

The app opens in your browser. Upload a CSV containing the feature columns expected by the trained model.

## Disclaimer

Predictions are for educational and demonstration purposes and should not replace laboratory testing or regulatory water-safety analysis.
