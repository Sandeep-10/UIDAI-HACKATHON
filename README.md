# UIDAI Biometric Risk Prediction System

A Streamlit web application for predicting biometric usage risk levels using machine learning.

## Features

- **Risk Prediction**: Predict risk levels based on state, district, pincode, and date
- **Data Analysis**: Interactive dashboards with visualizations
- **About Page**: Information about the application

## Deployment to Streamlit Cloud

### Prerequisites

1. GitHub account
2. Streamlit Cloud account (free at https://streamlit.io/cloud)

### Step-by-Step Deployment

1. **Push your code to GitHub**
   - Make sure all files are committed and pushed to your GitHub repository
   - Required files:
     - `streamlit_app.py` (main application file)
     - `requirements.txt` (dependencies)
     - `risk_prediction_model.pkl` (ML model)
     - `state_encoder.pkl` (state encoder)
     - `district_encoder.pkl` (district encoder)
     - `risk_encoder.pkl` (risk encoder)
     - CSV data files (if needed)

2. **Go to Streamlit Cloud**
   - Visit https://share.streamlit.io/
   - Sign in with your GitHub account

3. **Deploy your app**
   - Click "New app"
   - Select your GitHub repository: `Sandeep-10/UIDAI-HACKATHON`
   - Set the main file path: `streamlit_app.py`
   - Choose branch: `main`
   - Click "Deploy"

4. **Wait for deployment**
   - Streamlit Cloud will install dependencies from `requirements.txt`
   - Your app will be available at: `https://your-app-name.streamlit.app`

### Important Notes

- **File Size Limits**: Streamlit Cloud has file size limits. Large CSV files may need to be:
  - Compressed
  - Stored in cloud storage (S3, Google Cloud Storage) and loaded via URL
  - Or use a database instead

- **Model Files**: Ensure all `.pkl` files are in your repository

- **Data Files**: CSV files should be in the same directory as `streamlit_app.py` or use relative paths

## Local Development

To run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Requirements

See `requirements.txt` for all dependencies.
