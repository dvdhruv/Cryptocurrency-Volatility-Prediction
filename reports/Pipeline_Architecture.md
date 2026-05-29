# Pipeline Architecture

Raw Dataset -> Cleaning -> Feature Engineering -> Scaling -> Model Training -> Model Evaluation -> Model Saving -> Streamlit Prediction

## Notes
The pipeline preserves time order to avoid data leakage and uses rolling features to capture short-term market behavior.
