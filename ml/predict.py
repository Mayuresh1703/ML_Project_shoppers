# This file loads saved model and makes predictions on new input data

from pathlib import Path
import pickle  # load saved model, scaler, selector
import numpy as np  # numerical operations
import pandas as pd  # data manipulation
from dotenv import load_dotenv  # load env variables
import os  # access env variables

load_dotenv()  # load .env file

repo_root = Path(__file__).resolve().parent.parent

def load_artifacts():
    # resolve model path relative to project root if needed
    model_path = Path(os.getenv('MODEL_PATH'))
    if not model_path.is_absolute():
        model_path = (repo_root / model_path).resolve()

    # load saved model from disk
    with open(model_path, 'rb') as f:
        model = pickle.load(f)  # deserialize model

    # load saved scaler from disk
    with open(repo_root / 'ml' / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)  # deserialize scaler

    # load saved selector from disk
    with open(repo_root / 'ml' / 'selector.pkl', 'rb') as f:
        selector = pickle.load(f)  # deserialize selector

    # load selected feature names
    with open(repo_root / 'ml' / 'selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)  # deserialize feature names

    return model, scaler, selector, selected_features  # return all artifacts


def predict(input_data: dict):
    model, scaler, selector, selected_features = load_artifacts()  # load all artifacts

    # if training data contained only one class, warn and return a safe message
    if hasattr(model, 'classes_') and len(model.classes_) == 1:
        only_class = int(model.classes_[0])
        message = (
            "Model was trained only on one class and cannot reliably predict purchases. "
            "Current model always returns the majority class."
        )
        return {
            "prediction": only_class,
            "probability": 0.0,
            "message": message
        }

    # convert input dict to dataframe
    df = pd.DataFrame([input_data])

    # encode raw Month and VisitorType into the one-hot columns used during training
    month_columns = [
        'Month_Aug', 'Month_Dec', 'Month_Feb', 'Month_Jul',
        'Month_Jun', 'Month_Mar', 'Month_May', 'Month_Nov',
        'Month_Oct', 'Month_Sep'
    ]
    visitor_columns = [
        'VisitorType_New_Visitor',
        'VisitorType_Returning_Visitor'
    ]

    month_index = int(df.at[0, 'Month'])
    visitor_index = int(df.at[0, 'VisitorType'])

    month_map = {
        0: 'Month_Aug',
        1: 'Month_Dec',
        2: 'Month_Feb',
        3: 'Month_Jul',
        4: 'Month_Jun',
        5: 'Month_Mar',
        6: 'Month_May',
        7: 'Month_Nov',
        8: 'Month_Oct',
        9: 'Month_Sep'
    }
    visitor_map = {
        0: 'VisitorType_New_Visitor',
        1: None,
        2: 'VisitorType_Returning_Visitor'
    }

    # remove raw Month and VisitorType values before reconstructing one-hot features
    df = df.drop(columns=['Month', 'VisitorType'], errors='ignore')

    for col in month_columns:
        df[col] = 0
    for col in visitor_columns:
        df[col] = 0

    if month_index in month_map and month_map[month_index] is not None:
        df.loc[0, month_map[month_index]] = 1

    if visitor_index in visitor_map and visitor_map[visitor_index] is not None:
        df.loc[0, visitor_map[visitor_index]] = 1

    # align with scaler feature names and fill any missing columns with zeros
    if hasattr(scaler, 'feature_names_in_'):
        feature_names = list(scaler.feature_names_in_)
        df = df.reindex(columns=feature_names, fill_value=0)

    # scale ALL features first (same as training)
    df_scaled = scaler.transform(df)

    # then select top 10 features using saved selector
    df_selected = selector.transform(df_scaled)

    # make prediction
    prediction = model.predict(df_selected)[0]  # 0 or 1
    probability = model.predict_proba(df_selected)[0][1]  # probability of Revenue=1

    result = {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "message": "Will Purchase" if prediction == 1 else "Will Not Purchase"
    }

    return result





if __name__ == "__main__":
    # sample input — all 17 features (Revenue excluded as it is target)
    sample_input = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 1,
        "ProductRelated_Duration": 0.0,
        "BounceRates": 0.2,
        "ExitRates": 0.2,
        "PageValues": 0.0,
        "SpecialDay": 0.0,
        "Month": 2,
        "OperatingSystems": 1,
        "Browser": 1,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": 2,
        "Weekend": 0
    }

    result = predict(sample_input)  # run prediction
    print(result)  # print result