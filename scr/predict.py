import pandas as pd
import numpy as np
import joblib

def load_model(path):
    return joblib.load(path)

def make_predictions(model, data):
    predictions = model.predict(data)
    predictions = np.expm1(predictions)
    return predictions

if __name__ == "__main__":
    data = pd.read_csv("data/processed/test_data.csv")
    model = load_model("models/regressor.pkl")

    pred_price = make_predictions(model, data)
    data.rename(columns = {'log_price': 'price'}, inplace = True)
    data['price'] = np.expm1(data['price'])
    data['pred_price'] = pred_price
    data['diff_percentage'] = (((data['pred_price'] / data['price']) - 1) * 100).round(2)

    data = data[['locality_name', 'price', 'pred_price', 'diff_percentage']]
    data.to_excel("data/processed/predictions.xlsx", index = False)
    print("\nPredictions saved to 'data/processed/predictions.xlsx'")