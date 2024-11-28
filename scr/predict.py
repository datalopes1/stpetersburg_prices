import os
import pandas as pd
import numpy as np

class Predictor:
    def __init__(self, model_path, data_path, output_path):
        self.model_path = model_path
        self.data_path = data_path
        self.output_path = output_path
        self.pipeline = None
        self.data = None
        self.predictions = None

    def load_model(self):
        model_series = pd.read_pickle(self.model_path)
        self.model = model_series['model']
        print("Modelo carregado.")
        
    def load_data(self):
        self.data = pd.read_csv(self.data_path, sep = "\t")
        print("Dados carregados.")

    def predict(self):
        features = ['total_area', 'rooms', 'floors_total', 'living_area', 'floor', 'studio', 'open_plan', 'kitchen_area', 'locality_name', 'airports_nearest', 'cityCenters_nearest','parks_around3000', 'ponds_around3000']
        X = self.data[features]

        self.predictions = self.model.predict(X)
        print("Predições realizadas.")

        results = pd.DataFrame({
            'Locality': self.data["locality_name"],
            'Price': self.data["last_price"],
            'Predictions': np.expm1(self.predictions)
        })
        return results
    
    def save_predictions(self, results):
        results.to_excel(self.output_path, index = False)
        print(f"Resultados salvos em: {self.output_path}")

if __name__ == "__main__":

    model_path = "models/regressor.pkl"
    data_path = "data/raw/real_estate_data.csv"
    output_path = "data/processed/predictions.xlsx"

    predictor = Predictor(model_path, data_path, output_path)

    predictor.load_model()
    predictor.load_data()
    results = predictor.predict()
    predictor.save_predictions(results)