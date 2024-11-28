import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from category_encoders import TargetEncoder 

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path, sep = '\t')
        self._process_data()
        return self.data
    
    def _process_data(self):
        self.data['studio'] = self.data['studio'].map({"True": 1, "False": 0})
        self.data['open_plan'] = self.data['open_plan'].map({"True": 1, "False": 0})
        self.data['log_price'] = np.log1p(self.data["last_price"])

class Preprocessor:
    def __init__(self, cat_features, num_features):
        self.cat_features = cat_features
        self.num_features = num_features
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        cat_transformer = Pipeline([
            ('imput', CategoricalImputer(imputation_method='frequent')),
            ('encoding', TargetEncoder())
        ])

        num_transformer = Pipeline([
            ('imput', MeanMedianImputer(imputation_method = 'median'))
        ])

        return ColumnTransformer(
            transformers = [
                ('cat', cat_transformer, self.cat_features),
                ('num', num_transformer, self.num_features)
            ])
    
class ModelTrainer:
    def __init__(self, model, preprocessor, params):
        self.model = model
        self.preprocessor = preprocessor
        self.params = params
        self.pipeline = None

    def build_pipeline(self):
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', self.model(**self.params, random_state = 42))
        ])

    def train(self, X_train, y_train):
        if self.pipeline is None:
            raise ValueError("Pipeline não constrúido. Instancie build_pipeline() primeiro.")
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        return {
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': mean_squared_error(y_test, y_pred, squared = False),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }

class ModelSaver:
    @staticmethod
    def save_model(model, features, metrics, filepath):
        model_series = pd.Series({
            'model': model,
            'features': features,
            'metrics': metrics
        })
        model_series.to_pickle(filepath)

    @staticmethod
    def save_data(data, filepath):
        data.to_csv(filepath, index = False)

if __name__ == "__main__":

    data_loader = DataLoader(file_path="data/raw/real_estate_data.csv")
    df = data_loader.load_data()

    features = ['total_area', 'rooms', 'floors_total', 'living_area', 'floor', 'studio', 'open_plan', 'kitchen_area', 'locality_name', 'airports_nearest', 'cityCenters_nearest','parks_around3000', 'ponds_around3000']
    target = 'log_price'

    X = df[features]
    y = df[target]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    cat_features = ['studio', 'open_plan', "locality_name"]
    num_features = ['total_area', 'rooms', 'floors_total', 'living_area', 'floor', 'kitchen_area', 'airports_nearest', 'cityCenters_nearest', 'parks_around3000', 'ponds_around3000']

    preprocessor = Preprocessor(cat_features, num_features).pipeline

    params = {
        'objective': 'reg:squarederror', 
        'n_estimators': 1000,
        'verbosity': 0,
        'learning_rate': 0.04939478561082139, 
        'max_depth': 7, 
        'subsample': 0.7882157209337015, 
        'colsample_bytree': 0.6095977710612492, 
        'min_child_weight': 18
    }

    trainer = ModelTrainer(XGBRegressor, preprocessor, params)
    trainer.build_pipeline()
    trainer.train(X_train, y_train)

    metrics = trainer.evaluate(X_test, y_test)
    print(metrics)

    ModelSaver.save_model(trainer.pipeline, features, metrics, "models/regressor.pkl")
    print("Modelo salvo na pasta models")