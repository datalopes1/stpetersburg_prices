import pandas as pd
import xgboost as xgb
import joblib
from sklearn.pipeline import Pipeline
from utils import load_data, split_data
from data_preprocessing import get_preprocessor

def train_model(X_train, y_train):

    params = {'objective': 'reg:squarederror', 
              'n_estimators': 1000,
              'verbosity': 0,
              'learning_rate': 0.04939478561082139, 
              'max_depth': 7, 
              'subsample': 0.7882157209337015, 
              'colsample_bytree': 0.6095977710612492, 
              'min_child_weight': 18}
    
    model = xgb.XGBRegressor(**params, random_state = 42)
    model.fit(X_train, y_train)
    print("\nTraining complete.")
    return model

def save_model(model, preprocessor, path):

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    joblib.dump(pipeline, path)
    print(f"'Model saved to {path}'")


if __name__ == "__main__":
    data = load_data("data/raw/real_estate_data.csv")

    features = ['total_area', 'rooms', 'floors_total', 'living_area', 'floor', 'studio', 'open_plan', 'kitchen_area', 'locality_name', 'airports_nearest', 'cityCenters_nearest','parks_around3000', 'ponds_around3000']
    target = 'log_price'

    preprocessor = get_preprocessor()
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = split_data(X, y)
    train_data = pd.concat([X_train, y_train], axis = 1)
    train_data.to_csv("data/processed/train_data.csv", index = False)
    test_data = pd.concat([X_test, y_test], axis = 1)
    test_data.to_csv("data/processed/test_data.csv", index = False)

    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_train_transformed = pd.DataFrame(X_train_transformed)

    model = train_model(X_train_transformed, y_train)
    save_model(model, preprocessor, "models/regressor.pkl")