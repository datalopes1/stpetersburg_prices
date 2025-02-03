import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from utils import split_data
from predict import load_model
from sklearn.metrics import mean_squared_error, root_mean_squared_error,mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import KFold, cross_val_score

def evaluate_model(model, X, y):

    y_pred = model.predict(X)

    print(f"Métricas\n{'-' * 30}")
    print(f"MSE: {mean_squared_error(y, y_pred)}")
    print(f"RMSE: {root_mean_squared_error(y, y_pred)}")
    print(f"MAE: {mean_absolute_error(y, y_pred)}")
    print(f"R2 Score: {r2_score(y, y_pred)}")
    return y_pred

def cross_validate_model(model, X, y, cv_splits = 5):
    scoring = make_scorer(mean_squared_error)
    cv = KFold(n_splits = cv_splits, shuffle = True, random_state = 42)

    scores = cross_val_score(model, X, y, cv = cv, scoring = scoring)
    print(f"Média do RMSE: {np.sqrt(scores).mean()}")
    print(f"Desvio Padrão do RMSE: {np.sqrt(scores).std()}")

    return scores.mean(), scores.std(), scores

if __name__ == "__main__":
    data = pd.read_csv("data/processed/test_data.csv")
    model = load_model("models/regressor.pkl")

    X = data.drop(columns = ['log_price'], axis = 1)
    y = data['log_price']

    X_train, X_test, y_train, y_test = split_data(X, y)

    mean_rmse, std_rmse, all_scores = cross_validate_model(model, X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    