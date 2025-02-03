import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path, sep = '\t')
    df = df.drop(columns = ["total_images", "first_day_exposition", "days_exposition"], axis = 1)
    df["studio"] = df["studio"].map({"True": 1, "False": 0})
    df["open_plan"] = df["open_plan"].map({"True": 1, "False": 0})
    df["log_price"] = np.log1p(df["last_price"])
    
    return df

def split_data(X, y, test_size = 0.20, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test 