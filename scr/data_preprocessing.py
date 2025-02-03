import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from category_encoders import TargetEncoder

def get_preprocessor():
    cat_features = ['studio', 'open_plan', 'locality_name']
    num_features = ['total_area', 'rooms', 'floors_total', 'living_area', 'floor', 'kitchen_area', 'airports_nearest', 'cityCenters_nearest', 'parks_around3000', 'ponds_around3000']

    cat_transformer = Pipeline([
    ('cat_imput', CategoricalImputer(imputation_method = 'frequent')),
    ('cat_encoding', TargetEncoder())
    ])

    num_transformer = Pipeline([
        ('num_imput', MeanMedianImputer(imputation_method = 'median'))
    ])

    preprocessor = ColumnTransformer(
        transformers = [
            ('cat', cat_transformer, cat_features),
            ('num', num_transformer, num_features)
        ],
        remainder='passthrough'
    )

    return preprocessor