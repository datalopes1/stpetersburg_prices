# %%
# Manipulação de dados
import pandas as pd
import numpy as np

# Análise Exploratória de Dados
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from xgboost import XGBRegressor

# Pré-processamento
import optuna
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from feature_engine import imputation
from category_encoders import TargetEncoder
# %%
df = pd.read_csv("../../data/raw/real_estate_data.csv", sep = '\t')
df.head()
# %%
colunas = df.isnull().mean()
filtro = colunas[colunas < 0.30].index
df = df[filtro]
df['log_last_price'] = np.log1p(df['last_price'])
df.head()
# %%
features = df.drop(columns = ['last_price', 'log_last_price', 'first_day_exposition'], axis = 1).columns.to_list()
target = 'log_last_price'

X = df[features]
y = df[target]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=21)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25, random_state=21)
# %%
num_features = X_train.select_dtypes(include = 'number').columns.to_list()
cat_features = X_train.select_dtypes(exclude = 'number').columns.to_list()
#%%
num_transformer = Pipeline([
    ('imput', imputation.MeanMedianImputer(imputation_method='median'))
])

cat_transformer = Pipeline([
    ('imput', imputation.CategoricalImputer(imputation_method='frequent')),
    ('encoder', TargetEncoder())
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)
# %%
param_dict = {'n_estimators': 867, 
              'learning_rate': 0.09396356090246702, 
              'max_depth': 6, 
              'subsample': 0.6955868105322172, 
              'colsample_bytree': 0.40079564093218806, 
              'min_child_weight': 3}

model = XGBRegressor(objective='reg:squarederror', **param_dict,random_state = 21)

reg = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', model)
])

reg.fit(X_train, y_train)
# %%
y_pred = reg.predict(X_val)

mse = metrics.mean_squared_error(y_val, y_pred)
rmse = metrics.mean_squared_error(y_val, y_pred, squared = False)
mae = metrics.mean_absolute_error(y_val, y_pred)
r2 = metrics.r2_score(y_val, y_pred)

metricas = {
    'Métrica': 'Resultado',
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae,
    'R2 Score': r2
}
metricas
# %%
y_pred_test = reg.predict(X_test)

mse = metrics.mean_squared_error(y_test, y_pred_test)
rmse = metrics.mean_squared_error(y_test, y_pred_test, squared = False)
mae = metrics.mean_absolute_error(y_test, y_pred_test)
r2 = metrics.r2_score(y_test, y_pred_test)

metricas = {
    'Métrica': 'Resultado',
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae,
    'R2 Score': r2
}
metricas
# %%
fig, ax = plt.subplots(figsize = (12, 6))
sns.scatterplot(x = y_test, y = y_pred_test)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color = 'red')
ax.set_title("Real x Predito", loc = 'left')
ax.set_xlabel("Real")
ax.set_ylabel("Predito")
plt.show()
# %%
model_series = pd.Series({
    'model': reg,
    'features': features,
    'metricas': metricas
})
model_series.to_pickle("../../models/reg_model.pkl")