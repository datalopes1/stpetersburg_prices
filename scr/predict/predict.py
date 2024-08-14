# %% 
import pandas as pd
import numpy as np
# %%
model = pd.read_pickle("../../models/reg_model.pkl")
model
# %%
df = pd.read_csv("../../data/raw/real_estate_data.csv", sep = '\t')
df.head()
# %%
X = df[model['features']]
y_pred = np.expm1(model['model'].predict(X))
# %%
df['predPrice'] = y_pred
results = df[['locality_name','last_price', 'predPrice']].copy()
results.to_excel("../../data/processed/model_predictions.xlsx", index = False)
# %%
