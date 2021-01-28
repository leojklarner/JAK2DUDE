"""
A script that creates an XGBoost model and evaluates it
on the ChEMBL data.
"""

from learning import load
import os
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# get loader that has been initialised on the desired data
path_to_root = os.path.dirname(os.path.dirname(os.getcwd()))
loader = load(path_to_root, ['jak2'])

# calculate EC fingerprints
loader.featurize('fragprints')

# get cross-validation splits
splits = loader.split_and_scale(kfold_shuffle=True, num_splits=3, scale_labels=False)

rmse = []
pearsonsr = []
rsquared = []

i=1

for split in splits:
    print(f"Iteration {i}")
    i = i+1
    X_train, X_test, _, y_train, y_test, _ = split
    model = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    pearsonsr.append(pearsonr(y_test, y_pred)[0])
    rsquared.append(model.score(X_test, y_test))

print("RMSE", np.mean(rmse), np.var(rmse))
print("PearsonR", np.mean(pearsonsr), np.var(pearsonsr))
print("RSquared", np.mean(rsquared), np.var(rsquared))

if __name__ == '__main__':
    print("")