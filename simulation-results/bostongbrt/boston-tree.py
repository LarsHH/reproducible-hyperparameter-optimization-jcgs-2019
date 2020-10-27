import sherpa
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import sherpa

print("Trial starting")

client = sherpa.Client()
trial = client.get_trial()

args = trial.parameters
lr = 0.1 * args.pop('lr_scaler')

print("Trial {} training".format(trial.id))

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=1234, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=5678, test_size=0.2)

clf = GradientBoostingRegressor(learning_rate=lr, **args)
clf.fit(X_train, y_train)

predicted_valid = clf.predict(X_valid)
expected_valid = y_valid
mse_valid = np.mean((predicted_valid - expected_valid) ** 2)

predicted_test = clf.predict(X_test)
expected_test = y_test
mse_test = np.mean((predicted_test - expected_test) ** 2)

print("Trial {} done".format(trial.id))

client.send_metrics(trial=trial, iteration=1,
                    objective=mse_valid, context={'test_mse': mse_test})