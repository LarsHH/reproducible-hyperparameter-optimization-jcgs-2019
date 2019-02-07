print("HELLLOOOO")
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
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=1234)

clf = GradientBoostingRegressor(learning_rate=lr, **args)
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
expected = y_test

mse = np.mean((predicted - expected) ** 2)

print("Trial {} done".format(trial.id))

client.send_metrics(trial=trial, iteration=1,
                    objective=mse)