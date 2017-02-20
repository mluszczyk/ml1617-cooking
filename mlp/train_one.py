import sys

from mlp import train_one, load_split_features


X_train, y_train, val = load_split_features()
train_one(X_train, y_train, val, int(sys.argv[1]))

