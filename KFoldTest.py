import DataParser
from sklearn.model_selection import StratifiedKFold
import numpy as np

def fold():
   skf = StratifiedKFold(n_splits=10)
   X, y = DataParser.parseIris()

   for train_index, test_index in skf.split(X, y):
      print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      # Rodar aprendizado

fold()