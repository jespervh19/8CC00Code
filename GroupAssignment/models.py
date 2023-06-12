import sklearn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class Model():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def preprocessing(self):
        min_val = np.min(self.x, axis=0)
        max_val = np.max(self.x, axis=0)
        self.x = (self.x - min_val) / (max_val - min_val)
        self.x = np.nan_to_num(self.x, nan=0)
        return self.x

    def train(self):
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0)

        # Create a PCA object
        pca = PCA(n_components=0.9)

        # Create a classifier (e.g., Random Forest Classifier)
        regressor = LogisticRegression()

        # Create a pipeline with PCA and the classifier
        self.pipeline = Pipeline([('pca', pca), ('regressor', regressor)])

        # Fit the pipeline on the training data
        self.pipeline.fit(self.X_train, self.y_train)

        return self.pipeline
    
    def test(self):
        # Evaluate the model
        accuracy = self.pipeline.score(self.X_test, self.y_test)

        return accuracy
    
    def predict(self, X_new):
        # Predict using the trained pipeline
        y_pred = self.pipeline.predict_proba(X_new)[:, 0]
        y_pred = np.sort(y_pred)[::-1][:100]

        return y_pred