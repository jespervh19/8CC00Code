import sklearn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        # Create a PCA object
        pca = PCA(n_components=20)

        # Create a classifier (e.g., Random Forest Classifier)
        classifier = RandomForestClassifier()

        # Create a pipeline with PCA and the classifier
        self.pipeline = Pipeline([('pca', pca), ('classifier', classifier)])

        # Fit the pipeline on the training data
        self.pipeline.fit(self.X_train, self.y_train)

        return self.pipeline
    
    def test(self):
        # Predict using the trained pipeline
        self.y_pred = self.pipeline.predict(self.X_test)

        # Evaluate the model
        accuracy = accuracy_score(self.y_test, self.y_pred)

        return accuracy
    
    def predict(self, X_new):
        # Predict using the trained pipeline
        y_pred = self.pipeline.predict(X_new)

        return y_pred