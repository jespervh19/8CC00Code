import sklearn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def train(X_train, y_train, num_components):
    y_train = y_train["ALDH1_inhibition"]
    # Create a PCA object
    pca = PCA(n_components=num_components)

    # Create a classifier (e.g., Random Forest Classifier)
    regressor = LogisticRegression(max_iter=2500)

    # Create a pipeline with PCA and the classifier
    pipeline = Pipeline([('pca', pca), ('regressor', regressor)])

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_train)
    accuracy = np.round(accuracy_score(y_train, y_pred),3)
    print(f"Train accuracy = {accuracy}")

    return pipeline

def test(pipeline, X_test, y_test):
    y_test = y_test["ALDH1_inhibition"]
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = np.round(accuracy_score(y_test, y_pred),3)
    print(f"Test accuracy = {accuracy}")

    cf_matrix = confusion_matrix(y_test, y_pred)
    print("\nTest confusion_matrix")
    sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)

def predict(pipeline, X_new, labels):
    # Predict using the trained pipeline
    y_pred = pipeline.predict_proba(X_new)[:, 0]
    combined_data = pd.concat([labels, pd.DataFrame(y_pred, columns=['ALDH1_inhibition'])], axis=1)
    combined_data = combined_data.sort_values(by="ALDH1_inhibition", ascending=False)
    return combined_data