import sklearn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def train(X, y, num_components, degrees=[1,2], use_pca=[True, False]):
    y = y["ALDH1_inhibition"]
    # Create a PCA object
    best_score = 0
    best_model = None
    for use_pca_value in use_pca:
        for degree in degrees:
            poly = PolynomialFeatures(degree=degree, include_bias=False)

            pca = PCA(n_components=num_components)

            # Create a classifier (e.g., Random Forest Classifier)
            regressor = LogisticRegression(class_weight='balanced', max_iter=2500)

            if use_pca_value:
                # Create a pipeline with PCA and the classifier
                pipeline = Pipeline([('poly', poly), ('pca', pca), ('regressor', regressor)])
            else:
                pipeline = Pipeline([('poly', poly), ('regressor', regressor)])

            scores = cross_val_score(pipeline, X, y, cv=5)
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_model = pipeline

            print(f"Degree: {degree}, Use_PCA: {use_pca_value}")
            print("Cross-validation scores:", scores)
            print("Average score:", avg_score)
    
    best_model.fit(X,y)
    return best_model

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

def predict(pipeline, X_new, y_new):
    # Predict using the trained pipeline
    y_pred = pipeline.predict_proba(X_new)[:, 0]
    combined_data = pd.concat([y_new, pd.DataFrame(y_pred, columns=['ALDH1_inhibition'])], axis=1)
    combined_data = combined_data.sort_values(by="ALDH1_inhibition", ascending=False)
    return combined_data