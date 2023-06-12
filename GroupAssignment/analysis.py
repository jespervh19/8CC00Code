import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
  
def correlation(descriptors):
    corr_matrix = descriptors.corr()

    # Create an empty list to store the highly correlated pairs
    highly_correlated_pairs = []

    # Loop through the correlation matrix and find pairs with correlations higher than 0.9
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.9:
                # Add the pair of column names to the list
                pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                highly_correlated_pairs.append(pair)

    # Print the list of highly correlated pairs
    return highly_correlated_pairs

def remove_colinear(descriptors, highly_correlated_pairs):
    # Create a set to keep track of columns to delete
    columns_to_delete = set()

    # Delete one column from each correlated pair
    for pair in highly_correlated_pairs:
        column1, column2 = pair
        if column1 not in columns_to_delete:
            columns_to_delete.add(column2)
        elif column2 not in columns_to_delete:
            columns_to_delete.add(column1)

    # Delete the columns from the original DataFrame
    descriptors = descriptors.drop(columns=list(columns_to_delete))

    return descriptors

def ScaleDescriptors(descriptors):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(descriptors)
    descriptors = pd.DataFrame(scaled_data, columns=descriptors.columns)
    return descriptors

def plot_variance(descriptors, percentage=0.9):
    pca = PCA()
    pca.fit(descriptors)

    # Calculation of cumulative explained variance ratio
    cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_var_ratio >= percentage) + 1

    plt.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio, marker='o')
    plt.xlabel('Number of features [-]')
    plt.ylabel('Cumulative explained variance ratio [-]')
    plt.title('Explained variance ratio vs. number of features')
    plt.axhline(y=percentage, color='r', linestyle='--', label=f'{percentage*100}% Variance')
    plt.axvline(num_components, color='g', linestyle='--', label=f"{num_components} Num components")
    plt.legend()
    plt.show()

    return num_components

def plot_loadings(descriptors, num_components):
    pca = PCA(n_components=num_components)
    pca.fit(descriptors)
    
    # Get the loadings
    loadings = pca.components_

    # Calculate the variance explained by each principal component
    variance_explained = pca.explained_variance_ratio_

    # Create a variable names list
    variable_names = descriptors.columns

    # Set the number of principal components to display on the loading plot
    num_pc = len(loadings)

    # Set the number of variables to display on the loading plot
    num_vars = len(variable_names)

    # Plot the loading plot using a scatter plot
    plt.figure(figsize=(7, 6))
    for i in range(num_vars):
        plt.scatter(loadings[0, i], loadings[1, i], color='b')
        plt.text(loadings[0, i], loadings[1, i], variable_names[i])
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Loadings')
    plt.show()

def feature_rankings(descriptors, num_components):
    # Perform PCA
    pca = PCA(n_components=num_components)  # Choose the number of components that explain 90% of the variance
    principal_components = pca.fit_transform(descriptors)

    # Get the loadings for each feature in the principal components
    loadings = pd.DataFrame(pca.components_.T, columns=['PC{}'.format(i+1) for i in range(num_components)], index=descriptors.columns)

    # Calculate the average absolute loading for each feature
    average_loadings = loadings.abs().mean(axis=1)

    # Rank the features based on average loadings in descending order
    feature_rankings = average_loadings.sort_values(ascending=False)

    return feature_rankings