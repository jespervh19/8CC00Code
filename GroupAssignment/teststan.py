import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Read the CSV file
data = pd.read_csv('C:/Users/20183449/Desktop/TU/Jaar 5/Q4/Advanced programming and biomedical data analysis/8CC00Code/GroupAssignment/tested_molecules-1.csv')

# Create an empty DataFrame for descriptors
descriptors = pd.DataFrame()

# Iterate over the SMILES column
for i, row in data.iterrows():
    smiles = row['SMILES']
    molecule = Chem.MolFromSmiles(smiles)
    
    # Calculate descriptors
    descriptor_values = {}
    for descriptor_name, descriptor_function in Descriptors.descList:
        descriptor_values[descriptor_name] = descriptor_function(molecule)
    
    # Append descriptor values to the DataFrame
    descriptors = pd.concat([descriptors, pd.DataFrame(descriptor_values, index=[i])], axis=0)

# Merge the original data with the descriptors
merged_data = pd.concat([data, descriptors], axis=1)

# Save the merged data to a new CSV file
merged_data.to_csv('C:/Users/20183449/Desktop/TU/Jaar 5/Q4/Advanced programming and biomedical data analysis/8CC00Code/GroupAssignment/descriptors.csv', index=False)
