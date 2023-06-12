import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def get_molecular_descriptors(data, remove_fingerprints=False):
    # Create an empty DataFrame for descriptors
    descriptors = pd.DataFrame()

    # Iterate over the SMILES column
    for i, row in data.iterrows():
        smiles = row['SMILES']
        molecule = Chem.MolFromSmiles(smiles)
        
        # Calculate descriptors
        descriptor_values = {}
        for descriptor_name, descriptor_function in Descriptors.descList:
            if remove_fingerprints:
                if 'fr_' not in descriptor_name:
                    descriptor_values[descriptor_name] = descriptor_function(molecule)
            else:
                descriptor_values[descriptor_name] = descriptor_function(molecule)            

        # Append descriptor values to the DataFrame
        descriptors = pd.concat([descriptors, pd.DataFrame(descriptor_values, index=[i])], axis=0)
    return descriptors

def get_labels(data):
    labels = data
    return labels