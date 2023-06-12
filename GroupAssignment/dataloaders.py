import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

class DataLoader:
    def __init__(self, file):
        # Read the csv file
        self.data = pd.read_csv(file)

    def get_molecular_descriptors(self):
        # Create an empty DataFrame for descriptors
        descriptors = pd.DataFrame()

        # Iterate over the SMILES column
        for i, row in self.data.iterrows():
            smiles = row['SMILES']
            molecule = Chem.MolFromSmiles(smiles)
            
            # Calculate descriptors
            descriptor_values = {}
            for descriptor_name, descriptor_function in Descriptors.descList:
                if 'fr_' not in descriptor_name:
                    descriptor_values[descriptor_name] = descriptor_function(molecule)
            
            # Append descriptor values to the DataFrame
            descriptors = pd.concat([descriptors, pd.DataFrame(descriptor_values, index=[i])], axis=0)
                        
        return descriptors.values
    
    def get_labels(self):
        labels = self.data
        return labels