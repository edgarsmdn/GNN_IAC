'''
Project: GNN_IAC

                    Custom dataloader for 2 graphs-1 output

Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader

# For the example
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from mol2graph import mol2torchdata

class PairDataset(Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __getitem__(self, idx):
        return self.datasetA[idx], self.datasetB[idx]
    
    def __len__(self):
        return len(self.datasetA)
    

def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB

# For 2Graph-1Output
def get_dataloader_pairs(df, index, target, graphs_solvent, graphs_solute, batch_size, shuffle=False, drop_last=False):
    
    x_solvent = df.loc[index, graphs_solvent].tolist() # Get graphs for solvent
    x_solute  = df.loc[index, graphs_solute].tolist()  # Get graphs for solute
    
    pair_dataset = PairDataset(x_solvent, x_solute)
    data_loader  = DataLoader(pair_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate) 
    
    return data_loader


#########################
# --- Example pairs --- #
#########################

df = pd.read_csv('../Data/database_IAC_backup.csv')
target          = 'Literature'
exp_values      = df[target] 

# Build molecule from SMILE
mol_column_solvent     = 'Molecule_Solvent'
df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

mol_column_solute      = 'Molecule_Solute'
df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)

##################################
# --- Train_total/test split --- #
##################################
indices = df.index.tolist()
train_total_index, test_index = train_test_split(indices, test_size=0.10)
y = df[target].values.reshape(-1,1)         # Extract target values

y_scaler   = None
# Construct graphs from molecules
graph_column_solvent      = 'Graphs_Solvent'
df[graph_column_solvent]  = mol2torchdata(df, mol_column_solvent, target, y_scaler)

graph_column_solute      = 'Graphs_Solute'
df[graph_column_solute]  = mol2torchdata(df, mol_column_solute, target, y_scaler)

train_index, valid_index = train_test_split(train_total_index, test_size=0.10)


data_loader_ex = get_dataloader_pairs(df, train_index, target, 
                                      graph_column_solvent, 
                                      graph_column_solute, 
                                      batch_size=32, 
                                      shuffle=True, 
                                      drop_last=False)


i = 0
for batch_solvent, batch_solute in data_loader_ex:
    if i==0:
        print('\nFirst Batch of solvents')
        print(batch_solvent)
        print('\nFirst Batch of solutes')
        print(batch_solute)
    i +=1

print('\n\n Number of systems in train dataloader')
print(len(data_loader_ex.dataset))

print('\n\n First system in the dataloader')
print(data_loader_ex.dataset[0])
print('\n\n')

print('Solvent smiles: \n', df['Solvent_SMILES'].loc[df['Graphs_Solvent'] == data_loader_ex.dataset[0][0]])
print('\nSolute smiles : \n', df['Solute_SMILES'].loc[df['Graphs_Solute'] == data_loader_ex.dataset[0][1]])






