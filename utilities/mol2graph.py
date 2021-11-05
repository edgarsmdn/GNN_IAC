'''
Project: GNN_IAC

                              IAC mol2graph specific 

Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''

from rdkit import Chem
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

##################################################
# --- Parameters to be specified by the user --- #
##################################################

'''
possible_atom_list: Symbols of possible atoms

possible_atom_degree: The degree of an atom is defined to be its number of directly-bonded neighbors. 
                        The degree is independent of bond orders, but is dependent on whether or not Hs are 
                        explicit in the graph.
                        
possible_atom_valence: The implicit atom valence: number of implicit Hs on the atom

possible_hybridization: Type of Hybridazation. Nice reminder here:
    https://www.youtube.com/watch?v=vHXViZTxLXo
    
NOTE!!: Bond features are within the function bond_features

'''

possible_atom_list = ['C', 'Br', 'Cl', 'N', 'O', 'I', 'S', 'F', 'P']

possible_hybridization = [Chem.rdchem.HybridizationType.SP, 
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3]

possible_num_bonds = [1,2,3,4]

possible_formal_charge = [0,1,-1]

possible_num_Hs  = [0,1,2,3]

#########################
# --- Atom features --- #
#########################

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]                               # Specify as Unknown
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    '''
    Get atom features
    '''
    Symbol       = atom.GetSymbol()
    
    # Features
    Type_atom     = one_of_k_encoding(Symbol, possible_atom_list)
    Ring_atom     = [atom.IsInRing()]
    Aromaticity   = [atom.GetIsAromatic()]
    Hybridization = one_of_k_encoding(atom.GetHybridization(), possible_hybridization)
    Bonds_atom    = one_of_k_encoding(len(atom.GetNeighbors()), possible_num_bonds)
    Formal_charge = one_of_k_encoding(atom.GetFormalCharge(), possible_formal_charge)
    num_Hs        = one_of_k_encoding(atom.GetTotalNumHs(), possible_num_Hs)
    
    # Merge features in a list
    results = Type_atom + Ring_atom + Aromaticity + Hybridization + \
        Bonds_atom + Formal_charge + num_Hs
    
    return np.array(results).astype(np.float32)

#########################
# --- Bond features --- #
#########################

def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

def bond_features(bond):
    '''
    Get bond features
    '''
    bt = bond.GetBondType()
    
    # Features
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]
    return np.array(bond_feats).astype(np.float32)


###################################
# --- Molecule to torch graph --- #
###################################
def mol2torchdata(df, mol_column, target, y_scaler=None):
    '''
    Takes a molecule and return a graph
    '''
    graphs=[]
    mols = df[mol_column].tolist()
    ys   = df[target].tolist()
    for mol, y in zip(mols, ys):
        atoms  = mol.GetAtoms()
        bonds  = mol.GetBonds()
        
        # Information on nodes
        node_f = [atom_features(atom) for atom in atoms]
        
        # Information on edges
        edge_index = get_bond_pair(mol)
        edge_attr  = []
        
        for bond in bonds:
            edge_attr.append(bond_features(bond))
            edge_attr.append(bond_features(bond))
        
        # Store all information in a graph
        nodes_info = torch.tensor(node_f, dtype=torch.float)
        edges_indx = torch.tensor(edge_index, dtype=torch.long)
        edges_info = torch.tensor(edge_attr, dtype=torch.float)
        
        
        graph = Data(x=nodes_info, edge_index=edges_indx, edge_attr=edges_info)
        
        if y_scaler != None:
            y = np.array(y).reshape(-1,1)
            y = y_scaler.transform(y).astype(np.float32)
            graph.y = torch.tensor(y[0], dtype=torch.float)
        else:
            #y = y.astype(np.float32)
            graph.y = torch.tensor(y, dtype=torch.float)
        
        
        graphs.append(graph)
    
    return graphs

##########################
# --- Count features --- #
##########################

def n_atom_features():
    atom = Chem.MolFromSmiles('CC').GetAtomWithIdx(0)
    return len(atom_features(atom))


def n_bond_features():
    bond = Chem.MolFromSmiles('CC').GetBondWithIdx(0)
    return len(bond_features(bond))

#######################
# --- Data loader --- #
#######################

def get_dataloader(df, index, target, graph_column, batch_size, shuffle=False, drop_last=False):
    
    x = df.loc[index, graph_column].tolist() # Get graphs (x)
    data_loader = DataLoader(x, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) 
    
    # Note: drop_last argument drops the last non-full batch of each worker’s 
    #       iterable-style dataset replica. This ensure all batches to be of equal size
    
    return data_loader

######################################
# --- Data loader 2Graphs_1Output--- #
######################################

from torch_geometric.data import Batch
from torch.utils.data import Dataset

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
    data_loader  = torch.utils.data.DataLoader(pair_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate) 
    
    return data_loader


