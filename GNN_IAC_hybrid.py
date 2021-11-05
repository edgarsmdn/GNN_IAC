'''
Project: GNN_IAC

                               GNN_IAC_Hybrid

Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
jj = 0      # To change of method

# Scientific computing
import numpy as np
import pandas as pd

# RDKiT
from rdkit import Chem

# Sklearn
from sklearn.model_selection import train_test_split

# Internal utilities
from GNN_architecture import GNN
from utilities.mol2graph import get_dataloader_pairs, mol2torchdata
from utilities.Train_eval import train, eval, MAE, R2
from utilities.save_info import save_train_traj

# External utilities
#from tqdm import tqdm
#tqdm.pandas()
from collections import OrderedDict
import copy
import time
import os

# Pytorch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr

model_name = 'GNN_IAC'
path = os.getcwd()

methods = ['Hildebrand', 'HSP', 'COSMO_RS', 'UNIFAC', 'mod_UNIFAC_Ly', 
           'mod_UNIFAC_Do', 'Abraham', 'MOSCED']

method_name = methods[jj]

# Create folder for method
path_method = path + '/0' + str(jj+1) + '_' + method_name
if not os.path.exists(path_method):
    os.makedirs(path_method)
# Create folder for Hybrid
path_SPEC = path_method + '/Hybrid'
if not os.path.exists(path_SPEC):
    os.makedirs(path_SPEC)
    
# Open report file
report = open(path_SPEC+'/Report_training_' + model_name + '.txt', 'w')
def print_report(string, file=report):
    print(string)
    file.write('\n' + string)

print_report(' Report for ' + model_name)
print_report('-'*50)

########################
# --- Prepare data --- #
########################
df              = pd.read_csv('Data/database_IAC_ln_clean.csv')    
df_split        = pd.read_csv('Data/database_IAC_ln_clean.csv')   # To save splits used         

# Get only feasible molecules for the method
df              = df[df[method_name].notna()]
df['Error']     = df['Literature'].to_numpy() - df[method_name].to_numpy()
target          = 'Error'

# Build molecule from SMILE
mol_column_solvent     = 'Molecule_Solvent'
df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

mol_column_solute      = 'Molecule_Solute'
df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)

##################################
# --- Train_total/test split --- #
##################################
indices = df.index.tolist()
train_total_index, test_index = train_test_split(indices, test_size=0.20, random_state=0)

print_report('Total points       : ' + str(df.shape[0]))
print_report('Train total points : ' + str(len(train_total_index)))
print_report('Test points        : ' + str( len(test_index)))

#############################
# --- Train/Valid split --- #
#############################
# Define scaler
y_scaler   = None

# Construct graphs from molecules
graph_column_solvent      = 'Graphs_Solvent'
df[graph_column_solvent]  = mol2torchdata(df, mol_column_solvent, target, y_scaler)

graph_column_solute      = 'Graphs_Solute'
df[graph_column_solute]  = mol2torchdata(df, mol_column_solute, target, y_scaler)

#################################
# --- GNN ensemble training --- #
#################################
n_ensembles = 30

# Hyperparameters
num_layer   = 5
drop_ratio  = 0.1
conv_dim    = 30
lr          = 0.001
n_ms        = 64
n_epochs    = 200
batch_size  = 32
mlp_layers  = 3
mlp_dims    = [50, 25, 1]


for e in range(1, n_ensembles+1):
    train_index, valid_index = train_test_split(train_total_index, test_size=0.10)
    
    # Save splits used
    if e == 1:
        print_report('\nGNN datasets size')
        print_report('-'*40)
        print_report('Train      : ' + str(len(train_index)))
        print_report('Validation : ' + str(len(valid_index)))
        print_report('Test       : ' + str(len(test_index)))
    
    spliting_values = [0]*df_split.shape[0]
    for k in range(df_split.shape[0]):
        if k in train_index:
            spliting_values[k] = 'Train'
        elif k in valid_index:
            spliting_values[k] = 'Valid'
        elif k in test_index:
            spliting_values[k] = 'Test'
        else:
            spliting_values[k] = ''
    df_split['Ensemble_'+str(e)] = spliting_values
    
    start       = time.time()

    # Data loaders
    train_loader = get_dataloader_pairs(df, train_index, target, graph_column_solvent, graph_column_solute, batch_size, shuffle=True, drop_last=True)
    valid_loader = get_dataloader_pairs(df, valid_index, target, graph_column_solvent, graph_column_solute, batch_size, shuffle=False, drop_last=True)
    test_loader  = get_dataloader_pairs(df, test_index, target, graph_column_solvent, graph_column_solute, batch_size,  shuffle=False, drop_last=True)
    
    # Model
    model    = GNN(num_layer=num_layer, drop_ratio=drop_ratio, conv_dim=conv_dim, 
               gnn_type='NNConv', JK='mean', graph_pooling='set2set',
               neurons_message=n_ms, mlp_layers=mlp_layers, mlp_dims=mlp_dims)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = model.to(device)
    
    # Optimizer                                                           
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)     
    task_type = 'regression'
    scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7, verbose=False)
    
    # To save trajectory
    mae_train = []; r2_train = []; train_loss = []
    mae_valid = []; r2_valid = []; valid_loss = []
    mae_test  = []; r2_test  = []
    
    pbar = range(n_epochs)#tqdm(range(n_epochs))
    best_MAE = np.inf
    
    for epoch in pbar:
        stats = OrderedDict()
        # Train
        stats.update(train(model, device, train_loader, optimizer, task_type, stats))
      
        # Evaluation
        stats.update(eval(model, device, train_loader, MAE, stats, 'Train', task_type))
        stats.update(eval(model, device, valid_loader, MAE, stats, 'Valid', task_type))
        stats.update(eval(model, device, test_loader, MAE, stats, 'Test', task_type))
        stats.update(eval(model, device, train_loader, R2, stats, 'Train', task_type))
        stats.update(eval(model, device, valid_loader, R2, stats, 'Valid', task_type))
        stats.update(eval(model, device, test_loader, R2, stats, 'Test', task_type))
        
        # Scheduler
        scheduler.step(stats['MAE_Valid'])
      
        # Save info
        train_loss.append(stats['Train_loss'])
        valid_loss.append(stats['Valid_loss'])
        mae_train.append(stats['MAE_Train']); r2_train.append(stats['R2_Train'])
        mae_valid.append(stats['MAE_Valid']); r2_valid.append(stats['R2_Valid'])
        mae_test.append(stats['MAE_Test']);   r2_test.append(stats['R2_Test'])
        
        # Save best model
        if mae_valid[-1] < best_MAE:
            best_model = copy.deepcopy(model.state_dict())
            best_MAE   = mae_valid[-1]
      
        #pbar.set_postfix(stats) # include stats in the progress bar

    if task_type=='classification':
        pass
    elif task_type=='regression':
        best_val_epoch = np.argmin(np.array(mae_valid))
      
    print_report('\n\nEnsemble model ' + str(e))
    print_report('-'*30)
    print_report('Best epoch     : '+ str(best_val_epoch))
    print_report('Training MAE   : '+ str(mae_train[best_val_epoch]))
    print_report('Validation MAE : '+ str(mae_valid[best_val_epoch]))
    print_report('Test MAE       : '+ str(mae_test[best_val_epoch]))
    print_report('Training R2    : '+ str(r2_train[best_val_epoch]))
    print_report('Validation R2  : '+ str(r2_valid[best_val_epoch]))
    print_report('Test R2        : '+ str(r2_test[best_val_epoch]))
    
    # Save training trajectory
    df_model_training = pd.DataFrame(train_loss, columns=['Train_loss'])
    df_model_training['Valid_loss'] = valid_loss
    df_model_training['MAE_Train']  = mae_train
    df_model_training['MAE_Valid']  = mae_valid
    df_model_training['MAE_Test']   = mae_test
    df_model_training['R2_Train']   = r2_train
    df_model_training['R2_Valid']   = r2_valid
    df_model_training['R2_Test']    = r2_test
    
    path_model_info = path_SPEC + '/Ensemble_' + str(e)
    save_train_traj(path_model_info, df_model_training)
    
    # Save best model
    torch.save(best_model, path_model_info + '/Ensemble_' + str(e) + '.pth')
    
    end       = time.time()
    print_report('\nTraining time (min): ' + str((end-start)/60))


df_split.to_csv(path_SPEC +'/Split_'+model_name+'.csv', index=False) # Save splits
report.close()