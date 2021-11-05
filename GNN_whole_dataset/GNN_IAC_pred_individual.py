'''
Project: GNN_IAC

                                GNN_IAC

Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
from rdkit import Chem
from utilities.mol2graph import mol2torchdata, get_dataloader_pairs
from GNN_architecture import GNN
import torch
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np

model_name = 'GNN_IAC'

######################
# --- Prediction --- #
######################
# -- Prepare data
df              = pd.read_csv('Data/database_IAC_ln_clean.csv')
df_predictions  = pd.read_csv('Data/database_IAC_ln_clean.csv')
df_split        = pd.read_csv('Split_GNN_IAC.csv')

target          = 'Literature'

# Build molecule from SMILE
mol_column_solvent     = 'Molecule_Solvent'
df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

mol_column_solute      = 'Molecule_Solute'
df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)

# Define scaler
y_scaler   = None

# Construct graphs from molecules
graph_column_solvent      = 'Graphs_Solvent'
df[graph_column_solvent]  = mol2torchdata(df, mol_column_solvent, target, y_scaler)

graph_column_solute      = 'Graphs_Solute'
df[graph_column_solute]  = mol2torchdata(df, mol_column_solute, target, y_scaler)

# Dataloader
indices = df.index.tolist()
predict_loader = get_dataloader_pairs(df, indices, target, 
                                      graph_column_solvent, 
                                      graph_column_solute, 
                                      batch_size=df.shape[0], 
                                      shuffle=False, drop_last=False)

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

# Ensemble of models
n_ensembles = 30
path        = os.getcwd()

for e in range(1, n_ensembles+1):
    path_model_info = path + '/Ensemble_' + str(e)
    
    model = GNN(num_layer=num_layer, drop_ratio=drop_ratio, conv_dim=conv_dim, 
                   gnn_type='NNConv', JK='mean', graph_pooling='set2set',
                   neurons_message=n_ms, mlp_layers=mlp_layers, mlp_dims=mlp_dims)

    model.load_state_dict(torch.load(path_model_info + '/Ensemble_' + str(e) + '.pth'))
    
    model.eval()
    with torch.no_grad():
        for batch_solvent, batch_solute in predict_loader:
            with torch.no_grad():
                y_pred = model(batch_solvent, batch_solute).numpy().reshape(-1,)
            
    df_predictions['Ensemble_'+str(e)] = y_pred

# Open report file
report = open('Report_individual_prediction_' + model_name + '.txt', 'w')
def print_report(string, file=report):
    print(string)
    file.write('\n' + string)

print_report(' Report for ' + model_name)
print_report('-'*50)
        
#####################################
# --- Statistics of predictions --- #
#####################################
exp_values   = df_predictions[target]

train_old    = df_split.loc[df_split['Ensemble_1'] == 'Train']
valid_old    = df_split.loc[df_split['Ensemble_1'] == 'Valid']
test_old     = df_split.loc[df_split['Ensemble_1'] == 'Test']

train_index = train_old.index.tolist()
valid_index = valid_old.index.tolist()
test_index  = test_old.index.tolist()

print_report('-'*40)
print_report('Train points: ' + str(len(train_index)))
print_report('Valid points: ' + str(len(valid_index)))
print_report('Test  points: ' + str(len(test_index)))

Y_pred_total      = df_predictions.loc[:, 'Ensemble_1':'Ensemble_'+str(n_ensembles)].to_numpy()

# --- Average model performance
print_report('\nAverage GNN statistics')
print_report('-'*40)

# -- Train
print_report('\nTraining set')
print_report('-'*30)

mae_average,sdep_average,mse_average,rmse_average,r2_average,mape_average=[],[],[],[],[],[]
for i in range(n_ensembles):
    train_old   = df_split.loc[df_split['Ensemble_'+str(i+1)] == 'Train']
    train_index = train_old.index.tolist()
    
    y_true = exp_values.values[train_index]
    y_true = np.exp(y_true) 
    Y_pred = Y_pred_total[train_index]
    
    y_pred = np.exp(Y_pred[:,i])
    
    mae_average.append(mean_absolute_error(y_true, y_pred))
    sdep_average.append(np.std(np.abs(y_true - y_pred)))
    mse_average.append(mean_squared_error(y_true, y_pred))
    rmse_average.append(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_average.append(r2_score(y_true, y_pred))
    mape_average.append(mean_absolute_percentage_error(y_true, y_pred)*100)
    
print_report('MAE  :' + str(np.mean(mae_average)) + '  +/- ' + str(np.std(mae_average)))
print_report('SDEP :' + str(np.mean(sdep_average)) + '  +/- ' + str(np.std(sdep_average)))
print_report('MSE  :' + str(np.mean(mse_average)) + '  +/- ' + str(np.std(mse_average)))
print_report('RMSE :' + str(np.mean(rmse_average)) + '  +/- ' + str(np.std(rmse_average)))
print_report('R2   :' + str(np.mean(r2_average)) + '  +/- ' + str(np.std(r2_average)))
print_report('MAPE :' + str(np.mean(mape_average)) + '  +/- ' + str(np.std(mape_average)))

# -- Validation
print_report('\nValidation set')
print_report('-'*30)

mae_average,sdep_average,mse_average,rmse_average,r2_average,mape_average=[],[],[],[],[],[]
for i in range(n_ensembles):
    valid_old    = df_split.loc[df_split['Ensemble_'+str(i+1)] == 'Valid']
    valid_index = valid_old.index.tolist()
    
    y_true = exp_values.values[valid_index]
    y_true = np.exp(y_true) 
    Y_pred = Y_pred_total[valid_index]
    
    y_pred = np.exp(Y_pred[:,i])
    
    mae_average.append(mean_absolute_error(y_true, y_pred))
    sdep_average.append(np.std(np.abs(y_true - y_pred)))
    mse_average.append(mean_squared_error(y_true, y_pred))
    rmse_average.append(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_average.append(r2_score(y_true, y_pred))
    mape_average.append(mean_absolute_percentage_error(y_true, y_pred)*100)
    
print_report('MAE  :' + str(np.mean(mae_average)) + '  +/- ' + str(np.std(mae_average)))
print_report('SDEP :' + str(np.mean(sdep_average)) + '  +/- ' + str(np.std(sdep_average)))
print_report('MSE  :' + str(np.mean(mse_average)) + '  +/- ' + str(np.std(mse_average)))
print_report('RMSE :' + str(np.mean(rmse_average)) + '  +/- ' + str(np.std(rmse_average)))
print_report('R2   :' + str(np.mean(r2_average)) + '  +/- ' + str(np.std(r2_average)))
print_report('MAPE :' + str(np.mean(mape_average)) + '  +/- ' + str(np.std(mape_average)))

# -- Test
print_report('\nTest set')
print_report('-'*30)

y_true = exp_values.values[test_index]
y_true = np.exp(y_true) 
Y_pred = Y_pred_total[test_index]

mae_average,sdep_average,mse_average,rmse_average,r2_average,mape_average=[],[],[],[],[],[]

for i in range(n_ensembles):
    y_pred = np.exp(Y_pred[:,i])
    
    mae_average.append(mean_absolute_error(y_true, y_pred))
    sdep_average.append(np.std(np.abs(y_true - y_pred)))
    mse_average.append(mean_squared_error(y_true, y_pred))
    rmse_average.append(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_average.append(r2_score(y_true, y_pred))
    mape_average.append(mean_absolute_percentage_error(y_true, y_pred)*100)
    
print_report('MAE  :' + str(np.mean(mae_average)) + '  +/- ' + str(np.std(mae_average)))
print_report('SDEP :' + str(np.mean(sdep_average)) + '  +/- ' + str(np.std(sdep_average)))
print_report('MSE  :' + str(np.mean(mse_average)) + '  +/- ' + str(np.std(mse_average)))
print_report('RMSE :' + str(np.mean(rmse_average)) + '  +/- ' + str(np.std(rmse_average)))
print_report('R2   :' + str(np.mean(r2_average)) + '  +/- ' + str(np.std(r2_average)))
print_report('MAPE :' + str(np.mean(mape_average)) + '  +/- ' + str(np.std(mape_average)))

# -- Complete
print_report('\nComplete set')
print_report('-'*30)

y_true = exp_values.values
y_true = np.exp(y_true) 
Y_pred = Y_pred_total

mae_average,sdep_average,mse_average,rmse_average,r2_average,mape_average=[],[],[],[],[],[]

for i in range(n_ensembles):
    y_pred = np.exp(Y_pred[:,i])
    
    mae_average.append(mean_absolute_error(y_true, y_pred))
    sdep_average.append(np.std(np.abs(y_true - y_pred)))
    mse_average.append(mean_squared_error(y_true, y_pred))
    rmse_average.append(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_average.append(r2_score(y_true, y_pred))
    mape_average.append(mean_absolute_percentage_error(y_true, y_pred)*100)
    
print_report('MAE  :' + str(np.mean(mae_average)) + '  +/- ' + str(np.std(mae_average)))
print_report('SDEP :' + str(np.mean(sdep_average)) + '  +/- ' + str(np.std(sdep_average)))
print_report('MSE  :' + str(np.mean(mse_average)) + '  +/- ' + str(np.std(mse_average)))
print_report('RMSE :' + str(np.mean(rmse_average)) + '  +/- ' + str(np.std(rmse_average)))
print_report('R2   :' + str(np.mean(r2_average)) + '  +/- ' + str(np.std(r2_average)))
print_report('MAPE :' + str(np.mean(mape_average)) + '  +/- ' + str(np.std(mape_average)))

# Save predictions and report
df_predictions.to_csv('Predictions_'+model_name+'.csv', index=False) # Save predictions of ensemble model
report.close()

