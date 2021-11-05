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

methods = ['Hildebrand', 'HSP', 'COSMO_RS', 'UNIFAC', 'mod_UNIFAC_Ly', 
           'mod_UNIFAC_Do', 'Abraham', 'MOSCED']

path = os.getcwd()
for jj, method_name in enumerate(methods):

    path_method = path + '/0' + str(jj+1) + '_' + method_name
    path_SPEC = path_method + '/Hybrid'
    
    ######################
    # --- Prediction --- #
    ######################
    # -- Prepare data
    df              = pd.read_csv('Data/database_IAC_ln_clean.csv')
    df_predictions  = pd.read_csv('Data/database_IAC_ln_clean.csv')
    df_split        = pd.read_csv(path_SPEC+'/Split_GNN_IAC.csv')
    
    # Get only feasible molecules for the method
    df              = df[df[method_name].notna()]
    df_predictions  = df_predictions[df_predictions[method_name].notna()]
    df_split        = df_split[df_split[method_name] != 0]
    y_method        = df[method_name]
    
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
        path_model_info = path_SPEC + '/Ensemble_' + str(e)
        
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
    report = open(path_SPEC+'/Report_ensemble_prediction_' + model_name + '.txt', 'w')
    def print_report(string, file=report):
        print(string)
        file.write('\n' + string)
    
    print_report(' Report for ' + model_name)
    print_report('-'*50)
            
    #####################################
    # --- Statistics of predictions --- #
    #####################################
    exp_values   = df_predictions[target]
    
    trainvalid_old    = df_split.loc[(df_split['Ensemble_1'] == 'Train') | (df_split['Ensemble_1'] == 'Valid' )]
    test_old          = df_split.loc[df_split['Ensemble_1'] == 'Test']
    
    trainvalid_index = trainvalid_old.index.tolist()
    test_index       = test_old.index.tolist()
    
    # --- Ensemble model performance
    print_report('\nEnsemble GNN statistics')
    print_report('-'*40)
    print_report('Train/Valid points: ' + str(len(trainvalid_index)))
    print_report('Test  points      : ' + str(len(test_index)))
    
    Y_pred_total      = df_predictions.loc[:, 'Ensemble_1':'Ensemble_'+str(n_ensembles)].to_numpy()
    y_pred_total_mean = np.mean(Y_pred_total, axis=1)
    y_pred_total_std  = np.std(Y_pred_total, axis=1)
    
    df_predictions['ENSEMBLE_mean'] = y_pred_total_mean
    df_predictions['ENSEMBLE_std'] = y_pred_total_std
    ensemble_model = df_predictions['ENSEMBLE_mean']
    ensemble_model_std = df_predictions['ENSEMBLE_std']
    
    # -- Train/Validation
    print_report('\nTraining/Validation set')
    print_report('-'*30)
    
    y_true    = exp_values[trainvalid_index].values
    y_pred    = ensemble_model[trainvalid_index].values + y_method[trainvalid_index].values
    
    # Re-scale
    y_true = np.exp(y_true) 
    y_pred = np.exp(y_pred)
    
    mae_ensemble  = mean_absolute_error(y_true, y_pred)
    sdep_ensemble = np.std(np.abs(y_true - y_pred))
    mse_ensemble  = mean_squared_error(y_true, y_pred)
    rmse_ensemble = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_ensemble   = r2_score(y_true, y_pred)
    mape_ensemble  = mean_absolute_percentage_error(y_true, y_pred)*100
    
    print_report('MAE  :' + str(mae_ensemble))
    print_report('SDEP :' + str(sdep_ensemble))
    print_report('MSE  :' + str(mse_ensemble))
    print_report('RMSE :' + str(rmse_ensemble))
    print_report('R2   :' + str(r2_ensemble))
    print_report('MAPE :' + str(mape_ensemble))
    
    # -- Test
    print_report('\nTest set')
    print_report('-'*30)
    
    y_true     = exp_values[test_index].values
    y_pred     = ensemble_model[test_index].values + y_method[test_index].values
    
    # Re-scale
    y_true = np.exp(y_true) 
    y_pred = np.exp(y_pred)
    
    mae_ensemble  = mean_absolute_error(y_true, y_pred)
    sdep_ensemble = np.std(np.abs(y_true - y_pred))
    mse_ensemble  = mean_squared_error(y_true, y_pred)
    rmse_ensemble = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_ensemble   = r2_score(y_true, y_pred)
    mape_ensemble  = mean_absolute_percentage_error(y_true, y_pred)*100
    
    print_report('MAE  :' + str(mae_ensemble))
    print_report('SDEP :' + str(sdep_ensemble))
    print_report('MSE  :' + str(mse_ensemble))
    print_report('RMSE :' + str(rmse_ensemble))
    print_report('R2   :' + str(r2_ensemble))
    print_report('MAPE :' + str(mape_ensemble))
    
    # -- Complete
    print_report('\nComplete set')
    print_report('-'*30)
    
    y_true     = exp_values.values
    y_pred     = ensemble_model.values + y_method.values
    
    # Re-scale
    y_true = np.exp(y_true) 
    y_pred = np.exp(y_pred)
    
    mae_ensemble  = mean_absolute_error(y_true, y_pred)
    sdep_ensemble = np.std(np.abs(y_true - y_pred))
    mse_ensemble  = mean_squared_error(y_true, y_pred)
    rmse_ensemble = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_ensemble   = r2_score(y_true, y_pred)
    mape_ensemble  = mean_absolute_percentage_error(y_true, y_pred)*100
    
    print_report('MAE  :' + str(mae_ensemble))
    print_report('SDEP :' + str(sdep_ensemble))
    print_report('MSE  :' + str(mse_ensemble))
    print_report('RMSE :' + str(rmse_ensemble))
    print_report('R2   :' + str(r2_ensemble))
    print_report('MAPE :' + str(mape_ensemble))
    
    # Save predictions and report
    df_predictions.to_csv(path_SPEC+'/Predictions_'+model_name+'.csv', index=False) # Save predictions of ensemble model
    report.close()
