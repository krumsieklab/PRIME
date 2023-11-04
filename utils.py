#these contain some utility functions to help with plotting and preprocessing and other thigns with the data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from tqdm import tqdm

def plot(data, labels = None, plottype = 0, feature_list = None, col_nums = None, datatype = 'met'):
    if plottype == 'samples':
        fig, axes = plt.subplots(2, np.ceil(len(mse)/2), figsize=(10, 8))
        axes = axes.flatten()
        if datatype == 'met':
            type = 'Metabolite'
        else:
            type = 'Protein'
        for i, col in enumerate(col_nums):
            ax = axes[i]
            #data = data_df[col].dropna()
            tempdata = data[:,i]
            ax.hist(tempdata, bins=50, edgecolor='black')
            ax.set_title(f'{type} ID: {col}')
            ax.set_xlabel('Expression Level')
            ax.set_ylabel('Frequency')

        plt.tight_layout()
        return fig, axes
    
    if plottype == 0: #this plot finds the 4 most correlated features and plots them by default, but you can supply own mse list to find other ones
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        mse = feature_list
        for i, feature_idx in enumerate(mse):
            ax = axs.flatten()[i]
            ax.scatter(labels[:, feature_idx], data[:, feature_idx], alpha=0.5)
            ax.plot([min(labels[:, feature_idx]), max(labels[:, feature_idx])],
                    [min(labels[:, feature_idx]), max(labels[:, feature_idx])],
                    color='red', linestyle='--')
            R2 = np.corrcoef(labels[:, feature_idx], data[:, feature_idx])[0, 1] ** 2
            ax.set_title(f'Feature {feature_idx} (MSE: {mse[feature_idx]:.4f}, R2: {R2:.4f})')
            ax.set_xlabel('True Value')
            ax.set_ylabel('Predicted Value')
        return fig, axs
    
    if plottype == 1:
        fig, ax = plt.subplots()
        #now plot stuff
        return fig, ax
    
    if plottype == 2:
        fig, ax = plt.subplots()
        #now plot stuff
        return fig, ax
    
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
class LoadData():
    def __init__(self, filemet = 'UKBB_300K_Overlapping_MET.csv', fileprot = 'UKBB_300K_Overlapping_OLINK.csv', preprocess = True, impute = None, removenan = True, load = False):
        
        #reads in the files and separates the data from the column labels
        met = pd.read_csv(f'/home/sat4017/PRIME/{filemet}', index_col = 0)
        prot = pd.read_csv(f'/home/sat4017/PRIME/{fileprot}', index_col = 0)
        self.p_cols = prot.columns.to_numpy().astype('float64')
        self.m_cols = met.columns.to_numpy() #not making float64 because has the X in it and some weird naming scheme, will not save it out for now we also don't delete metabolites, so use the full one
        met = met.to_numpy()
        prot = prot.to_numpy()
        
        if preprocess and not load: #if we are doing any preprocessing steps like z score
            prot = self.preprocessor(prot, datatype = 'prot', impute = impute, removenan = removenan)
            met = self.preprocessor(met, datatype = 'met', impute = impute, removenan = removenan)
        self.m = met #assigns them
        self.p = prot
        
        if load:
            self.loadfile()
        
        #goes through and creates a dictionary of the protein names and their indices for use later on
        protein_dict = {}

        with open('/home/sat4017/PRIME/protein_coding143.tsv', 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader)
            for row in reader:
                key = int(row[0])  # Converting the key to integer
                value = row[1].split(';')  # Splitting the string by ';' to get a list
                protein_dict[key] = value
                
        self.p_dict = protein_dict
        
        return None
    
    def preprocessor(self, data, datatype = 'met', scale = True, impute = None, removenan = True): #potentially we add dealing with NaNs here
        if datatype == 'met':
            #first we log each dataset
            if removenan: #uses the stored rows and columns we are supposed to keep to make them the same size and removes the associated values from the other dataset
                #if we remove samples from one, we have to remove samples from the other, but only the rows are the issue
                data = data[self.rowkeep,:]
            data = np.log10(data+1e-9)
            if scale:
                #now we scale each dataset
                data = self.zscore(data)
            return data
        
        if datatype == 'prot':
            if removenan:
                #first we remove the nan values
                data = self.filter_fct(data, 200, 500) #this threshold keeps a lot more proteins but removes about 2000 columns, makes sense though
                
            if impute == 'min':
                #now we impute each dataset
                #first find min of each column
                min = np.nanmin(data, axis = 0)
                nan_pos = np.isnan(data)
                for i in range(nan_pos.shape[1]):
                    data[nan_pos[:,i],i] = min[i]
                #data = SimpleImputer(strategy = 'constant', fill_value=min).fit_transform(data)
            elif impute == 'mean':
                data = SimpleImputer(strategy = 'mean').fit_transform(data)
            elif impute == 'KNN':
                data = KNNImputer(n_neighbors=5).fit_transform(data)
                
            if scale:
                #now we scale each dataset
                data = self.zscore(data)
            return data
        
    def zscore(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean)/std
    
    def filter_fct(self, array, row_threshold, col_threshold, print_shape = False):
        #this function does what I propose
        row_nan_count = np.isnan(array).sum(axis=1)
        self.rowkeep = np.where(row_nan_count <= row_threshold)[0]
        filtered_array_by_row = array[row_nan_count <= row_threshold, :]
        
        col_nan_count = np.isnan(filtered_array_by_row).sum(axis=0)
        self.colkeep = np.where(col_nan_count <= col_threshold)[0]
        self.p_cols = self.p_cols[self.colkeep]
        filtered_array_by_row_and_col = filtered_array_by_row[:, col_nan_count <= col_threshold]
        
        if print_shape:
            print("Original array shape:", array.shape)
            print("Array shape after removing both", filtered_array_by_row_and_col.shape)
            print('numer of NaNs in filtered array', np.sum(np.isnan(filtered_array_by_row_and_col)))
            print('\n')
        return filtered_array_by_row_and_col
    
    def savefile(self, name_met, name_prot, name_pcols):
        #saves out the m and p arrays
        np.save(f'/home/sat4017/PRIME/saved_data/{name_met}', self.m)
        np.save(f'/home/sat4017/PRIME/saved_data/{name_prot}', self.p)
        np.save(f'/home/sat4017/PRIME/saved_data/{name_pcols}', self.p_cols)
        return None
        
    def loadfile(self, name_met = 'm_knn.npy', name_prot = 'p_knn.npy', name_pcols = 'p_cols.npy'):
        #loads in the m and p arrays
        self.m = np.load(f'/home/sat4017/PRIME/saved_data/{name_met}')
        self.p = np.load(f'/home/sat4017/PRIME/saved_data/{name_prot}')
        self.p_cols = np.load(f'/home/sat4017/PRIME/saved_data/{name_pcols}')
        return None
    
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import seaborn as sns
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
class CV():
    def __init__(self, dataloader, n = 5):
        self.met = dataloader.m
        self.prot = dataloader.p
        # self.n = n
        self.predict = np.zeros(self.prot.shape)
        self.p_dict = dataloader.p_dict
        self.p_cols = dataloader.p_cols
        self.m_cols = dataloader.m_cols
        self.folds(n = n)
        
    def folds(self, n = 5, random_state = 42):
        #first we shuffle the data, then we return the folds, also use a random state
        #use kfold to split the data, then return it
        #also save out the fold numbeer, so we know how to save it, but basically when we get indices, we can just save it out
        
        self.fold_list = []
        #create n random groups
        kf = KFold(n_splits=n, shuffle=True, random_state=random_state)
        for train_idx, test_idx in kf.split(self.prot):
            self.fold_list.append(test_idx)
        
        #so we get n sets of indices, then we save the indices, then when we do k fold, we know which set of indices predicting
        #save out those ones in the end, so need to return the fold number alongside the indices
        #so loop through the folds, and then index into the fold_list
        
        #no need to return
        
        #return self.fold_list
    
    def save_out(self, predicts, fold):
        #saves out the data chunk by chunk
        self.predict[self.fold_list[fold],:] = predicts #basically it assigns it to the proper one, definitiley check, likely bug!
        
    def train_loop(self, model):
        
        for fold, idx in tqdm(enumerate(self.fold_list), desc="Training Folds", total=len(self.fold_list)):
            test_idx = idx
            #X_train = self.met.drop(test_idx, axis = 0)
            m_train = np.delete(self.met, test_idx, axis = 0)
            m_test = self.met[test_idx,:]
            p_train = np.delete(self.prot, test_idx, axis = 0)
            #p_test = self.prot[test_idx,:]
            model.fit(m_train, p_train)
            #model.fit(self.met[train_idx,:], self.prot[train_idx,:])
            predicts = model.predict(m_test)
            self.save_out(predicts, fold)
        return self.predict
    
    def train_loop_pl(self, model = None, device = 'cuda', batch_size = 128, input_dim = 251, output_dim = 1039, hidden_dims = [512, 512], custom_layers = None):
        #first thing we need to do is to create the dataset and dataloader
        self.device = device
        self.batch_size = batch_size
        
        if model is None:
            model = self.default_model(input_dim, output_dim, hidden_dims, custom_layers)
        torch.set_float32_matmul_precision('medium')
        self.model = model
        
        for fold, idx in tqdm(enumerate(self.fold_list), desc="Training Folds", total=len(self.fold_list)):
            test_idx = idx
            #X_train = self.met.drop(test_idx, axis = 0)
            m_train = np.delete(self.met, test_idx, axis = 0)
            m_test = self.met[test_idx,:]
            p_train = np.delete(self.prot, test_idx, axis = 0)
            p_test = self.prot[test_idx,:]
            train_loader = self.data_loader(m_train, p_train, 'train')
            test_loader = self.data_loader(m_test, p_test, 'test')
            
            if fold == 0:
                trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1)
            else:
                trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1, enable_progress_bar=False, logger=False, progress_bar_refresh_rate=0)
                        
            model = copy.deepcopy(self.model)
                        
            trainer.fit(model, train_loader, test_loader)
            
            pred = trainer.predict(model, test_loader)
            pred = torch.cat(pred, dim=0).cpu().numpy()
            #save out the data
            self.save_out(pred, fold)
        
        return None
        
    def data_loader(self, m, p, type = 'test'):
        X = torch.tensor(m, dtype=torch.float32).to(self.device)
        y = torch.tensor(p, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X, y)
        if type == 'test':
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        if type == 'train':
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader
    
    def default_model(self, input_dim, output_dim, hidden_dims=[512, 512], custom_layers=None):
        model = LinearNet(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, custom_layers=custom_layers)
        #to do custom layers use this
        # custom_layers = nn.Sequential(
        #     nn.Linear(251, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1039)
        # )
        return model
    
    def idx_plot(self, feature_list = None):
        #plots the best ones corresponding to the indices, and by default it will plot the 4 most correlated ones
        fig, axs = plt.subplots(2, int(np.ceil(len(feature_list)/2)), figsize=(10, 10))
        labels = self.prot
        data = self.predict
        mse = feature_list
        if mse is None: #if mse is empty, then we just plot the 4 most correlated ones whcih have lowest mse
            mse = np.argsort(np.mean((labels - data) ** 2, axis=0))[:4]
        for i, feature_idx in enumerate(mse):
            ax = axs.flatten()[i]
            ax.scatter(labels[:, feature_idx], data[:, feature_idx], alpha=0.5)
            ax.plot([min(labels[:, feature_idx]), max(labels[:, feature_idx])],
                    [min(labels[:, feature_idx]), max(labels[:, feature_idx])],
                    color='red', linestyle='--')
            R2 = np.corrcoef(labels[:, feature_idx], data[:, feature_idx])[0, 1] ** 2
            tempmse = np.mean((labels[:, feature_idx]-data[:, feature_idx])**2, axis = 0) #calculate the mse!
            ax.set_title(f'Feature {self.p_dict[int(self.p_cols[feature_idx])][0]} (MSE: {tempmse:.4f}, R2: {R2:.4f})')
            ax.set_xlabel('True Value')
            ax.set_ylabel('Predicted Value')
        return fig, axs
    

    def pred_summary_plot(self, correlations = None):
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(12, 10))
        
        if correlations is None:
            # Calculate Pearson Correlation
            correlations = []
            for i in range(self.predict.shape[1]):
                correlations.append(pearsonr(self.predict[:,i], self.prot[:,i])[0])

        correlations = np.array(correlations)
        
        sorted_indices = np.argsort(np.abs(correlations))[::-1]
        top_10_indices = sorted_indices[:10]
        top_10_proteins = [self.p_dict[int(self.p_cols[i])][0] for i in top_10_indices]
        top_10_correlations = correlations[top_10_indices]
        

        # Create a GridSpec object
        gs = gridspec.GridSpec(2, 2)  # 3 rows and 2 columns

        # Create subplots
        ax1 = fig.add_subplot(gs[0, :])  # First row, spanning all columns
        ax2 = fig.add_subplot(gs[1:, 0])  # Second and third rows, first column
        ax3 = fig.add_subplot(gs[1:, 1])  # Second and third rows, second column

        #fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Top subplot: Horizontal Bar Plot for Top 10 Correlations
        ax1.barh(top_10_proteins[::-1], top_10_correlations[::-1], color='blue')
        ax1.set_xlabel('Pearson Correlation')
        ax1.set_title('Top 10 Pearson Correlations')

        # Add text labels
        for i, v in enumerate(top_10_correlations[::-1]):
            ax1.text(v + 0.008, i, str(round(v, 2)), color='black', verticalalignment='center')

        # Bottom subplot: Violin Plot for Distribution of Correlations
        sns.violinplot(ax=ax2, y=correlations)
        ax2.set_xlabel('Distribution')
        ax2.set_ylabel('Pearson Correlation')
        ax2.set_title('Distribution of Pearson Correlations')
        ylim = ax2.get_ylim()

        #now we do a scatter plot ofo the pearson correlations that is ordered
        sorted_indices = np.argsort(correlations)[::-1]
        sorted_correlations = correlations[sorted_indices]
        #sorted_proteins = [protein_names[i] for i in sorted_indices]
        ax3.scatter(range(len(sorted_correlations)), sorted_correlations, s=5, c='red')
        ax3.set_xlabel('Distribution')
        ax3.set_ylabel('Pearson Correlation')
        ax3.set_title('Distribution of Pearson Correlations')
        ax3.set_ylim(ylim)

        axes = [ax1, ax2, ax3]
        
        return fig, axes
    
    def save_file(self, name):
        #saves out the file
        np.save(f'/home/sat4017/PRIME/saved_data/{name}', self.predict)
        return None
    
    def load_file(self, name):
        #loads in the file
        self.predict = np.load(f'/home/sat4017/PRIME/saved_data/{name}')
        return None
    
    def remove_cols(self, col_labels):
        #this function removes the columns from the data, this is if there are certain proteins not in our ppi network
        #actually this function may indeed be useless
        idx_rm = np.where(np.isin(self.p_cols, col_labels))[0]
        self.p_cols = np.delete(self.p_cols, idx_rm)
        self.prot = np.delete(self.prot, idx_rm, axis = 1)
        self.predict = np.zeros(self.prot.shape) #reinitialize it cuz now different size

class LinearNet(pl.LightningModule):
    def __init__(self, input_dim=251, output_dim=1039, hidden_dims=[512, 512], custom_layers=None):
        super(LinearNet, self).__init__()
        
        # Use custom layers if provided
        if custom_layers:
            self.layers = custom_layers
        else:
            all_layers = []
            last_dim = input_dim
            for hidden_dim in hidden_dims:
                all_layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU()])
                last_dim = hidden_dim
            all_layers.append(nn.Linear(last_dim, output_dim))
            self.layers = nn.Sequential(*all_layers)
        
        self.losses = []

    def forward(self, x):
        x = self.layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.losses.append(loss)
        return {'test_loss': loss}, y_hat

    def on_test_end(self):
        avg_loss = torch.stack([x for x in self.losses]).mean()
        print(f'avg_test_loss: {avg_loss}')
        #self.log('avg_test_loss', avg_loss)
        
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        outputs = self(x)
        return outputs
class PPI():
    #this class serves as the major class for dealing with the ppi network, uses the same dataloader
    def __init__(self, dataloader, infofile = '9606.protein.info.v12.0.txt', ppi_file = '9606.protein.links.v12.0.txt'):
        self.met = dataloader.m
        self.prot = dataloader.p
        self.p_dict = dataloader.p_dict
        self.p_cols = dataloader.p_cols
        self.m_cols = dataloader.m_cols
        self.load_info_and_match(textfile=infofile)
        self.load_ppi(textfile=ppi_file)
        
        
    def load_info_and_match(self, textfile = '9606.protein.info.v12.0.txt'):
        #loads in the protein names and ensemble IDs, and removes from the self.prot what isnt' already in the ppi network
        ppi_infodict = {}

        with open(textfile, 'r') as file:
            # Skip the header
            next(file)
            
            # Iterate over lines
            for line in file:
                parts = line.split('\t')
                
                # If there are at least two parts
                if len(parts) >= 2:
                    key = parts[1]
                    value = parts[0]
                    ppi_infodict[key] = value
        
        col_remove = []
        for p_idx in self.p_dict.keys():
            if self.p_dict[p_idx][0] not in ppi_infodict.keys() and p_idx in self.p_cols:
                col_remove.append(p_idx)
                #print(self.p_dict[p_idx][0])
                
        self.remove_cols(col_remove)
        
        self.ppi_p2e = ppi_infodict
        
        self.ppi_e2p = {}
        
        for key in ppi_infodict.keys():
            self.ppi_e2p[ppi_infodict[key]] = key
        
    def load_ppi(self, textfile = '9606.protein.links.v12.0.txt'):
        #need to first find the size of the matrix which is going to be self.prot.shape[1] by self.prot.shape[1]
        #then we will need to fill in the matrix with the values, by finding the associated protein name and then finding the column number, which is the associated key in self.prot_dict
        #make a prot dict that is inverse of that, so we don't search each
        self.ppi = np.zeros((self.prot.shape[1], self.prot.shape[1]))
        
        # #now develop a dictionary that is the inverse of the protein name
        self.p_inv_dict = {}
        for val in self.p_dict:
            self.p_inv_dict[self.p_dict[val][0]] = val
            
        with open(textfile, 'r') as file:
            # Skip the header
            next(file)
            
            # Iterate over lines
            for line in file:
                parts = line.split(' ')
                
                # If there are at least two parts
                if len(parts) >= 2:
                    key1 = self.ppi_e2p[parts[0]]
                    key2 = self.ppi_e2p[parts[1]]
                    value = parts[2][:-1] #defines keys and values
                    if key1 in self.p_inv_dict and key2 in self.p_inv_dict: #first makes sure it exists in there
                        t1 = self.p_inv_dict[key1]
                        t2 = self.p_inv_dict[key2]
                        if t1 in self.p_cols and t2 in self.p_cols:    
                            idx1 = np.where(self.p_cols == int(t1))[0][0] #finds the index of protein 1
                            idx2 = np.where(self.p_cols == int(t2))[0][0] #and for protein 2
                            self.ppi[idx1, idx2] = value #finally assigns it to 1,2
                            # self.ppi[idx2, idx1] = value #here assigns it to 2,1 assumes symmetrical, we not assuming that
        
    def remove_cols(self,col_labels):
        idx_rm = np.where(np.isin(self.p_cols, col_labels))[0]
        self.p_cols = np.delete(self.p_cols, idx_rm)
        self.prot = np.delete(self.prot, idx_rm, axis = 1)