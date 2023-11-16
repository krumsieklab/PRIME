import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from tqdm import tqdm
import pingouin as pg

from utils import LoadData, PPI
from sklearn.metrics import mean_squared_error

dataloader = LoadData(impute='KNN',load = True)
ppi = PPI(dataloader)
#now we will have a different model, but the key idea is that we somehow need a normalizing flow term
import pyro
import pyro.distributions as dist
import torch

headers = []
for p_idx in ppi.p_cols:
    headers.append(ppi.p_dict[p_idx][0])
# print(headers)

#now turn to dataframe
df = pd.DataFrame(ppi.prot, columns = headers)

updated_ppi = np.zeros((ppi.mask.shape))
for i in tqdm(range(ppi.mask.shape[0])):
    for j in range(i + 1, ppi.mask.shape[0]):
        if ppi.mask[i,j] == 1:
            #now we find the correlation and condition it on all the other values in the row that are 1
            #first find the indices of the other 1s in the row
            idx = np.where(ppi.mask[i,:] == 1)[0]
            idx = idx[idx != j]  # Exclude the current column protein
            
            #now we will find th eproteins to condition it on
            # cond = ppi.prot[:,idx]
            x = ppi.p_dict[ppi.p_cols[i]][0]
            y = ppi.p_dict[ppi.p_cols[j]][0]
            cond_list = []
            for c in idx:
                cond_list.append(ppi.p_dict[ppi.p_cols[c]][0])
            # print(x)
            # print(y)
            # print(cond_list)
            #now we will find the partial correlation
            pcorr = pg.partial_corr(df, x, y, cond_list, method = 'spearman')
            updated_ppi[i, j] = updated_ppi[j, i] = pcorr['r'].values[0]
            
#save out the updated_ppi matrix
np.save('updated_ppi.npy', updated_ppi)