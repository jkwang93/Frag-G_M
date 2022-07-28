# _*_ encoding: utf-8 _*_
from data_structs_fragment import encode_molecule

__author__ = 'wjk'
__date__ = '2021/6/2 0:35'

import pandas as pd
data = pd.read_csv('used_smiles.smi',header=None).values.flatten()
count = 0
train_data = []
for i,smiles in enumerate(data):

    try:
        encode_molecule(smiles)
        train_data.append(smiles)
        print(i)

    except Exception as e:
        count+=1
        # print(e)
        print('error: ',count)

pd.DataFrame(train_data).to_csv('used_smiles_clean.csv',header=False,index=False)

