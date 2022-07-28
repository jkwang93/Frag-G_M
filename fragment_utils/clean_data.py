# _*_ encoding: utf-8 _*_
from data_structs_fragment import encode_molecule

__author__ = 'wjk'
__date__ = '2021/6/2 0:35'

import pandas as pd
data = pd.read_csv('../data/canonical_semi_train.csv',header=None).values.flatten()
count = 0
train_data = []
for i,smiles in enumerate(data):

    try:
        y = encode_molecule(smiles)
        train_data.extend(y)
        print(i)

    except Exception as e:
        count+=1
        # print(e)
        print('error: ',count)

train_data = list(set(train_data))
pd.DataFrame(train_data).to_csv('fragment.csv',header=False,index=False)

