# _*_ encoding: utf-8 _*_
__author__ = 'wjk'
__date__ = '2021/6/1 23:53'

import pandas as pd
import numpy as np
fragment_smiles = pd.read_csv('../data/canonical_semi_100w.csv').values.flatten()
print(len(fragment_smiles))
index = pd.read_csv('frag.o34473',header=None).values.flatten().tolist()

output = np.delete(fragment_smiles,index,0)
print(len(output))

pd.DataFrame(output).to_csv('canonical_semi_train.csv',header=False,index=False)

