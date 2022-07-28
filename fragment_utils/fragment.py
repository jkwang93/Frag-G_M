# Decode a string representation into a fragment.
# from mol_utils import drop_salt
from rdkit import Chem


# Read a file containing SMILES
# The file should be a .smi or a .csv where the first column should contain a SMILES string
# def read_file(file_name, drop_first=True):
#     molObjects = []
#
#     with open(file_name) as f:
#         for l in f:
#             if drop_first:
#                 drop_first = False
#                 continue
#
#             l = l.strip().split(",")[0]
#             smi = drop_salt(l.strip())
#             molObjects.append(Chem.MolFromSmiles(smi))
#
#     return molObjects


# Encode a fragment.
from rdkit import Chem
from mol_utils import join_fragments
from mol_utils import split_molecule


def encode_molecule(m):
    fs_list = []
    count = 0
    for index,i in enumerate(m):
        try:
            fs = [Chem.MolToSmiles(f) for f in split_molecule(Chem.MolFromSmiles(i))]
            # encoded = "-".join([encodings[f] for f in fs])
            fs_list.extend(fs)
            print(index)
        except Exception as e:
            print('error_mol_id: ',index)
            count+=1
            print(count)
            # print(index)



    return fs_list

def decode_molecule(enc):
    fs_list = []
    for index, i in enumerate(enc):
        fs = [Chem.MolFromSmiles(x) for x in i]
        fs = join_fragments(fs)
        fs_list.append(Chem.MolToSmiles(fs))
        print(index)


    return fs_list

def decode_molecule_test(enc):
    fs_list = []
    fs = [Chem.MolFromSmiles(x) for x in enc]

    fs = join_fragments(fs)
    fs_list.append(Chem.MolToSmiles(fs))


    return fs_list

import pandas as pd
# fragment_smiles = pd.read_csv('../data/used_smiles.smi',header=None).values.flatten().tolist()
fragment_smiles = pd.read_csv('../data/LTK.smi',header=None).values.flatten().tolist()

# fragment_mols = [Chem.MolFromSmiles(x) for x in fragment_smiles]
enc = encode_molecule(fragment_smiles)

output = (enc)

print(output)

# pd.DataFrame(output).to_csv('fragments.csv',header=False,index=False)

# frg_list = ['C[Yb]',  '[Yb]C1CCN([Lu])CC1', '[Yb]N[Lu]', '[Yb]c1cc([Lu])ccc1[Ta]','[Yb]c1cc2c([Lu])ncnc2[nH]1', '[Yb]O[Lu]', '[Yb]c1ccc2oc([Lu])nc2c1', '[Yb]N[Lu]', '[Yb]c1ccc([Lu])cc1', 'Cl[Yb]', 'C[Yb]']

dec = decode_molecule_test(enc)

print(dec)