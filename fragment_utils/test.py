# Decode a string representation into a fragment.

# Encode a fragment.
from rdkit import Chem

from file_reader import read_file
from mol_utils import join_fragments
from mol_utils import split_molecule


def encode_molecule(m):
    fs_list = []
    for i in m:
        fs = [Chem.MolToSmiles(f) for f in split_molecule(i)]
    # encoded = "-".join([encodings[f] for f in fs])
        fs_list.append(fs)
    return fs_list

def decode_molecule(enc):
    fs_list = []
    for i in enc:
        fs = [Chem.MolFromSmiles(x) for x in i]
        fs = join_fragments(fs)
        fs_list.append(Chem.MolToSmiles(fs))


    return fs_list


fragment_mols = read_file('../data/canonical_semi_100w.csv')[:10]
for i in fragment_mols:
    print(Chem.MolToSmiles(i))


enc = encode_molecule(fragment_mols)

dec = decode_molecule(enc)

print(dec)