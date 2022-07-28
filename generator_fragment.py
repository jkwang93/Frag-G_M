#!/usr/bin/env python

import torch

from rdkit import Chem


from data_structs_fragment import MolData, Vocabulary
from fragment_utils.mol_utils import join_fragments

from model import RNN

batch_size = 24
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

print(device)




def pretrain(restore_from='./data/6w_Prior_fragment.ckpt'):
    # token_list = ['is_JNK3', 'is_GSK3', 'high_QED', 'good_SA']

    smile_list = []
    """Trains the Prior rnn"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="./fragment_utils/fragments.csv")

    Prior = RNN(voc)

    # Can restore from a saved rnn
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_from))

    else:
        Prior.rnn.load_state_dict(torch.load(restore_from, map_location=lambda storage, loc: storage))

    Prior.rnn.to(device)

    Prior.rnn.eval()
    valid = 0

    for step in range(50):
        # Every 500 steps we decrease learning rate and print some information
        # seqs = Prior.generate(batch_size, max_length=140, con_token_list=token_list)
        seqs, likelihood, _ = Prior.sample(batch_size)


        for i, seq in enumerate(seqs.cpu().numpy()):
            print(i)

            mol = voc.decode_frag(seq)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)

                if mol != None:
                    valid += 1
                else:
                    print(smiles)
                # if i < 5:
                #     tqdm.write(smile)
                smile_list.append(smiles)
    print(valid / (batch_size * 50))
    write_in_file('./output/6w_fragment.smi', smile_list)
    #
    # tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
    # tqdm.write("*" * 50 + "\n")
    # torch.save(Prior.rnn.state_dict(), "../data/Prior.ckpt")


def write_in_file(path_to_file, data):
    with open(path_to_file, 'a+') as f:
        for item in data:
            f.write("%s\n" % item)


if __name__ == "__main__":
    pretrain()
    torch.cuda.empty_cache()
