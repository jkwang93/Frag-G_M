#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit.Chem.QED as QED
import scripts.sascorer as sascorer
import pickle
import pandas as pd

rdBase.DisableLog('rdApp.error')


class gsk3_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = '/apdcephfs/private_jikewang/W4_reduce_RL/data/gsk3/gsk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = gsk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)
    

class jnk3_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = '/apdcephfs/private_jikewang/W4_reduce_RL/data/jnk3/jnk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = jnk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)


class drd2_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    # clf_path = '/apdcephfs/private_jikewang/W4_reduce_RL/data/drd2/drd2.pkl'
    clf_path = '/apdcephfs/private_jikewang/W4_reduce_RL/data/drd2/drd2.pkl'


    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = drd2_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)

class qed_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(0)
            else:
                try:
                    qed = QED.qed(mol)
                except:
                    qed = 0
                scores.append(qed)
        return np.float32(scores)


class sa_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(100)
            else:
                scores.append(sascorer.calculateScore(mol))
        return np.float32(scores)



def get_scoring_function(prop_name):
    """Function that initializes and returns a scoring function by name"""
    if prop_name == 'jnk3':
        return jnk3_model()
    elif prop_name == 'gsk3':
        return gsk3_model()
    elif prop_name == 'qed':
        return qed_func()
    elif prop_name == 'sa':
        return sa_func()
    elif prop_name == 'drd2':
        return drd2_model()


def multi_scoring_functions_one_hot_drd(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs])

    props = pd.DataFrame(props).T
    props.columns = function_list

    scoring_sum = condition_convert(props).values.sum(1)

    # scoring_sum = props.sum(axis=0)

    return scoring_sum

def multi_scoring_functions_one_hot_dual(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs])

    props = pd.DataFrame(props).T
    props.columns = function_list

    scoring_sum = condition_convert_dual(props).values.sum(1)

    # scoring_sum = props.sum(axis=0)

    return scoring_sum

def multi_scoring_functions_one_hot_jnk_gsk(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs])

    props = pd.DataFrame(props).T
    props.columns = function_list

    scoring_sum = condition_convert_jnk_gsk(props).values.sum(1)

    # scoring_sum = props.sum(axis=0)

    return scoring_sum

def multi_scoring_functions_one_hot_jnk_qed_sa(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs])

    props = pd.DataFrame(props).T
    props.columns = function_list

    scoring_sum = condition_convert_jnk_qed_sa(props).values.sum(1)

    # scoring_sum = props.sum(axis=0)

    return scoring_sum

def multi_scoring_functions_one_hot_gsk_qed_sa(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs])

    props = pd.DataFrame(props).T
    props.columns = function_list

    scoring_sum = condition_convert_gsk_qed_sa(props).values.sum(1)

    # scoring_sum = props.sum(axis=0)

    return scoring_sum

def condition_convert(con_df):
    # convert to 0, 1
    con_df['drd2'][con_df['drd2'] >= 0.5] = 1
    con_df['drd2'][con_df['drd2'] < 0.5] = 0
    con_df['qed'][con_df['qed'] >= 0.6] = 1
    con_df['qed'][con_df['qed'] < 0.6] = 0
    con_df['sa'][con_df['sa'] <= 4.0] = 1
    con_df['sa'][con_df['sa'] > 4.0] = 0
    return con_df

def condition_convert_dual(con_df):
    # convert to 0, 1
    con_df['jnk3'][con_df['jnk3'] >= 0.5] = 1
    con_df['jnk3'][con_df['jnk3'] < 0.5] = 0
    con_df['gsk3'][con_df['gsk3'] >= 0.5] = 1
    con_df['gsk3'][con_df['gsk3'] < 0.5] = 0
    con_df['qed'][con_df['qed'] >= 0.6] = 1
    con_df['qed'][con_df['qed'] < 0.6] = 0
    con_df['sa'][con_df['sa'] <= 4.0] = 1
    con_df['sa'][con_df['sa'] > 4.0] = 0
    return con_df

def condition_convert_jnk_gsk(con_df):
    # convert to 0, 1
    con_df['jnk3'][con_df['jnk3'] >= 0.5] = 1
    con_df['jnk3'][con_df['jnk3'] < 0.5] = 0
    con_df['gsk3'][con_df['gsk3'] >= 0.5] = 1
    con_df['gsk3'][con_df['gsk3'] < 0.5] = 0
    #con_df['qed'][con_df['qed'] >= 0.6] = 1
    #con_df['qed'][con_df['qed'] < 0.6] = 0
    #con_df['sa'][con_df['sa'] <= 4.0] = 1
    #con_df['sa'][con_df['sa'] > 4.0] = 0
    return con_df

def condition_convert_jnk_qed_sa(con_df):
    # convert to 0, 1
    con_df['jnk3'][con_df['jnk3'] >= 0.5] = 1
    con_df['jnk3'][con_df['jnk3'] < 0.5] = 0
    #con_df['gsk3'][con_df['gsk3'] >= 0.5] = 1
    #con_df['gsk3'][con_df['gsk3'] < 0.5] = 0
    con_df['qed'][con_df['qed'] >= 0.6] = 1
    con_df['qed'][con_df['qed'] < 0.6] = 0
    con_df['sa'][con_df['sa'] <= 4.0] = 1
    con_df['sa'][con_df['sa'] > 4.0] = 0
    return con_df

def condition_convert_gsk_qed_sa(con_df):
    # convert to 0, 1
    #con_df['jnk3'][con_df['jnk3'] >= 0.5] = 1
    #con_df['jnk3'][con_df['jnk3'] < 0.5] = 0
    con_df['gsk3'][con_df['gsk3'] >= 0.5] = 1
    con_df['gsk3'][con_df['gsk3'] < 0.5] = 0
    con_df['qed'][con_df['qed'] >= 0.6] = 1
    con_df['qed'][con_df['qed'] < 0.6] = 0
    con_df['sa'][con_df['sa'] <= 4.0] = 1
    con_df['sa'][con_df['sa'] > 4.0] = 0
    return con_df

if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--prop', required=True)

    args = parser.parse_args()
    funcs = [get_scoring_function(prop) for prop in args.prop.split(',')]

    data = [line.split()[:2] for line in sys.stdin]
    all_x, all_y = zip(*data)
    props = [func(all_y) for func in funcs]

    col_list = [all_x, all_y] + props
    for tup in zip(*col_list):
        print(*tup)
