"""
A script containing utility functions.
"""

from dataloader import DataLoaderMP
import pandas as pd
import os
import numpy as np


def loadjak2(path_to_root):
    """
    A function to load the ligands and decoys for JAK2

    Args:
        path_to_root: path to the root of the repository

    Returns: a dataloader object with the respective features
    and labels

    """

    # get DUD-E JAK2 file paths
    active_name = "jak2_actives_final.ism"
    decoy_name = "jak2_decoys_final.ism"

    datadir = os.path.join(path_to_root, "data")

    active_path = os.path.join(datadir, active_name)
    decoy_path = os.path.join(datadir, decoy_name)

    # read the SMILES into a dataframe
    actives = pd.read_csv(active_path, sep=' ', header=None).iloc[:, 0].tolist()
    decoys = pd.read_csv(decoy_path, sep=' ', header=None).iloc[:, 0].tolist()

    # create labels where actives are class 1 and decoys class 0
    labels = np.concatenate((np.ones(len(actives)), np.zeros(len(decoys))))

    # initialise the data loader
    loader = DataLoaderMP()
    loader.features = actives + decoys
    loader.labels = labels

    # check that all smiles are valid
    #loader.validate()

    return loader
