"""
Instantiation of the abstract data loader class for
molecular property prediction datasets.
"""

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles

from dataloader import DataLoader


class DataLoaderMP(DataLoader):
    """
    Instantiation of the abstract data loader class for
    molecular property prediction datasets.
    """

    def __init__(self):
        super(DataLoaderMP, self).__init__()
        self.task = "molecular_property_prediction"
        self._features = None
        self._labels = None

    @property
    def features(self):
        """
        Property for storing features.
        Returns: currently loaded features

        """
        return self._features

    @features.setter
    def features(self, value):
        """
        Setter to initialise or change features.
        Args:
            value: feature data

        """
        self._features = value

    @property
    def labels(self):
        """
        Property for storing labels
        Returns: currently loaded labels

        """
        return self._labels

    @labels.setter
    def labels(self, value):
        """
        Setter to initialise or change labels.
        Args:
            value: label data

        """
        self._labels = value

    def validate(self, drop=True):
        """Checks if the features are valid SMILES strings and (potentially)
        drops the entries that are not.

        Args:
            drop: whether to drop invalid entries

        """

        invalid_idx = []

        # iterate through the features
        for i in range(len(self.features)):

            # try to convert each SMILES to an rdkit molecule
            mol = MolFromSmiles(self.features[i])

            # if it does not work, save the index and print its position to the console
            if mol is None:
                invalid_idx.append(i)
                print(f"Invalid SMILES at position {i+1}: {self.features[i]}")

        if drop:
            self.features = np.delete(self.features, invalid_idx).tolist()
            self.labels = np.delete(self.labels, invalid_idx)

    def featurize(self, representation, bond_radius=3, nBits=2048):
        """Transforms SMILES into the specified molecular representation.

        Args:
            representation: the desired molecular representation, one of [fingerprints, fragments, fragprints]
            bond_radius: int giving the bond radius for Morgan fingerprints. Default is 3
            nBits: int giving the bit vector length for Morgan fingerprints. Default is 2048

        """

        def fingerprints():
            """
            Auxiliary function to transform the loaded features to a fingerprint representation

            Returns: numpy array of features in fingerprint representation

            """

            rdkit_mols = [MolFromSmiles(smiles) for smiles in self.features]
            fps = [
                AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=nBits)
                for mol in rdkit_mols
            ]

            return np.asarray(fps)

        def fragments():
            """
            Auxiliary function to transform the loaded features to a fragment representation

            Returns: numpy array of features in fragment representation

            """

            # descList[115:] contains fragment-based features only
            # (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)
            fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
            frags = np.zeros((len(self.features), len(fragments)))
            for i in range(len(self.features)):
                mol = MolFromSmiles(self.features[i])
                try:
                    features = [fragments[d](mol) for d in fragments]
                except:
                    raise Exception("molecule {}".format(i) + " is not canonicalised")
                frags[i, :] = features

            return frags

        valid_representations = ["fingerprints", "fragments", "fragprints"]

        if representation == "fingerprints":

            self.features = fingerprints()

        elif representation == "fragments":

            self.features = fragments()

        elif representation == "fragprints":

            self.features = np.concatenate((fingerprints(), fragments()), axis=1)

        else:

            raise Exception(
                f"The specified representation choice {representation} is not a valid option."
                f"Choose between {valid_representations}."
            )
