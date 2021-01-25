"""
A script that loads JAK2 ligand and decoys and runs an SVM classifier
"""

from learning import loadjak2
import os

root = os.path.dirname(os.path.dirname(os.getcwd()))

loader = loadjak2(root)
loader.featurize("fragprints")

if __name__ == '__main__':
    print(loader.features)