"""
A script containing utility functions.
"""

from dataloader import DataLoaderMP
import pandas as pd
import os
import re
import numpy as np


def load(path_to_root, target):
    """
    Loads SMILES and affinity data from the specified csv files.
    Args:
        path_to_root: path to the repository root
        target: choose between ['all', 'kinase', list if ID's],
        specifies whether to load the ligands for all targets,
        only the kinases or only the specified targets

    Returns: either a dataloader object or a dataframe

    """

    def add_to_df(dataframe, filename, targetname):
        """Auxiliary function to concatenate read-in data to the main DataFrame
        Args:
            dataframe: the dataframe to which to add the read-in data
            filename: name of the file being read in
            targetname: the name of the target
        Returns:
            The augmented DataFrame
        """
        tempDataFrame = pd.read_csv(os.path.join(datadir, 'chembl_data', filename))
        tempDataFrame['target'] = [targetname] * len(tempDataFrame.index)
        return pd.concat([dataframe, tempDataFrame], ignore_index=True)

    # navigate to directory containing the extracted ligands
    datadir = os.path.join(path_to_root, 'data')

    # load the names of the DUD-E kinases, if specified
    if target == 'kinase':
        with open(os.path.join(datadir, 'kinase_targets.txt')) as file:
            kinase_targets = [kinase.strip('\n') for kinase in file.readlines()]

    # initialise empty DataFrame for results
    results = pd.DataFrame()

    # iterate through each csv in the target directory
    for csv in os.listdir(os.path.join(datadir, 'chembl_data')):

        # extract target names
        pattern = re.compile(r'(.*)_extracted_chembl_data.csv')
        name = re.findall(pattern, csv)[0]

        # append read-in data to results data frame,
        # depending on which option was specified
        if target == "all":
            results = add_to_df(results, csv, name)
        elif target == "kinase" and name in kinase_targets:
            results = add_to_df(results, csv, name)
        elif isinstance(target, list) and name in target:
            results = add_to_df(results, csv, name)

    if results.empty:
        print("Could not retrieve results. Please check that you specified the target"
              "as 'all', 'kinase' or a list of target names used in DUD-E.")

    # drop entries for which there is no pChEMBL entry
    results = results[['canonical_smiles', 'pchembl_value']].dropna()

    #return results

    loader = DataLoaderMP()
    loader.features = results['canonical_smiles'].to_list()
    loader.labels = results['pchembl_value'].to_numpy()

    #loader.validate()

    return loader

if __name__ == '__main__':
    res = load(os.path.dirname(os.getcwd()), 'kinase')
    print("")
