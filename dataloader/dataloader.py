"""
Abstract class implementing the data loading, data splitting,
type validation and feature extraction functionalities.
"""

from abc import ABCMeta, abstractmethod

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


class DataLoader(metaclass=ABCMeta):
    """
    Abstract class implementing the data loading, data splitting,
    type validation and feature extraction functionalities.
    """

    def __init__(self):
        self.task = None

    @property
    @abstractmethod
    def features(self):
        """
        Abstract property for storing features.
        """
        raise NotImplementedError

    @features.setter
    @abstractmethod
    def features(self, value):
        """
        Abstract setter for setting features.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self):
        """
        Abstract property for storing labels
        """
        raise NotImplementedError

    @labels.setter
    @abstractmethod
    def labels(self, value):
        """
        Abstract setter for setting labels
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, drop=True):
        """Checks whether the loaded data is a valid instance of the specified
        data type, potentially dropping invalid entries.

        Args:
            drop (bool):  whether to drop invalid entries

        """
        raise NotImplementedError

    @abstractmethod
    def featurize(self, representation):
        """Transforms the features to the specified representation (in-place).

        Args:
            representation: desired feature format

        """
        raise NotImplementedError

    def _scale(self, train, test=None):
        """
        Auxiliary function to perform scaling on features and labels.
        Fits the standardisation scaler on the training set and
        transforms training and (optionally) testing set
        Args:
            train: proportion of features/labels used for training
            test: proportion of features/labels used for testing, optional

        Returns: a tuple of scaled training set, scaled testing set and the fitted scaler
        """

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)

        return train_scaled, test_scaled, scaler

    def split_and_scale(self, kfold_shuffle=True, num_splits=5, test_size=0.2, scale_labels=True, scale_features=False):
        """Splits the data into training and test sets.

        Args:
            kfold_shuffle: whether to split the data into k folds (True) or k re-shuffled splits (False)
            num_splits: number of folds or re-shuffled splits
            test_size: size of the test set for re-shuffled splits
            scale_labels: whether to standardize the labels (after splitting)
            scale_features: whether to standardize the features (after splitting)

        Returns:
            (potentially standardized) training and testing sets with associated scalers

        """

        # reshape labels
        self.labels = self.labels.reshape(-1, 1)

        # define splitter
        if kfold_shuffle:
            splitter = StratifiedKFold(
                n_splits=num_splits, shuffle=True
            )

        else:
            splitter = StratifiedShuffleSplit(
                n_splits=num_splits, test_size=test_size
            )

        splits = []

        for train_index, test_index in splitter.split(self.features, self.labels):
            X_train = self.features[train_index]
            X_test = self.features[test_index]
            y_train = self.labels[train_index]
            y_test = self.labels[test_index]

            # scale features, if requested
            if scale_features:
                features_out = self._scale(X_train, X_test)
            else:
                features_out = X_train, X_test, None

            # scale labels, if requested
            if scale_labels:
                labels_out = self._scale(y_train, y_test)
            else:
                labels_out = y_train, y_test, None

            splits.append([features_out + labels_out])

        # return list of concatenated tuples for each split
        return splits
