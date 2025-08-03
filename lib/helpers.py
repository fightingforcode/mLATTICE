
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


#
# This file contains a variety of help functions used in the train_classification.py file, and in the
# Jupyter notebooks.
#

# Standard Python libraries

import time
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def get_optimal_threshold(y_true, probs, thresholds, labels, image_path, N=3):
    """
    Calculates the best threshold per class by calculating the median of the optimal thresholds
    over stratified subsets of the training set.

    Input:  N          - number of stratifications
            strat_size - size of the stratification, between 0.05 and 0.5
            y_true     - binary matrix with the ground-truth values (nr_images, nr_labels)
            probs      - matrix containing the prediction probabilities (nr_images, nr_labels)
            thresholds - possible thresholds (e.g. [0.05, 0.10, 0.15, ..., 0.95])
            labels     - labes [Art, Architecture, ...]
    Output: best_thresholds (nr_labels, )

    Inspiration: "GHOST: Adjusting the Decision Threshold to Handle Imbalanced Data in Machine Learning" by Esposito et al.
    """

    def to_label(probs, threshold):
        return (probs >= threshold) * 1

    nr_images = y_true.shape[0]

    best_thresholds = np.zeros((len(labels), N))

    fig, axs = plt.subplots(8, 4, figsize=(12, 12))
    fig.tight_layout(h_pad=3.0, w_pad=3.0)

    for i in range(N):
        subset_indices = np.random.choice(a=np.arange(nr_images), size=int(np.round(nr_images * 0.2)), replace=False)

        for label_idx, ax in zip(range(len(labels)), axs.flatten()):
            f1_scores = [
                f1_score(y_true=y_true[subset_indices, label_idx], y_pred=to_label(probs[subset_indices, label_idx], t))
                for t in thresholds]
            best_thresholds[label_idx, i] = thresholds[np.argmax(f1_scores)]
            # ax.axvline(x=best_thresholds[label_idx, i], color='k', linestyle='--')
            ax.plot(thresholds, f1_scores)
            ax.set_title(labels[label_idx])
            ax.set_xlabel('Threshold')
            ax.set_ylabel('F1-score')

    optim_thresholds = np.median(best_thresholds, axis=1)

    for label_idx, ax in zip(range(len(optim_thresholds)), axs.flatten()):
        ax.axvline(x=optim_thresholds[label_idx], color='k', linestyle='--')

    save_img(image_path + '/optimal_threshold.png')

    fig, axs = plt.subplots(8, 4, figsize=(12, 12))
    fig.tight_layout(h_pad=3.0, w_pad=3.0)
    bins = np.linspace(0, 1, 50)

    for label_idx, ax in zip(range(len(labels)), axs.flatten()):
        ax.hist(probs[y_true[:, label_idx] == 0][:, label_idx], bins, alpha=0.5, label='false', log=True)
        ax.hist(probs[y_true[:, label_idx] == 1][:, label_idx], bins, alpha=0.5, label='true', log=True)
        ax.axvline(x=optim_thresholds[label_idx], color='k', linestyle='--')
        ax.legend(loc='upper right')
        ax.set_title(labels[label_idx])
        ax.set_xlabel('Probability')
        ax.set_ylabel('Count')

    save_img(image_path + '/probs.png')

    return optim_thresholds



def compute_class_weights(y_true):
    """
    Computes class_weights to compensate imbalanced classes. Inspired in
    'https://towardsdatascience.com/dealing-with-imbalanced-data-in-tensorflow-class-weights-60f876911f99'.
    Dictionary mapping class indices (integers) to a weight (float) value,
    used for weighting the loss function (during training only).
    """
    class_count = y_true.sum(axis=0)
    n_samples = y_true.shape[0]
    n_classes = y_true.shape[1]

    # Compute class weights using balanced method
    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights))
    return dict(zip(class_labels, class_weights))


# ====================================== Metrics ============================================
# The following classes and functions are used to calculate metrics for multilabel datasets.

class MultilabelDatasetCharacteristicsMetrics:
    def __init__(self, labels):
        """
        Initialize with labels, where `labels` is a list of lists,
        each list containing binary indicators (0 or 1) for the presence of each label.
        """
        self.labels = np.array(labels)  # Convert labels to a NumPy array for easier handling
        self.n = len(labels)  # Total number of instances

        # Proper check for an empty array
        if self.labels.size == 0:
            print("Warning: 'labels' is empty or not properly formatted. Please check the input data.")
        self.k = self.labels.shape[1] if self.labels.ndim > 1 and self.labels.size > 0 else 0  # Number of labels
        self.LSet = len(set(map(tuple, self.labels))) if self.labels.size > 0 else 0  # Number of distinct label sets

    # Additional methods remain unchanged

    def label_cardinality(self):
        """Calculate the average number of active labels per instance (Card)."""
        total_active_labels = self.labels.sum(axis=1).mean()
        return total_active_labels

    def label_density(self):
        """Calculate the normalized label cardinality (Dens)."""
        card = self.label_cardinality()
        dens = card / self.k if self.k else 0
        return dens

    def percentage_single_labeled_instances(self):
        """Calculate the percentage of instances with exactly one label (Pmin)."""
        single_label_count = (self.labels.sum(axis=1) == 1).sum()
        pmin = (single_label_count / self.n) * 100 if self.n else 0
        return pmin

    def label_diversity(self):
        """Calculate the diversity of label sets (Div)."""
        div = self.LSet / self.n if self.n else 0
        return div

    def calculate_scumble(self):
        """
        Calculate the SCUMBLE value for a multi-label dataset.

        Returns:
        float: The SCUMBLE value.
        """
        label_frequencies = self.labels.sum(axis=0)
        frequency_median = np.median(label_frequencies)
        minority_indices = np.where(label_frequencies < frequency_median)[0]
        majority_indices = np.where(label_frequencies >= frequency_median)[0]
        co_occurrence_matrix = np.dot(self.labels.T, self.labels)
        minority_majority_co_occurrence = co_occurrence_matrix[np.ix_(minority_indices, majority_indices)]
        minority_occurrences = label_frequencies[minority_indices][:, np.newaxis]
        normalized_concurrence = minority_majority_co_occurrence / minority_occurrences
        scumble_value = np.sum(normalized_concurrence) / (len(minority_indices) * len(majority_indices))
        return scumble_value


class MultilabelImbalanceAnalysis:
    """
    A class to analyze label imbalance in multilabel datasets, providing separate properties
    for each metric: IRLbl, MeanIR, and CVIR.
    """

    def __init__(self, labels):
        """
        Initialize with labels, where `labels` is a binary matrix (n_samples, n_labels)
        indicating the presence or absence of each label.

        Parameters:
        - labels (np.ndarray or list of lists): Binary label matrix for the dataset.
        """
        self.labels = np.array(labels) if isinstance(labels, list) else labels
        self.label_frequencies = self.labels.sum(axis=0)
        self.max_frequency = self.label_frequencies.max()

    @property
    def IRLbl(self):
        """
        Calculate and return the Imbalance Ratio per Label (IRLbl).

        Returns:
        - np.ndarray: IRLbl values for each label.
        """
        return self.max_frequency / self.label_frequencies

    @property
    def MeanIR(self):
        """
        Calculate and return the Mean Imbalance Ratio (MeanIR).

        Returns:
        - float: Mean of the IRLbl values.
        """
        return self.IRLbl.mean()

    @property
    def CVIR(self):
        """
        Calculate and return the Coefficient of Variation of IRLbl (CVIR).

        Returns:
        - float: CVIR value.
        """
        return self.IRLbl.std() / self.MeanIR


def get_metrics(y_true, y_pred, label_names, image_path):
    """
    Prints F1-score, precision, and recall for all classes, top 5 majority classes, and the rest minority classes.

    Output:
        - f1-scores for all classes.
    """
    print(f'\nMean number of label assignments per image in ground-truth: {np.sum(y_true) / y_true.shape[0]:.4f}')
    print(f'Mean number of label assignments per image in predictions: {np.sum(y_pred) / y_pred.shape[0]:.4f}\n')

    n_labels = y_pred.shape[1]
    metrics_df = pd.DataFrame(
        classification_report(y_true, y_pred, target_names=label_names, output_dict=True)).transpose()
    metrics_df['index'] = np.concatenate((np.arange(start=0, stop=n_labels), [None, None, None, None]))
    print(metrics_df)

    # Sort the classes by size, descending order
    sorted_class_sizes = np.argsort(np.sum(y_true, axis=0))[::-1]
    print(f'Ordered class in descending order of samples:\n {np.array(label_names)[sorted_class_sizes]}')

    # F1-scores
    sorted_f1score_per_class = metrics_df['f1-score'][0:n_labels][sorted_class_sizes]

    print(f'\nUnweighted avg. F1-score of all classes: {np.sum(sorted_f1score_per_class) / n_labels}')
    print(f'Unweighted avg. F1-score of top 5 classes: {np.sum(sorted_f1score_per_class[:5]) / 5}')
    print(f'Unweighted avg. F1-score of the rest: {np.sum(sorted_f1score_per_class[5:]) / (n_labels - 5)}\n')

    # Precision
    sorted_precision_per_class = metrics_df['precision'][0:n_labels][sorted_class_sizes]

    print(f'\nUnweighted avg. precision of all classes: {np.sum(sorted_precision_per_class) / n_labels}')
    print(f'Unweighted avg. precision of top 5 classes: {np.sum(sorted_precision_per_class[:5]) / 5}')
    print(f'Unweighted avg. precision of the rest: {np.sum(sorted_precision_per_class[5:]) / (n_labels - 5)}\n')

    # Recall
    sorted_recall_per_class = metrics_df['recall'][0:n_labels][sorted_class_sizes]

    print(f'\nUnweighted avg. recall of all classes: {np.sum(sorted_recall_per_class) / n_labels}')
    print(f'Unweighted avg. recall of top 5 classes: {np.sum(sorted_recall_per_class[:5]) / 5}')
    print(f'Unweighted avg. recall of the rest: {np.sum(sorted_recall_per_class[5:]) / (n_labels - 5)}\n')

    if image_path:
        _ = plt.figure(figsize=(8, 14))

        _ = plt.title('F1-score per class')
        _ = plt.barh(range(y_true.shape[1]), sorted_f1score_per_class, color='blue', alpha=0.6)
        _ = plt.yticks(ticks=range(n_labels), labels=np.array(label_names)[sorted_class_sizes])
        _ = plt.xlabel('F1-score')
        _ = plt.grid(True)
        save_img(image_path)

    return metrics_df['f1-score'][0:n_labels]


# ====================================== Utilities ==========================================



def print_time(start, ms=False):
    end = time.time()
    try:
        if ms:
            total_time_in_ms = round((end - start) * 1000, 3)
            print(f'Time passed: {total_time_in_ms} ms\n')
        else:
            total_time_in_hours = round((end - start) / 3600, 2)
            print(f'Time passed: {total_time_in_hours} hours')
            print(time.strftime("%H:%M:%S", time.localtime()))
    except:
        print('failed to print time')


def save_img(image_path):
    try:
        plt.savefig(image_path, bbox_inches='tight')
    except:
        print(f'ERROR: Could not save image {image_path}')


