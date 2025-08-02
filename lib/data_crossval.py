import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def load_data(file_path,label_path):
    label_list = [line.strip() for line in open(label_path)]
    data, labels, targets = [], [], []
    label2id = {label: i for i, label in enumerate(label_list)}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            features, label = line.strip().split('\t')
            data.append(features)
            labels.append(label)
            label = [label2id[l] for l in label.split(',')]
            target = np.zeros(9).astype(np.float32)
            target[label] = 1.0
            targets.append(target)
    return data, labels,targets


def save_fold_data(fold_num, train_data, test_data, train_labels, test_labels):
    import os
    path = "../data/mRNA/folds/"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+f'fold_{fold_num}_train.txt', 'w', encoding='utf-8') as f_train:
        for features, label in zip(train_data, train_labels):
            f_train.write(f"{features}\t{label}\n")

    with open(path+f'fold_{fold_num}_test.txt', 'w', encoding='utf-8') as f_test:
        for features, label in zip(test_data, test_labels):
            f_test.write(f"{features}\t{label}\n")


def split_and_save_k_fold(data, labels,targets,n_splits=5):
    kf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2024)
    fold_num = 1
    for train_index, test_index in kf.split(data,targets):
        train_data, test_data = np.array(data)[train_index], np.array(data)[test_index]
        train_labels, test_labels = np.array(labels)[train_index], np.array(labels)[test_index]
        save_fold_data(fold_num, train_data, test_data, train_labels, test_labels)
        fold_num += 1


# Load data
data_path = '../data/mRNA/train.txt'
label_path = '../data/mRNA/label.txt'
data, labels,targets = load_data(data_path,label_path)

# Split and save data
split_and_save_k_fold(data, labels,targets)