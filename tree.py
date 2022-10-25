import numpy as np
import math
from numpy.random import default_rng
import matplotlib as plt

# need to re-initialize this variable if train a new tree
count_leaf = 0
class_number = 4
max_depth = np.inf


def load_data(file_path):
    return np.loadtxt(file_path)


def find_split(dataset):
    # calculate H(all)
    h_all = 0
    for i in range(class_number):
        class_i_count = np.sum(dataset[:, -1] == i+1)
        p_i = class_i_count / len(dataset)
        # prevent corner case log0 when class_i_count = 0
        if (class_i_count != 0):
            h_all -= p_i * math.log2(p_i)
        else:
            h_all -= 0

    # record all information related to current max information gain
    current_max_information_gain = -100
    current_max_attribute = 0
    current_max_split_value = 0
    current_max_left_dataset = None
    current_max_right_dataset = None
    # sort the dataset by an attribute(column)
    # repeat k times, k = number of attributes
    for i in range(len(dataset[0])-1):
        current_dataset = dataset[dataset[:, i].argsort()]
        unique_values = np.unique(current_dataset[:, i])
        # for each attribute, try all possible split values
        # that is the mid value between each two nested samples
        for j in range(len(unique_values)-1):
            next_value = (unique_values[j+1] + unique_values[j]) / 2
            # split the dataset
            dataset_left = current_dataset[current_dataset[:, i] < next_value]
            dataset_right = current_dataset[current_dataset[:, i] >= next_value]
            # calculate information gain
            h_left = 0
            h_right = 0
            for k in range(class_number):
                class_i_count_left = np.sum(dataset_left[:, -1] == k+1)
                class_i_count_right = np.sum(dataset_right[:, -1] == k+1)
                # prevent corner case log0 when class_i_count = 0
                p_i_left = class_i_count_left / len(dataset_left)
                p_i_right = class_i_count_right / len(dataset_right)
                if (class_i_count_left != 0):
                    h_left -= p_i_left * math.log2(p_i_left)
                else:
                    h_left -= 0
                if (class_i_count_right != 0):
                    h_right -= p_i_right * math.log2(p_i_right)
                else:
                    h_right -= 0
            information_gain = h_all - len(dataset_left) / len(dataset) * h_left \
                               - len(dataset_right) / len(dataset) * h_right
            # record the current max
            if (information_gain > current_max_information_gain):
                current_max_attribute = i
                current_max_split_value = next_value
                current_max_information_gain = information_gain
                current_max_left_dataset = dataset_left
                current_max_right_dataset = dataset_right

    return (current_max_attribute, current_max_split_value, current_max_left_dataset, current_max_right_dataset)


def decision_tree_learning(dataset, depth):
    global count_leaf, max_depth
    # return when all the samples has the same label
    arr = np.array(dataset[:, -1],int)
    all_labels = set(arr)
    counter = np.bincount(arr,minlength=5)
    if (len(all_labels) == 1) or depth > max_depth:
        count_leaf += 1
        return (all_labels.pop(), depth, counter)
    else:
        # find a split value with the highest information gain
        split_attribute, split_value, left_dataset, right_dataset = find_split(dataset)
        # build the tree iteratively
        left_total = decision_tree_learning(left_dataset, depth+1)
        left_branch, left_depth, left_count = left_total
        right_total = decision_tree_learning(right_dataset, depth+1)
        right_branch, right_depth, right_count = right_total
        new_node = {'attribute': split_attribute, 'value': split_value, 'left': left_total, 'right': right_total}
        return (new_node, max(left_depth, right_depth), counter)


def evaluate(test_db, trained_tree):
    # row is the actual label, column is the prediction value
    confusion_matrix = np.zeros(shape=(class_number,class_number))
    for i in range(len(test_db)):
        test_sample = test_db[i]
        current_node = trained_tree
        # traverse the tree until encounter a leaf node
        while(isinstance(current_node[0], dict)):
            current_attribute = current_node[0]['attribute']
            current_split_value = current_node[0]['value']
            if (test_sample[current_attribute] < current_split_value):
                current_node = current_node[0]['left']
            else:
                current_node = current_node[0]['right']
        confusion_matrix[round(test_sample[-1])-1][round(current_node[0])-1] += 1

    return confusion_matrix


def k_fold_cross_validation(dataset, n_fold, enable_pruning=True):
    # shuffle the dataset
    np.random.shuffle(dataset)
    labels = np.unique(dataset[:, -1])
    # add sample with the same label into list
    data_each_class = {}
    for i in range(1, len(labels)+1):
        data_each_class[i] = dataset[dataset[:, -1] == i]

    # split into n fold
    # use 1 fold as testing set, n-1 folds as training set
    confusion_matrix = np.zeros(shape=(class_number,class_number))
    for i in range(n_fold):
        training_set = np.zeros(shape=(1,len(dataset[0])))
        testing_set = np.zeros(shape=(1,len(dataset[0])))
        for j in range(1, len(labels)+1):
            n_class_j = len(data_each_class[j])
            n_test_class_j = round(1/n_fold * len(data_each_class[j]))
            index = i * n_test_class_j
            testing_set_j = data_each_class[j][(0+index):(n_test_class_j+index)]
            testing_set = np.vstack((testing_set, testing_set_j))
            # take the % to prevent index exceed range after the addition
            training_set_j = np.vstack((data_each_class[j][(n_test_class_j+index):n_class_j],
                                       data_each_class[j][:index]))
            training_set = np.vstack((training_set, training_set_j))
        # remove the initialization line with all zeros
        training_set = training_set[1:]
        testing_set = testing_set[1:]
        # use training set to build tree then pass it for evaluation
        dst = decision_tree_learning(training_set, 0)

        
        if enable_pruning: pruning(dst, testing_set)
        ith_fold_confusion_matrix = evaluate(testing_set, dst)
        confusion_matrix += ith_fold_confusion_matrix

    # calculate different metric
    average_accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    precisions = []
    recalls = []
    F1_scores = []
    for i in range(1, class_number+1):
        TP_i = confusion_matrix[i-1][i-1]
        TN_i = np.sum(np.delete(np.delete(confusion_matrix, i-1, 0), i-1, 1))
        FP_i = np.sum(confusion_matrix[:i-1, i-1]) + np.sum(confusion_matrix[i:, i-1])
        FN_i = np.sum(confusion_matrix[i-1, :i-1]) + np.sum(confusion_matrix[i-1, i:])
        precision_i = TP_i / (TP_i + FP_i)
        recall_i = TP_i / (TP_i + FN_i)
        F1_score_i = 2*precision_i*recall_i / (precision_i + recall_i)
        precisions.append(precision_i)
        recalls.append(recall_i)
        F1_scores.append(F1_score_i)

    print("average accuracy: \n", average_accuracy)
    print("\nconfusion matrix: \n", confusion_matrix)
    print("\nprecision per class: \n", precisions)
    print("\nrecall per class: \n", recalls)
    print("\nF1 score per class: \n", F1_scores)


    return (average_accuracy, confusion_matrix, precisions, recalls, F1_scores)


def pruning(tree, validation):
    def travel(node):
        if type(node[0]) != dict: return node[0],node[2] # return label and composition of the node
        (left_leaf, left_count), (right_leaf, right_count) = travel(node[0]['left']), travel(node[0]['right'])
        if left_leaf: node[0]['left'] = (left_leaf, node[1]+1, left_count)
        if right_leaf: node[0]['right'] = (right_leaf, node[1]+1, right_count)
        new_count = left_count + right_count
        if left_leaf and right_leaf: # not None means leaf
            before = sum(np.diag(evaluate(validation, tree)))
            rec = node[0]['value']
            if new_count[left_leaf] > new_count[right_leaf]: node[0]['value'] = np.inf
            else: node[0]['value'] = -np.inf
            after = sum(np.diag(evaluate(validation, tree)))
            if before > after: # dont prune
                node[0]['value'] = rec
                return None, new_count
            else: # prune
                return max([left_leaf, right_leaf], key=lambda x: new_count[x]), new_count
        return None, new_count
    # print(type(tree))
    travel(tree)

def main():
    # load clean dataset
    dataset = load_data('wifi_db/clean_dataset.txt')
    # k fold cross validation for clean dataset
    print("clean dataset \n")
    accuracy, confusion_matrix, precision, recall, f1_score = k_fold_cross_validation(dataset, 10)

    # load noisy dataset
    dataset = load_data('wifi_db/noisy_dataset.txt')
    # k fold cross validation for noisy dataset
    print("\nnoisy dataset \n")
    accuracy, confusion_matrix, precision, recall, f1_score = k_fold_cross_validation(dataset, 10)

    return


if __name__ == '__main__':
    main()