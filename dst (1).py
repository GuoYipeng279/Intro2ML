import numpy as np
import math
from numpy.random import default_rng
import matplotlib.pyplot as plt
import copy

# need to re-initialize this variable if train a new tree
count_leaf = 0
class_number = 0


def load_data(file_path):
    global class_number
    dataset = np.loadtxt(file_path)
    class_number = len(np.unique(dataset[:, -1]))
    return np.loadtxt(file_path)


def plot_node(node, cur_x, cur_y, current_level, widths, constant_x, constant_y, width_x_before):
    # check it is a node or leaf
    if isinstance(node[0], dict):
        # if it is a node, plot itself and call this function, use its childs as new node
        plt.text(cur_x-0.5, cur_y,
                 f"x{node[0]['attribute']} < {node[0]['value']}",
                 bbox=dict(boxstyle='round'))

        # some parameters try to avoid overlap between trees
        x_space = constant_x / widths[current_level]
        if (x_space >= width_x_before):
            x_space = width_x_before * 0.8
        width_x_before = x_space

        # plot left child
        plt.plot([cur_x, cur_x - x_space], [cur_y, cur_y - constant_y], '-')
        plot_node(node[0]['left'], cur_x - x_space, cur_y - constant_y,
                  current_level + 1, widths, constant_x, constant_y, width_x_before)

        # plot right child
        plt.plot([cur_x, cur_x + x_space], [cur_y, cur_y - constant_y], '-')
        plot_node(node[0]['right'], cur_x + x_space, cur_y - constant_y,
                  current_level + 1, widths, constant_x, constant_y, width_x_before)
    else:
        # plot leaf directly
        plt.text(cur_x-0.5, cur_y, f"leaf: {node[0]}")


def plot(dst, file_path):
    widths = get_widths(dst)
    plt.rcParams['figure.figsize'] = [200, 100]
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['figure.facecolor'] = 'black'
    start_x = -0.2
    start_y = 0
    plot_node(dst, start_x, start_y, 1, widths, 200, 2, 100)
    plt.savefig(file_path)
    print("\ndepth of ploted tree: \n", dst[1])


def get_widths(dst):
    current_child = [dst]
    widths = [1]

    while (len(current_child) != 0):
        counter = 0
        length = len(current_child)
        for i in range(length):
            current_node = current_child.pop(0)
            if (isinstance(current_node[0], dict)):
                current_child.append(current_node[0]['left'])
                current_child.append(current_node[0]['right'])
            counter += 2
        widths.append(counter)

    return widths


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


def decision_tree_learning(dataset, depth, parent, prev_direction):
    global count_leaf
    # return when all the samples has the same label
    all_labels = set(dataset[:, -1])
    label_list = np.array(dataset[:, -1],int)
    sample_distribution = np.bincount(label_list, minlength=5)
    if (len(all_labels) == 1):
        count_leaf += 1
        return [all_labels.pop(), depth, sample_distribution, parent, prev_direction]
    else:
        # find a split value with the highest information gain
        split_attribute, split_value, left_dataset, right_dataset = find_split(dataset)
        # build the tree iteratively
        node_property = {'attribute': split_attribute, 'value': split_value, 'left': None, 'right': None}
        # node, left and right are all list objects
        # consist of [node/leaf, depth, sample distribution, parent, it is left child or right child]
        new_node = [node_property, None, sample_distribution, parent, prev_direction]
        left = decision_tree_learning(left_dataset, depth+1, new_node, 'left')
        right = decision_tree_learning(right_dataset, depth+1, new_node, 'right')
        new_node[0]['left'] = left
        new_node[0]['right'] = right
        new_node[1] = max(left[1], right[1])
        return new_node


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

    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    return (accuracy, confusion_matrix)


def update_depth(new_node):
    current_node = new_node
    # exit the loop when current node doesn't have parent node
    # which means it is root
    while(current_node[3] is not None):
        parent_node = current_node[3]
        if (current_node[4] == 'left'):
            another_child = 'right'
        else:
            another_child = 'left'
        # if current node's depth < another child node
        # the update can stop because the new node has
        # no effect on its parent's depth
        if (current_node[1] < parent_node[0][another_child][1]):
            break
        # otherwise update parent node's depth and continue
        else:
            parent_node[1] = current_node[1]
            current_node = parent_node


def find_all_possible_pruning(dst):
    nodes = []
    initial_node = dst
    stack = [initial_node]

    while(len(stack) != 0):
        current_node = stack.pop()
        # if the current node is a leaf, skip
        if (not isinstance(current_node[0], dict)):
            continue
        # if a node with two leaves then it is a node can be pruned potentially
        if (isinstance(current_node[0]['left'][0], float) and isinstance(current_node[0]['right'][0], float)):
            nodes.append(current_node)
        else:
            stack.append(current_node[0]['left'])
            stack.append(current_node[0]['right'])

    return nodes


def pruning(dst, validation_set):
    # accuracy before pruning
    current_accuracy, _ = evaluate(validation_set, dst)
    # find all possible pruning nodes
    potential_pruning_nodes = find_all_possible_pruning(dst)
    # check one node each time
    while (len(potential_pruning_nodes) != 0):
        # each node just need to check one time
        current_node = potential_pruning_nodes.pop()
        # prune and create a new node
        # the new label is the majority class at current stage according to the training set
        # if there are multiple class with same number of samples
        # break the tie by selecting the first one
        n_sample = current_node[2]
        # convert to float to keep consistent with other leave
        new_label = float(np.argmax(n_sample))
        parent_node = current_node[3]
        new_depth = current_node[1] - 1
        left_or_right_child = current_node[4]
        new_node = [new_label, new_depth, n_sample, parent_node, left_or_right_child]
        parent_node[0][left_or_right_child] = new_node
        # evaluate the new tree using validation set
        new_accuracy, _ = evaluate(validation_set, dst)
        # keep the pruning if validation accuracy increase or unchanges
        if (new_accuracy >= current_accuracy):
            current_accuracy = new_accuracy
            # update the depth of the tree due to this change
            update_depth(new_node)
            # if a node is pruned, this may possibly make its parent prunable
            # check it has two leaves now
            if (isinstance(parent_node[0]['left'][0], float) and isinstance(parent_node[0]['right'][0], float)):
                potential_pruning_nodes.append(parent_node)
        # if validation accuracy decrease, cancel the pruning
        else:
            parent_node[0][left_or_right_child] = current_node


def k_fold_cross_validation(dataset, n_fold, prune=False):
    # shuffle the dataset
    np.random.shuffle(dataset)
    labels = np.unique(dataset[:, -1])
    # add sample with the same label into list
    data_each_class = {}
    for i in range(1, len(labels)+1):
        data_each_class[i] = dataset[dataset[:, -1] == i]

    confusion_matrix = np.zeros(shape=(class_number,class_number))
    total_depth = 0
    counter = 0

    # split into n fold
    # use 1 fold as testing set, n-1 folds as training set
    for i in range(n_fold):
        training_set = np.zeros(shape=(1,len(dataset[0])))
        testing_set = np.zeros(shape=(1,len(dataset[0])))
        # record the remaining data for pruning
        remain_each_class = {}
        # stratified proportional sampling to ensure the data distribution in training and testing set
        # or in other words the proportion of each class is almost the same as the original dataset
        for j in range(1, len(labels)+1):
            n_test_class_j = round(1/n_fold * len(data_each_class[j]))
            index = i * n_test_class_j
            testing_set_j = data_each_class[j][(0+index):(n_test_class_j+index)]
            testing_set = np.vstack((testing_set, testing_set_j))
            training_set_j = np.delete(data_each_class[j], np.s_[(0+index):(n_test_class_j+index)], 0)
            remain_each_class[j] = copy.copy(training_set_j)
            training_set = np.vstack((training_set, training_set_j))
        # remove the initialization line with all zeros
        training_set = training_set[1:]
        testing_set = testing_set[1:]
        if (prune):
            # the remaining dataset consist of n-1 fold, so the inner cross-validation
            # loop n-1 times
            for k in range(n_fold-1):
                validation_set = np.zeros(shape=(1, len(dataset[0])))
                training_set = np.zeros(shape=(1, len(dataset[0])))
                # use stratified proportional sampling again, but this time for training and validation set
                for l in range(1, len(labels)+1):
                    n_valid_class_l = round(1/n_fold * len(data_each_class[l]))
                    index = k * n_valid_class_l
                    validation_set_l = remain_each_class[l][(0 + index):(n_valid_class_l + index)]
                    validation_set = np.vstack((validation_set, validation_set_l))
                    training_set_l = np.delete(remain_each_class[l], np.s_[(0 + index):(n_valid_class_l + index)], 0)
                    training_set = np.vstack((training_set, training_set_l))
                # remove the initialization line with all zeros
                training_set = training_set[1:]
                validation_set = validation_set[1:]
                dst = decision_tree_learning(training_set, 0, None, None)
                pruning(dst, validation_set)
                total_depth += dst[1]
                _, i_times_k_confusion_matrix = evaluate(testing_set, dst)
                counter += 1
                confusion_matrix += i_times_k_confusion_matrix
        else:
            # use training set to build tree then pass it for evaluation
            dst = decision_tree_learning(training_set, 0, None, None)
            total_depth += dst[1]
            _, ith_fold_confusion_matrix = evaluate(testing_set, dst)
            counter += 1
            confusion_matrix += ith_fold_confusion_matrix

    # calculate different metric
    confusion_matrix = confusion_matrix / counter
    average_accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    average_depth = total_depth / counter
    precisions = []
    recalls = []
    F1_scores = []
    for i in range(1, class_number+1):
        TP_i = confusion_matrix[i-1][i-1]
        FP_i = np.sum(confusion_matrix[:i-1, i-1]) + np.sum(confusion_matrix[i:, i-1])
        FN_i = np.sum(confusion_matrix[i-1, :i-1]) + np.sum(confusion_matrix[i-1, i:])
        precision_i = TP_i / (TP_i + FP_i)
        recall_i = TP_i / (TP_i + FN_i)
        F1_score_i = 2*precision_i*recall_i / (precision_i + recall_i)
        precisions.append(precision_i)
        recalls.append(recall_i)
        F1_scores.append(F1_score_i)

    print("average accuracy: \n", average_accuracy)
    print("\naverage depth: \n", average_depth)
    print("\nconfusion matrix: \n", confusion_matrix)
    print("\nprecision per class: \n", precisions)
    print("\nrecall per class: \n", recalls)
    print("\nF1 score per class: \n", F1_scores)

    return (average_accuracy, confusion_matrix, precisions, recalls, F1_scores, dst)


def calculate_std_and_mean(dataset):
    labels = np.unique(dataset[:, -1])
    # add sample with the same label into list
    data_each_class = {}
    for i in range(1, len(labels)+1):
        data_each_class[i] = dataset[dataset[:, -1] == i]

    for c in range(1, len(labels)+1):
        for a in range(len(dataset[0]) - 1):
            std = np.std(data_each_class[c][:, a])
            print(std)

    for c in range(1, len(labels)+1):
        for a in range(len(dataset[0]) - 1):
            mean = np.mean(data_each_class[c][:, a])
            print(mean)


def main():
    # load clean dataset
    dataset = load_data('wifi_db/clean_dataset.txt')
    # k fold cross validation for clean dataset
    print("clean dataset \n")
    accuracy, confusion_matrix, precision, recall, f1_score, dst = k_fold_cross_validation(dataset, 10, False)

    # load noisy dataset
    dataset = load_data('wifi_db/noisy_dataset.txt')
    # k fold cross validation for noisy dataset
    print("\nnoisy dataset \n")
    accuracy, confusion_matrix, precision, recall, f1_score, dst = k_fold_cross_validation(dataset, 10, False)

    # uncomment it if you want to visualize a dst
    """
    # load clean dataset
    dataset = load_data('wifi_db/clean_dataset.txt')
    # plot the tree that trained on the entire dataset
    print("\nploting dst \n")
    dst = decision_tree_learning(dataset, 0, None, None)
    plot(dst, 'dst_on_clean_entre_dataset.png')
    """

    return


if __name__ == '__main__':
    main()