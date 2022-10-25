"""
Create decision tree
"""
import sys
# print(sys.version)
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=2)

# from numpy.random import default_rng
def load_dataset(filepath):
    data = np.loadtxt(filepath)
    return data

def compute_entropy(dataset):
    classes, counts = np.unique(dataset[:, -1], return_counts=True)
    p = counts / len(dataset)
    entropy = -np.sum(p * np.log2(p))
    return entropy

def compute_grain(dataset, left_subset, right_subset):
    remainder = len(left_subset) / len(dataset) * compute_entropy(left_subset) + \
                len(right_subset) / len(dataset) * compute_entropy(right_subset)
    
    gain = compute_entropy(dataset) - remainder
    return gain

def find_split_along_one_feature(dataset, feature_index):
    sorted_data = dataset[dataset[:, feature_index].argsort()]
    max_gain = 0
    best_split_point_value = None
    margin = 0.5
    for j in range(len(sorted_data)):
        split_point = sorted_data[j, feature_index]
        if split_point == np.min(sorted_data[:, feature_index]):
            split_point += margin
        elif split_point == np.max(sorted_data[:, feature_index]):
            split_point -= margin
        left_subset = sorted_data[sorted_data[:, feature_index] <= split_point]
        right_subset = sorted_data[sorted_data[:, feature_index] > split_point]
        current_gain = compute_grain(sorted_data, left_subset, right_subset)
        if current_gain >= max_gain:
            max_gain = current_gain
            best_split_point_value = split_point
    return max_gain, best_split_point_value

def find_split(training_dataset):
    # print("-------------------------------")
    # print(training_dataset.shape)
    best_gain = 0
    best_split_attribute = None
    best_split_value = None
    for i in range(len(training_dataset[0])-1):
        feature_gain, feature_split_value = find_split_along_one_feature(training_dataset, i)
        # print()
        # print("attribute index: ", i)
        # print("current feature gain: ", feature_gain)
        # print("best gain is: ", best_gain)
        if feature_gain >= best_gain:
            best_gain = feature_gain
            best_split_attribute = i
            best_split_value = feature_split_value
        # if best_split_value is None:
            # print("No best split value")
    # print("best split attribute: ", best_split_attribute)
    # print("best_split_value: ", best_split_value)
    return best_split_attribute, best_split_value


def decision_tree_learning(training_dataset, depth):
    if np.all(training_dataset[:, -1] == training_dataset[0, -1]):
        leaf_node = {
            "attribute": None,
            "value": training_dataset[0, -1], # store the label
            "left": None,
            "right": None
        }
        return leaf_node, depth
    else:
        split_attribute, split_value = find_split(training_dataset)
        # sort training_dataset by the split_attribute
        sorted_dataset = training_dataset[training_dataset[:, split_attribute].argsort()]
        l_dataset = sorted_dataset[sorted_dataset[:, split_attribute] <= split_value]
        r_dataset = sorted_dataset[sorted_dataset[:, split_attribute] > split_value]

        l_branch, l_depth = decision_tree_learning(l_dataset, depth+1)
        r_branch, r_depth = decision_tree_learning(r_dataset, depth+1)
        node = {
            "attribute": split_attribute,
            "value": split_value,
            "left": l_branch,
            "right": r_branch
        }
        return node, max(l_depth, r_depth)


def predict_with_dt(dt, data_point):
    # Return dt['value'] as the label for data_point if encounters a leaf node
    if dt['left'] == dt['right'] == None:
        return dt['value']
    
    # Otherwise, predict the label by searching through the tree recursively
    test_attribute = dt['attribute']
    test_value = dt['value']
    if data_point[test_attribute] <= test_value:
        return predict_with_dt(dt['left'], data_point)
    else:
        return predict_with_dt(dt['right'], data_point)


def evaluate(dt, val_set):
    # get the number of classes and number of instances for each class
    classes, counts = np.unique(val_set[:, -1], return_counts=True)
    # matrix to store prediction result
    num_of_class = len(classes)
    preds = np.zeros((num_of_class, num_of_class), dtype=int)
    for data_point in val_set:
        true_label = int(data_point[-1])
        # print(true_label)
        # print(type(true_label))
        pred_label = int(predict_with_dt(dt, data_point))
        # print(pred_label)
        # print(type(pred_label))
        preds[pred_label-1, true_label-1] += 1
    print(preds)
    accuracies = [preds[i, i] / counts[i] for i in range(num_of_class)]
    print(accuracies)
    total_acc = np.trace(preds) / len(val_set)
    print(total_acc)
    return preds, accuracies, total_acc


def k_fold_cross_validation(dataset, n_fold=10):
    np.random.shuffle(dataset)

    interval = int(len(dataset) / n_fold)
    num_class = len(np.unique(dataset[:, -1]))
    decision_trees = []
    depths = []
    pred_matrices = []
    accuracy_of_each_class = []
    acc_of_each_tree = []

    for i in range(n_fold):
        print("----------------------")
        print(f"Validation num: {i+1}")
        val_set = dataset[i*interval : (i+1)*interval]
        train_set = np.delete(dataset, slice(i*interval, (i+1)*interval), axis=0)
        dt, depth = decision_tree_learning(train_set, depth=0)
        decision_trees.append(dt)
        depths.append(depth)
        preds, class_acc, tree_acc = evaluate(dt, val_set)
        pred_matrices.append(preds)
        accuracy_of_each_class.append(class_acc)
        acc_of_each_tree.append(tree_acc)
    confusion_matrix = np.sum(pred_matrices, axis=0)
    print("class accuracy for each tree: ")
    pp.pprint(accuracy_of_each_class)
    print()
    print("accuracy of each tree")
    print(acc_of_each_tree)
    print("depth of each tree")
    print(depths)
    
    avg_acc = np.average(acc_of_each_tree)
    precisions = [confusion_matrix[i,i] / np.sum(confusion_matrix[i])
                for i in range(num_class)]
    recalls = [confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
                for i in range(num_class)]
    f1_scores = [2*precisions[i]*recalls[i] / (precisions[i]+recalls[i])
                for i in range(num_class)]
    return decision_trees, confusion_matrix, avg_acc, precisions, recalls, f1_scores


if __name__ == "__main__":
    print("Clean dataset")
    data = load_dataset("wifi_db/clean_dataset.txt")
    trees, cm, avg_acc, precisions, recalls, f1_scores = k_fold_cross_validation(data)
    print(cm)
    print("Average accuracy: ", avg_acc)
    print("Precision for each class: ", precisions)
    print("Recalls for each class: ", recalls)
    print("F1 scores for each class: ", f1_scores)

    print()
    print("------------------")
    print("Noisy dataset")
    noisy_data = load_dataset("wifi_db/noisy_dataset.txt")
    trees, cm, avg_acc, precisions, recalls, f1_scores = k_fold_cross_validation(noisy_data)
    print(cm)
    print("Average accuracy: ", avg_acc)
    print("Precision for each class: ", precisions)
    print("Recalls for each class: ", recalls)
    print("F1 scores for each class: ", f1_scores)
    