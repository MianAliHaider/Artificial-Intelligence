import pandas as pd
import numpy as np

# Calculate entropy of a dataset
def calculate_entropy(data, target_col):
    elements, counts = np.unique(data[target_col], return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# Calculate information gain for an attribute
def calculate_information_gain(data, attribute, target_col):
    total_entropy = calculate_entropy(data, target_col)
    values, counts = np.unique(data[attribute], return_counts=True)
    weighted_entropy = 0
    total_count = np.sum(counts)

    for i in range(len(values)):
        subset = data[data[attribute] == values[i]]
        subset_entropy = calculate_entropy(subset, target_col)
        weight = counts[i] / total_count
        weighted_entropy += weight * subset_entropy

    information_gain = total_entropy - weighted_entropy
    return information_gain

# Build the decision tree
def build_tree(data, attributes, target_col, depth=0, max_depth=3):
    if len(np.unique(data[target_col])) <= 1:
        return np.unique(data[target_col])[0]
    elif len(data) == 0:
        return None
    elif len(attributes) == 0 or depth == max_depth:
        return np.unique(data[target_col])[np.argmax(np.unique(data[target_col], return_counts=True)[1])]
    else:
        best_attr = max(attributes, key=lambda attribute: calculate_information_gain(data, attribute, target_col))
        tree = {best_attr: {}}
        attributes = [i for i in attributes if i != best_attr]
        for value in np.unique(data[best_attr]):
            sub_data = data[data[best_attr] == value]
            subtree = build_tree(sub_data, attributes, target_col, depth + 1, max_depth)
            tree[best_attr][value] = subtree
        return tree

# Predict the class for a given data point
def predict(tree, data_point):
    for attribute in tree.keys():
        value = data_point[attribute]
        subtree = tree[attribute].get(value, None)
        if subtree is None:
            return None
        if isinstance(subtree, dict):
            return predict(subtree, data_point)
        else:
            return subtree

# Build Random Forest
def build_random_forest(data, attributes, target_col, n_trees=2):
    trees = []
    for _ in range(n_trees):
        bootstrap_data = data.sample(frac=1, replace=True)
        tree = build_tree(bootstrap_data, attributes, target_col)
        trees.append(tree)
    return trees

# Example Dataset
data = pd.DataFrame({
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Overcast', 'Rainy', 'Sunny', 'Sunny', 'Rainy', 'Rainy', 'Overcast', 'Overcast', 'Sunny'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Hot', 'Mild', 'Mild', 'Cool', 'Mild', 'Mild', 'Hot'],
    'Play?': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Building Random Forest
attributes = ['Weather', 'Temperature']
target_col = 'Play?'
trees = build_random_forest(data, attributes, target_col, n_trees=2)

# Make a prediction
data_point = {'Weather': 'Sunny', 'Temperature': 'Cool'}
predictions = [predict(tree, data_point) for tree in trees]
final_prediction = max(set(predictions), key=predictions.count)
print(f'Final Prediction: {final_prediction}')
