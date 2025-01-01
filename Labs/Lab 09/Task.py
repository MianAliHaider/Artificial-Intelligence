import numpy as np
import pandas as pd

def entropy(data, target_col):
    elements, counts = np.unique(data[target_col], return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def information_gain(data, attribute, target_col):
    total_entropy = entropy(data, target_col)
    values, counts = np.unique(data[attribute], return_counts=True)
    weighted_entropy = 0
    total_count = np.sum(counts)

    for i in range(len(values)):
        subset = data[data[attribute] == values[i]]
        subset_entropy = entropy(subset, target_col)
        weight = counts[i] / total_count
        weighted_entropy += weight * subset_entropy
        information_gain = total_entropy - weighted_entropy
        return information_gain

def build_tree(data, attributes, target_col, parent_node_class=None):
    if len(np.unique(data[target_col])) <= 1:
        return np.unique(data[target_col])[0]
    elif len(data) == 0:
        return parent_node_class
    elif len(attributes) == 0:
        return np.unique(data[target_col])[np.argmax(np.unique(data[target_col], return_counts=True)[1])]
    else:
        parent_node_class = np.unique(data[target_col])[np.argmax(np.unique(data[target_col], return_counts=True)[1])]
        item_values = [information_gain(data, attribute, target_col) for attribute in attributes]
        best_attr_index = np.argmax(item_values)
        best_attr = attributes[best_attr_index]
        tree = {best_attr: {}}
        attributes = [i for i in attributes if i != best_attr]
        for value in np.unique(data[best_attr]):
            sub_data = data.where(data[best_attr] == value).dropna()
            subtree = build_tree(sub_data, attributes, target_col, parent_node_class)
            tree[best_attr][value] = subtree
        return tree

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


data = pd.DataFrame({ 'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'], 
                     'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'], 
                     'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'] })
attributes = ['Weather', 'Temperature']
target_col = 'Play'
tree = build_tree(data, attributes, target_col)
a = entropy(data, target_col)
print(f"Entropy: {a}")
data_point = {'Weather': 'Sunny','Temperature':'Mild'}
prediction = predict(tree, data_point)
print(f'Prediction: {prediction}')
