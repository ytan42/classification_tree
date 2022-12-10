import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score


X, y = make_classification(n_samples=4000, n_features=10, n_classes=2, flip_y=0.25, weights=(0.7, 0.3))


def gini(y_leaf : np.ndarray) -> float:
    """Calculate the Gini index
    Returns: float
    """
    _, k_n = np.unique(y_leaf, return_counts = True)
    n = len(y_leaf)
    # If only 1 category left then return gini of 0
    if len(k_n) == 1:
        return 0.0
    else:
        # Cumulative sum of prob of each category in this leaf
        index = 1 - sum([(k / n) ** 2 for k in k_n])
        return index

# y = np.array([1,1,1,0,0])
# gini(y)

def entropy(y_leaf : np.ndarray) -> float:
    """Calculates the entropy
    Returns: float
    """
    _, k_n = np.unique(y_leaf, return_counts = True)
    n = len(y_leaf)
    if len(k_n) == 1:
        return 0.0
    else:
        e = -sum([(k / n) * np.log2(k / n) for k in k_n])
        return e

# y = np.array([1,1,1,0,0])
# entropy(y)

def mid_point(v : np.ndarray) -> np.ndarray:
    """Given a 1D vector, return a 1D vector of mid points with n - 1 elements

    Args:
        v (np.array): A vector of numeric values

    Returns:
        np.array: A vectors of mid-points
    
    Usage:
        v = [0, 1, 1, 2]
        unique_element = [0, 1, 2]
        returns [0.5, 1.5]
    """

    v_unique = np.unique(v) # ordered unique values

    if len(v_unique) == 1:
        return None

    v_n1 = v_unique[:-1]    # first number in calculating average eg. [0, 1]
    v_n2 = v_unique[1:]     # second number in calculating average eg. [1, 2]
    mid = np.mean([v_n1, v_n2], axis=0)

    return mid

# mid_point(np.array([1, 2, 5, 3, 0]))


def best_split(x : np.ndarray, 
               y : np.ndarray,
               eval_func : callable) -> tuple:
    """Find the best split of X based on infomation gain of y

    Args:
        x (np.ndarray): Predictors
        y (np.ndarray): Target variable
        eval_func (callable): Either Gini or Entropy function to determine information gain

    Returns:
        tuple: (col_idx, mid_point, info_gain)
    """
    
    # Existing gini or the entory before a split
    value_existing = eval_func(y)

    # column index of the X
    col_idx = 0
    col_idx_max = x.shape[1]
    split_results = list()      # split results contains cut points and info gains for each column of x

    # Loop over all columns of x
    while col_idx < col_idx_max:
        
        col_result = list()     # col result stores cut points and info gain for each cut
        cut_points = list()     # a list of cut points
        info_gains = list()     # Change in gini or the entory for a given cut
        
        # Map the mid point function to the current column
        # This give us a list of mid points we can cut
        points = mid_point(x[:, col_idx])

        # Make a cut for all mid-points and calc the metric using eval_func
        # NOTE here we define the left node as <lesser than> and right node asa <greater or equal to>
        # This must not be confused in the recurrsion

        for point in points:

            mask = x[:, col_idx] < point
            y_left = y[mask]
            y_left_weight = len(y_left) / len(y)

            y_right = y[~mask]
            y_right_weight = len(y_right) / len(y)

            # Evaluate the left and right node
            # The weighted avg is the gini (or entory) for the split
            y_left_value = eval_func(y_left)
            y_right_value = eval_func(y_right)

            y_weighted_value = y_left_value * y_left_weight + y_right_value * y_right_weight
            info_gain = value_existing - y_weighted_value

            # Add the cut point and the info gain to the list
            cut_points.append(point)    
            info_gains.append(info_gain)

        col_result.append(cut_points)
        col_result.append(info_gains)

        # Add the result for the column to the result set
        split_results.append(col_result)
        col_idx = col_idx + 1

    # Once we looped over all columns and all mid-points, we need to pick the best
    # one in terms of information gain 
    results = np.array(split_results)

    # The results object is a 3D numpy array where
    # Dimension i: column of x
    # Dimension j: value of mid points
    # Dimension k: info gain of the cut
    # Index [:, 1:, :] is to skip dimention j
    i, j, k = np.unravel_index(results[:, 1:, :].argmax(), results[:, 1:, :].shape)
    
    best_col_idx = i
    best_mid_point = results[i, j, k]
    highest_info_gain = results[i, j+1, k]

    return best_col_idx, best_mid_point, highest_info_gain

# best_col_idx, best_mid_point, highest_info_gain = best_split(x=X, y=y, eval_func=gini, min_samples_leaf=2)

def perform_split(X : np.ndarray, 
                  y : np.ndarray, 
                  col_idx : int, 
                  mid_point : float) -> tuple:
    """Performs a split of X, y based on col_idx of X and the split point (ie. mid_point)
    
    Returns:
        tuple: tuple of X_left, y_left, X_right, y_right
    """
    mask = X[:, col_idx] < mid_point
    X_left = X[mask, :]
    y_left = y[mask]
    X_right = X[~mask, :]
    y_right = y[~mask]

    return X_left, y_left, X_right, y_right

# perform_split(X, y, 1, 0.22638)

def calc_proba(leaf) -> np.ndarray:
    """Calculates the class and probability of the class given a leaf node

    Args:
        leaf : nothing but an array of y values

    Returns: tuple in the format of (most_freq_value_y, proba_of_y), eg. (1, 0.85)
    """
    if isinstance(leaf, np.ndarray):
        values, counts = np.unique(leaf, return_counts=True)
        ind = np.argmax(counts)
        most_freq_class = values[ind]
        proba = counts[ind] / sum(counts)
        return np.array([most_freq_class, proba])
    else:
        raise("Leaf node must be a numpy array")

# calc_proba(np.array([0, 0, 0, 0, 0, 1]))

def fit_tree(X : np.ndarray, 
             y : np.ndarray,
             eval_func : callable, 
             min_samples_leaf : int = 2,
             max_depth : int = 5,
             depth=0):
    """Recursively apply the fit_tree function to the left and right node
    until certain termination criteria is met

    Args:
        X (np.array): Predictors, np.ndarray
        y (np.array): Target variable
        eval_func (callable): Gini or Entropy
        min_samples_leaf (int, optional): Minimum number of records in a leaf node. Defaults to 2.
        max_depth (int, optional): Maximum depth of the tree. Defaults to 5.
        left_depth (int, optional): A variable to track left branch depth. Defaults to 0.
        right_depth (int, optional): A variable to track right branch depth. Defaults to 0.

    Returns:
        tree: The tree object is a nested tuple
    """

    # Perform a split if the number of records in X is greater than min_samples_leaf
    # We try make the split first and then test whether or not the value of info_gain has reached to 0
    if len(X) > min_samples_leaf and depth + 1 <= max_depth:
        col_idx, mid_point, info_gain = best_split(X, y, eval_func)
        
        # We proceed with a split when info_gain is greater than 0 and the depth of the current branch hasn't reached the max_depth
        if info_gain > 0:
            X_left, y_left, X_right, y_right = perform_split(X, y, col_idx, mid_point)

            left_branch = fit_tree(X=X_left, y=y_left, eval_func=eval_func, max_depth=max_depth, depth=depth+1)
            right_branch = fit_tree(X=X_right, y=y_right, eval_func=eval_func, max_depth=max_depth, depth=depth+1)
        else:
            return calc_proba(y)
    else:
        return calc_proba(y)

    return (col_idx, mid_point, (left_branch, right_branch), np.unique(y))


# tree = fit_tree(X=X, y=y, eval_func=gini, min_samples_leaf=2, max_depth=3)
# tree

def print_tree(tree, spacing=""):
    """Helper function to print a tree in a more readable manner
    """

    # Since we don't have a lead node object, we just check whether it's a tuple
    # True means we have reached a leaf node. 
    # Remember that the lead_node is in the format of (y_value, y_proba)
    if isinstance(tree, np.ndarray):
        most_freq_class, pred_proba = tree
        print(f"{spacing} Predict: {most_freq_class}, Probability: {round(pred_proba, 2)}")
        return
    else:
        col_idx = tree[0]
        mid_point = tree[1]
        print(f"{spacing} Column: {col_idx}, Splitting value: {round(mid_point, 2)}")

        print(f"{spacing} --> less than {round(mid_point, 2)}")
        print_tree(tree=tree[2][0], spacing=spacing + "  ")

        print(f"{spacing} --> greater or equal to {round(mid_point, 2)}")
        print_tree(tree=tree[2][1], spacing=spacing + "  ")

# print_tree(tree)

def pred_record(X : np.ndarray, tree : tuple) -> tuple:
    """Predicts a single record of X

    Args:
        X (np.ndarray): A single record of X [1, n]
        tree (tuple): Nested tuple

    Returns:
        tuple: A tuple of most_freq_class, pred_proba
    """

    # The leaf_node (ie. the np array) s
    if isinstance(tree, np.ndarray):
        most_freq_class, pred_proba = tree
        return most_freq_class, pred_proba
    else:
        # Refer to the fit_tree function for the "magic" index number
        col_idx = tree[0]
        mid_point = tree[1]
        X_val = X[col_idx]
        # Remember that the left node is <less than> and right node is <greater or equal to>
        # If the node is not a terminal node, the left node is in tree[2][0] and right is in tree[2][1]
        # tree[2][0] and tree[2][1] are essentially sub-tree
        if X_val < mid_point:
            return pred_record(X, tree[2][0])
        else:
            return pred_record(X, tree[2][1])

# pred_record(X[3, ], tree)

def predict_proba(X : np.ndarray, 
                  tree : tuple):

    """Predicts the entire X matrix
    """

    classes = tree[3]
    # Apply over dimension 1 of the X matrix, ie. apply the func to each row of X
    # Remember that pred_record function returns tuple of (pred_class, pred_proba)
    predictions = np.apply_along_axis(pred_record, axis=1, arr=X, tree=tree)

    # We want to convert the predictions object from [pred_class, pred_proba]
    # to [class_0_pred_proba, class_1_pred_proba, class_2_pred_proba, ...]
    proba_dict = dict()
    for sub_class in classes:
        class_proba = np.where(predictions[:, 0] == sub_class, predictions[:, 1], 1-predictions[:, 1]).reshape(-1, 1)
        proba_dict[sub_class] = class_proba

    pred_proba = np.hstack((proba_dict[0], proba_dict[1]))

    return pred_proba


def predict(X : np.ndarray, tree : tuple):

    predictions = np.apply_along_axis(pred_record, axis=1, arr=X, tree=tree)

    return np.reshape(predictions[:, 0], (-1, 1))


def main():

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

    tree = fit_tree(X=X_train, y=y_train, eval_func=gini, min_samples_leaf=2, max_depth=5)
    pred_proba = predict_proba(X_test, tree)
    pred_class = predict(X_test, tree)

    acc = accuracy_score(y_test, pred_class)
    roc = roc_auc_score(y_test, pred_proba[:, 1])

    print(f"The accuracy is: {acc}")
    print(f"The ROC AUC is: {roc}")

if __name__ == '__main__':
    main()