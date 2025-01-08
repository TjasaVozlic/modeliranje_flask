import numpy as np


def normalize_criteria(matrix, criteria_types):
    """
    Normalize the decision matrix based on criteria types (benefit or cost).

    Parameters:
        matrix (numpy.ndarray): The decision matrix (alternatives x criteria).
        criteria_types (list): A list of strings specifying each criterion type:
                               'benefit' for benefit criteria,
                               'cost' for cost criteria.

    Returns:
        numpy.ndarray: The normalized decision matrix.
    """
    normalized_matrix = np.zeros_like(matrix, dtype=float)

    for j in range(matrix.shape[1]):  # Iterate over columns (criteria)
        if criteria_types[j] == 'max':
            # For benefit criteria, higher is better
            min_val = np.min(matrix[:, j])
            max_val = np.max(matrix[:, j])
            normalized_matrix[:, j] = (matrix[:, j] - min_val) / (max_val - min_val)
        elif criteria_types[j] == 'min':
            # For cost criteria, lower is better
            min_val = np.min(matrix[:, j])
            max_val = np.max(matrix[:, j])
            normalized_matrix[:, j] = (max_val - matrix[:, j]) / (max_val - min_val)
        else:
            raise ValueError(f"Invalid criteria type '{criteria_types[j]}'. Use 'benefit' or 'cost'.")

    return normalized_matrix


def build_comparison_matrix(df, i):
    num_companies = len(df)
    comparison_matrix = np.ones((num_companies, num_companies))  # Inicializiraj matriko 1s
    df_criterion = df[:, i]

    for i in range(num_companies):
        for j in range(num_companies):
            if i != j:
                comparison_matrix[i, j] = df_criterion[i]/df_criterion[j]
            else:
                comparison_matrix[i, j] = 1

    normalized_comparison_matrix = comparison_matrix / comparison_matrix.sum(axis=0)
    row_averages = np.mean(normalized_comparison_matrix, axis=1)
    return row_averages


def normalize_criteria_ahp(matrix, criteria_types):
    """
    Normalize the decision matrix based on criteria types (benefit or cost).

    Parameters:
        matrix (numpy.ndarray): The decision matrix (alternatives x criteria).
        criteria_types (list): A list of strings specifying each criterion type:
                               'benefit' for benefit criteria,
                               'cost' for cost criteria.

    Returns:
        numpy.ndarray: The normalized decision matrix.
    """
    normalized_matrix = np.zeros_like(matrix, dtype=float)

    for j in range(matrix.shape[1]):  # Iterate over columns (criteria)
        if criteria_types[j] == 'max':
            # For benefit criteria, higher is better
            min_val = np.min(matrix[:, j])
            max_val = np.max(matrix[:, j])
            normalized_matrix[:, j] = 1 + (matrix[:, j] - min_val)
        elif criteria_types[j] == 'min':
            # For cost criteria, lower is better
            min_val = np.min(matrix[:, j])
            max_val = np.max(matrix[:, j])
            normalized_matrix[:, j] = 1 + (max_val - matrix[:, j])
        else:
            raise ValueError(f"Invalid criteria type '{criteria_types[j]}'. Use 'benefit' or 'cost'.")

    return normalized_matrix


# # Example Usage:
# decision_matrix = np.array([
#     [8.84, 8.79, 6.43, 6.95],  # Alternative 1
#     [8.57, 8.51, 5.47, 6.91],  # Alternative 2
#     [7.76, 7.75, 5.34, 8.76],  # Alternative 3
#     [7.97, 9.12, 5.93, 8.09]  # Alternative 4
# ])
#
# criteria_types = ['max', 'max', 'min', 'max']
#
# # Normalize the matrix
# normalized_matrix = normalize_criteria(decision_matrix, criteria_types)
#
# print("Normalized Decision Matrix:")
# print(normalized_matrix)
