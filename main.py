import pandas as pd
from flask import Flask, request, jsonify
from pymongo import MongoClient
import numpy as np
from pymcdm.methods import PROMETHEE_II
from pyDecision.algorithm import topsis_method, waspas_method, promethee_ii, ahp_method
from flask_cors import CORS
from helpers import normalize_criteria, build_comparison_matrix,normalize_criteria_ahp
import math
app = Flask(__name__)

# Enable CORS for the entire app
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

# Configure MongoDB client
client = MongoClient('mongodb://localhost:27017/')
db = client['local']  # Replace with your database name
collection = db['pripravljeni_podatki']

# Fetch all documents from the collection and convert them to a DataFrame
data = list(collection.find())  # Convert the cursor to a list
df = pd.DataFrame(data)  # Create a DataFrame

@app.route('/')
def home():
    return "Welcome to the MCDM API! Methods available: AHP, TOPSIS, PROMETHEE, WSM"


# AHP (Analytic Hierarchy Process)
@app.route('/ahp', methods=['POST'])
def ahp():
    data = request.json.get('data')
    column_order = request.json.get('selectedCriteria')

    weight_derivation = 'geometric'  # 'mean'; 'geometric' or 'max_eigen'

    # Define the types of criteria (1 for benefit, -1 for cost)
    types = ['max' if col not in ['debt_to_equity_ratio'] else 'min' for col in column_order]

    weights, rc = ahp_method(np.array(data), wd=weight_derivation)
    # Handle cases where rc might be NaN
    if rc is None or math.isnan(rc):
        rc = 0  # Assign 0 if rc is NaN or None

    normalized_matrix = normalize_criteria_ahp(df[column_order].to_numpy(), types)
    # Gradnja primerjalnih matrik za vsak kriterij
    prioritetna_matrika = np.zeros((20, len(column_order)))
    for i in range(len(column_order)):
        prioritetna_matrika[:, i] = build_comparison_matrix(normalized_matrix, i)

    # Multiply weights with the prioritetna matrix and get sum on rows
    final_scores = np.dot(prioritetna_matrika, weights)

    ranked_data = list(zip(df['Name'], final_scores))

    # Sort by relative_closeness (descending order, since higher is better)
    ranked_data.sort(key=lambda x: x[1], reverse=True)

    # Create ranks based on sorted data
    ranks = sorted(
        [
            {
                "rank": rank + 1,
                "name": name,
                "relative_closeness": closeness
            }
            for rank, (name, closeness) in enumerate(ranked_data)
        ],
        key=lambda x: x["rank"]
    )
    # Return the ranks as a response (or use them as needed)
    return jsonify({"weights": weights.tolist(), "rc": rc, "ranks": ranks})


# TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
@app.route('/topsis', methods=['POST'])
def topsis():
    try:
        # Get the input data from the request
        criteria = request.json

        # Extract criteria and weights
        column_order = list(criteria.keys())  # Column order based on criteria
        weights = np.array([criteria[col] for col in column_order])  # Convert weights to numpy array

        # Get the decision matrix from your DataFrame
        decision_matrix = df[column_order].to_numpy()  # Matrix with columns in the correct order

        # Define the types of criteria (1 for benefit, -1 for cost)
        types = np.array(['max' if col not in ['debt_to_equity_ratio'] else 'min' for col in column_order])

        # Run the TOPSIS method
        relative_closeness = topsis_method(decision_matrix, weights, types, graph=False, verbose=False)

        # Zip relative_closeness with df['name']
        ranked_data = list(zip(df['Name'], relative_closeness))

        # Sort by relative_closeness (descending order, since higher is better)
        ranked_data.sort(key=lambda x: x[1], reverse=True)

        # Create ranks based on sorted data
        ranks = sorted(
            [
                {
                    "rank": rank + 1,
                    "name": name,
                    "relative_closeness": closeness
                }
                for rank, (name, closeness) in enumerate(ranked_data)
            ],
            key=lambda x: x["rank"]
        )
        # Return the ranks as a response (or use them as needed)
        return jsonify(ranks)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# PROMETHEE (Preference Ranking Organization Method for Enrichment Evaluations)
@app.route('/promethee', methods=['POST'])
def promethee():
    criteria = request.json.get("weights")
    preference_function = request.json.get("preference_function")

    # Extract criteria and weights
    column_order = list(criteria.keys())  # Column order based on criteria
    weights = np.array([criteria[col] for col in column_order])  # Convert weights to numpy array

    # Get the decision matrix from your DataFrame
    decision_matrix = df[column_order].to_numpy()  # Matrix with columns in the correct order

    # Define the types of criteria (1 for benefit, -1 for cost)
    types = ['max' if col not in ['debt_to_equity_ratio'] else 'min' for col in column_order]

    normalized_matrix = normalize_criteria(decision_matrix, types)

    Q = [0.1 for i in range(len(column_order))]
    S = [0.3 for i in range(len(column_order))]
    P = [0.6 for i in range(len(column_order))]
    F = [preference_function for i in range(len(column_order))]

    p2 = promethee_ii(normalized_matrix, W=weights, Q=Q, S=S, P=P, F=F, sort=False, graph=False, verbose=True)

    ranked_data = list(zip(df['Name'],  p2[:, 1]))

    # Sort by relative_closeness (descending order, since higher is better)
    ranked_data.sort(key=lambda x: x[1], reverse=True)

    # Create ranks based on sorted data
    ranks = sorted(
        [
            {
                "rank": rank + 1,
                "name": name,
                "relative_closeness": closeness
            }
            for rank, (name, closeness) in enumerate(ranked_data)
        ],
        key=lambda x: x["rank"]
    )
    # Return the ranks as a response (or use them as needed)
    return jsonify(ranks)

    # num_alternatives = matrix.shape[0]
    # positive_flow = np.zeros(num_alternatives)
    # negative_flow = np.zeros(num_alternatives)
    #
    # for i in range(num_alternatives):
    #     for j in range(num_alternatives):
    #         if i != j:
    #             preference = np.maximum(matrix[i] - matrix[j], 0)
    #             positive_flow[i] += np.sum(preference * weights)
    #             negative_flow[j] += np.sum(preference * weights)
    #
    # net_flows = positive_flow - negative_flow


# WSM (Weighted Sum Model)
@app.route('/waspas', methods=['POST'])
def waspas_methods():
    criteria = request.json  # Extract criteria and weights from the request body

    # Ensure `criteria` is provided and has valid data
    if not criteria or not isinstance(criteria, dict):
        return jsonify({"error": "Invalid input data"}), 400

    # Extract column order and weights
    column_order = list(criteria.keys())  # Column order based on criteria
    weights = np.array([criteria[col] for col in column_order])  # Convert weights to numpy array

    # Ensure `weights` and `column_order` are consistent
    if len(weights) != len(column_order):
        return jsonify({"error": "Mismatch between criteria and weights"}), 400

    # Get the decision matrix from your DataFrame
    decision_matrix = df[column_order].to_numpy()  # Matrix with columns in the correct order

    # Define the types of criteria (1 for benefit, -1 for cost)
    types = np.array(['max' if col not in ['debt_to_equity_ratio'] else 'min' for col in column_order])

    # Use the `waspas_method` to calculate scores (assuming it's implemented and imported)
    wsm, wpm, waspas = waspas_method(decision_matrix, types, weights, 0.5, graph=False)

    # Combine scores into ranked data
    ranked_data = [
        {"name": name, "wsm_distance": wsm_score, "wpm_distance": wpm_score, "waspas_distance": waspas_score}
        for name, wsm_score, wpm_score, waspas_score in zip(df['Name'], wsm, wpm, waspas)
    ]

    # Create separate ranks for WSM, WPM, and WASPAS
    ranked_wsm = sorted(ranked_data, key=lambda x: x["wsm_distance"], reverse=True)
    ranked_wpm = sorted(ranked_data, key=lambda x: x["wpm_distance"], reverse=True)
    ranked_waspas = sorted(ranked_data, key=lambda x: x["waspas_distance"], reverse=True)

    # Add ranks to the data for each method
    for rank, item in enumerate(ranked_wsm, start=1):
        item["wsm_rank"] = rank
    for rank, item in enumerate(ranked_wpm, start=1):
        item["wpm_rank"] = rank
    for rank, item in enumerate(ranked_waspas, start=1):
        item["waspas_rank"] = rank

        # Sort the final ranked data by WASPAS ranks
    ranked_data_sorted_by_waspas = sorted(ranked_data, key=lambda x: x["waspas_rank"])

    # Return the final ranked data sorted by WASPAS rank as JSON response
    return jsonify(ranked_data_sorted_by_waspas)


@app.route('/data', methods=['GET'])
def get_data():
    try:
        # Fetch all company documents from MongoDB
        companies = list(collection.find())

        # If there are no companies in the database
        if not companies:
            return jsonify({"error": "No data available"}), 404

        # Extract the column names dynamically from the first company document
        first_company = companies[0]
        criteria = [key for key in first_company.keys() if
                    key != "_id" and key != "name"]  # Exclude '_id' and 'name' columns

        # Initialize lists to hold criteria values
        names = []
        data_matrix = []

        # Loop through the MongoDB documents and extract values for each criterion
        for company in companies:
            names.append(company.get("Name"))
            row = []
            for criterion in criteria:
                row.append(company.get(criterion))  # Add each criterion value for this company
            data_matrix.append(row)

        # Convert the matrix into a NumPy array (optional)
        data_matrix_np = np.array(data_matrix)

        # Convert the matrix into a list of lists (for JSON response)
        matrix_list = data_matrix_np.tolist()

        # Return the matrix data as JSON, along with the company names
        return jsonify({
            "names": names,  # List of company names
            "criteria": criteria,  # List of criteria names
            "data_matrix": matrix_list  # Matrix with company data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/ensemble', methods=['POST'])
def ensemble_method():
    try:
        # Pridobimo uteži kriterijev in metode
        criteria = request.json.get("weights")
        preference_function = request.json.get("preference_function")

        # Extract criteria and weights
        column_order = list(criteria.keys())  # Column order based on criteria
        weights = np.array([criteria[col] for col in column_order])  # Convert weights to numpy array

        # Get the decision matrix from your DataFrame
        decision_matrix = df[column_order].to_numpy()  # Matrix with columns in the correct order

        # Define the types of criteria (1 for benefit, -1 for cost)
        types = np.array(['max' if col not in ['debt_to_equity_ratio'] else 'min' for col in column_order])

        # Run individual methods
        topsis_response = topsis_method(decision_matrix, weights, types, graph=False, verbose=False)

        normalized_matrix = normalize_criteria(decision_matrix, list(types))
        Q = [0.1 for _ in range(len(column_order))]
        S = [0.3 for _ in range(len(column_order))]
        P = [0.6 for _ in range(len(column_order))]
        F = [preference_function for _ in range(len(column_order))]
        promethee_response = promethee_ii(normalized_matrix, W=weights, Q=Q, S=S, P=P, F=F, sort=False, graph=False)

        wsm, wpm, waspas_response = waspas_method(decision_matrix, types, weights, 0.5, graph=False)

        normalized_matrix_ahp = normalize_criteria_ahp(df[column_order].to_numpy(), list(types))
        prioritetna_matrika = np.zeros((len(df), len(column_order)))
        for i in range(len(column_order)):
            prioritetna_matrika[:, i] = build_comparison_matrix(normalized_matrix_ahp, i)
        ahp_response = np.dot(prioritetna_matrika, weights)

        # Convert AHP response into rankable data
        ahp_ranks =[{"name": name, "rank": rank + 1} for rank, (name, _) in enumerate(
            sorted(zip(df['Name'], topsis_response), key=lambda x: x[1], reverse=True)
        )]

        # Convert other methods' results into rankable data
        topsis_ranks = [{"name": name, "rank": rank + 1} for rank, (name, _) in enumerate(
            sorted(zip(df['Name'], topsis_response), key=lambda x: x[1], reverse=True)
        )]

        promethee_ranks = [{"name": name, "rank": rank + 1} for rank, (name, _) in enumerate(
            sorted(zip(df['Name'], promethee_response[:, 1]), key=lambda x: x[1], reverse=True)
        )]

        waspas_ranks = [{"name": name, "rank": rank + 1} for rank, (name, score) in enumerate(
            sorted(zip(df['Name'], waspas_response), key=lambda x: x[1], reverse=True)
        )]

        # Combine ranks for each company
        # Combine ranks for each company
        ensemble_ranks = {}
        for method_ranks, method_name in zip([ahp_ranks, topsis_ranks, promethee_ranks, waspas_ranks],
                                             ["ahp_rank", "topsis_rank", "promethee_rank", "waspas_rank"]):
            for rank in method_ranks:
                if rank["name"] not in ensemble_ranks:
                    ensemble_ranks[rank["name"]] = {
                        "total_rank": 0,  # Initialize total rank
                        "ahp_rank": None,
                        "topsis_rank": None,
                        "promethee_rank": None,
                        "waspas_rank": None
                    }
                # Assign rank to the appropriate key
                ensemble_ranks[rank["name"]][method_name] = rank["rank"]
                ensemble_ranks[rank["name"]]["total_rank"] += rank["rank"]

        # Convert combined ranks into sorted data
        final_scores = sorted(
            [{"name": name, "total_rank": int(total_rank["total_rank"]),
              "ahp_rank": total_rank["ahp_rank"],
              "topsis_rank": total_rank["topsis_rank"],
              "promethee_rank": total_rank["promethee_rank"],
              "waspas_rank": total_rank["waspas_rank"]}
             for name, total_rank in ensemble_ranks.items()],
            key=lambda x: x["total_rank"]
        )

        # Assign final ranks
        for rank, entry in enumerate(final_scores, start=1):
            entry["final_rank"] = rank

        # Return the name, final rank, and summed ranks along with individual method ranks
        return jsonify([
            {"name": entry["name"],
             "final_rank": entry["final_rank"],
             "sum_of_ranks": entry["total_rank"],
             "ahp_rank": entry["ahp_rank"],
             "topsis_rank": entry["topsis_rank"],
             "promethee_rank": entry["promethee_rank"],
             "waspas_rank": entry["waspas_rank"]}
            for entry in final_scores
        ])

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

