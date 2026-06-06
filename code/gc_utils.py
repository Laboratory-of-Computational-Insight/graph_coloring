import torch
from sklearn.cluster import KMeans
import networkx as nx
import picos
from operator import itemgetter
from scipy.linalg import eigh
import cvxpy as cp
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px


def is_k_color(adj, k_assignment, print_bad=False):
    """
    Check if the graph is a valid k-coloring.
    """
    disagreement_count = 0
    adj_clone = adj.clone().detach().cpu().numpy()
    n = adj_clone.shape[0]
    k_assignment += 1
    # for i in range(n):
    #     adj_clone[i, adj_clone[i] > 0] = k_assignment[i] + 1
    bad_indexes = []
    for i in range(n):
        neighbors_assignment = k_assignment[adj_clone[i]!=0]
        bad_count = (neighbors_assignment == k_assignment[i]).sum()
        if bad_count > 0:
            bad_indexes.append(i)
            if print_bad:
                print(i, bad_count, np.arange(n)[adj_clone[i] != 0][neighbors_assignment == k_assignment[i]])
            # disagreement_count += bad_count
            disagreement_count += 1
    return disagreement_count == 0, disagreement_count, bad_indexes

def sklearn_k_means(X, k, max_iters=100, centroids = None):
    """
    Perform k-means clustering on a dataset using scikit-learn and reorder the clusters based on PCA.

    Parameters:
    - X: A torch tensor of shape (n, d) where n is the number of data points and d is the dimensionality.
    - k: The number of clusters.
    - max_iters: Maximum number of iterations to run the algorithm.

    Returns:
    - cluster_assignments: An array of reordered cluster assignments for each data point.
    - centroids: The reordered centroids of the clusters.
    """
    # Ensure X is a numpy array for sklearn compatibility
    X_np = X.clone().detach().cpu().numpy() if type(X) == torch.Tensor else X
    pca = PCA(n_components=2)
    pca.fit(X_np)

    if centroids is not None:
        distances = np.linalg.norm(X_np[:, np.newaxis] - centroids, axis=2)
        assignments = np.argmin(distances, axis=1)
        return assignments, centroids


    X_np_unique = np.unique(X_np, axis=0)
    k = min(k, X_np_unique.shape[0])
    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=k, max_iter=max_iters)
    kmeans.fit(X_np)

    # Retrieve the cluster assignments and centroids
    cluster_assignments = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Apply PCA to reduce centroids to 2 dimensions
    centroids_2d = (centroids - pca.mean_) @ pca.components_.T

    # Identify the most right centroid
    most_right_index = np.argmax(centroids_2d[:, 0])

    # Calculate angles for the remaining centroids
    angles = np.arctan2(centroids_2d[:, 1], centroids_2d[:, 0])
    angles = (angles - angles[most_right_index]) % (2 * np.pi)

    # Sort indices based on angles in a clockwise manner
    sorted_indices = [most_right_index] + sorted(
        [i for i in range(k) if i != most_right_index],
        key=lambda i: angles[i]
    )

    # Create a mapping from old cluster indices to new ones
    new_cluster_map = {old: new for new, old in enumerate(sorted_indices)}

    # Reassign cluster labels based on the sorted centroids
    new_cluster_assignments = [new_cluster_map[label] for label in cluster_assignments]
    new_cluster_assignments = np.array(new_cluster_assignments)
    new_centroids = centroids[sorted_indices]

    return new_cluster_assignments, new_centroids

def solve_max_k_cut_sdp(G: nx.Graph, k: int, weight: str = "weight"):
    """
    solve the sdp problem with object of max-3-cut and G as an input.
    Frieze & Jerrum: https://www.math.cmu.edu/~af1p/Texfiles/cuts.pdf
    :param G: undirected graph
    :return: embedding
    """

    num_nodes = G.number_of_nodes()
    if num_nodes <= 1:
        return None

    sum_edges_weight = sum(map(itemgetter(weight), map(itemgetter(2), G.edges(data=True))))
    if sum_edges_weight == 0:
        return None

    maxcut = picos.Problem()
    # print(f'num nodes: {num_nodes}')
    # Add the symmetric matrix variable.
    X = maxcut.add_variable('X', (num_nodes, num_nodes), 'symmetric')

    # Retrieve the Laplacian of the graph.
    LL = ((k - 1) / (2 * k)) * nx.laplacian_matrix(G, weight=weight).todense()
    L = picos.new_param('L', LL)

    # Constrain X to have ones on the diagonal.
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                maxcut.add_constraint(X[i, j] >= (-1 / (k - 1)))
    # maxcut.add_constraint(X >= (-1 / (k - 1)))

    maxcut.add_constraint(picos.diag_vect(X) == 1)

    # Constrain X to be positive semidefinite.
    maxcut.add_constraint(X >> 0)

    # Set the objective.
    maxcut.set_objective('max', L | X)

    # Solve the problem.
    maxcut.solve(verbose=0, solver='cvxopt')
    indexed_nodes = list(G.nodes)

    ### Perform the random relaxation
    # Use a fixed RNG seed so the result is reproducable.
    # cvx.setseed(1919)

    # Perform a Cholesky factorization (in the lower triangular part of the matrix)
    # https://en.wikipedia.org/wiki/Cholesky_decomposition#Proof_for_positive_semi-definite_matrices
    # https://en.wikipedia.org/wiki/Square_root_of_a_matrix
    # https://math.stackexchange.com/questions/1801403/decomposition-of-a-positive-semidefinite-matrix
    # https://en.wikipedia.org/wiki/Matrix_decomposition#Cholesky_decomposition
    # https://stackoverflow.com/questions/5563743/check-for-positive-definiteness-or-positive-semidefiniteness
    # https://en.wikipedia.org/wiki/Matrix_decomposition#Cholesky_decomposition:~:text=Cholesky%20decomposition%5Bedit%5D
    # https://proceedings.neurips.cc/paper/2021/file/45c166d697d65080d54501403b433256-Paper.pdf
    D, V = eigh(X.value)
    # Z = (V * np.sqrt(D)) @ V.T
    s = np.diag(np.abs(D))
    z = np.sqrt(s)
    # V = np.linalg.cholesky(X.value)
    V = V @ z
    return maxcut.value, {indexed_nodes[i]: np.expand_dims(np.array(V[i, :]), axis=0) for i in range(num_nodes)}

def generate_random_unit_vector(dim, num_of_vectors):
    random_unit_vectors = []
    for i in range(num_of_vectors):
        v = np.random.rand(dim)
        random_unit_vectors.append(v / (v ** 2).sum() ** 0.5)
    return random_unit_vectors

def classify_nodes(G: nx.Graph, nodes_embeddings, k):
    unit_vectors = generate_random_unit_vector(G.number_of_nodes(), k)
    result = {}
    indexed_nodes = list(G.nodes)
    for node in indexed_nodes:
        classify = 0
        min_distance = 1000000
        for i, vector in enumerate(unit_vectors):
            distance = np.linalg.norm(nodes_embeddings[node] - vector)
            if distance < min_distance:
                classify = i
                min_distance = distance
        result[node] = classify
    return result

def run_max_k_cut(G: nx.Graph, k):
    value, node_id_to_embedding = solve_max_k_cut_sdp(G, k)
    node_id_to_classification = classify_nodes(G, node_id_to_embedding, k)
    return value, node_id_to_embedding, node_id_to_classification

def find_least_common_neighbor_color(adj, assignment, feautres, k=3):
    n = adj.shape[0]
    color_counts = np.zeros((n, k), dtype=int)  # To store counts of neighbors for each color
    conflict_sums = np.zeros(n, dtype=int)  # To store the sum of conflicts for other colors
    neighbor_dist = np.zeros(n)
    degrees = adj.sum(axis=1)
    conflict_avg = np.zeros(n)
    conflict_max = np.zeros(n)
    conflict_min = np.zeros(n)

    for i in range(n):
        neighbors = torch.where(adj[i] == 1)[0]
        for neighbor in neighbors:
            color_counts[i][assignment[neighbor] - 1] += 1  # -1 to convert to 0-based index
            neighbor_dist[i] += np.linalg.norm(feautres[i] - feautres[neighbor])
        if len(neighbors)>0:neighbor_dist[i] = neighbor_dist[i] / len(neighbors)

    least_common_colors = np.zeros(n, dtype=int)
    for i in range(n):
        current_color_index = assignment[i] - 1  # Convert to 0-based index
        min_count = n+100 #color_counts[i][current_color_index]
        least_common_color_index = current_color_index

        for j in range(k):
            if j==current_color_index:
                continue
            if color_counts[i][j] < min_count or (color_counts[i][j] == min_count and j == current_color_index):
                min_count = color_counts[i][j]
                least_common_color_index = j

        least_common_colors[i] = least_common_color_index + 1  # Convert back to 1-based index
        conflict_sums[i] = np.sum(color_counts[i]) - color_counts[i][least_common_color_index]
        conflict_avg[i] = conflict_sums[i]/degrees[i] if degrees[i] > 0 else 0
        conflict_max[i] = max(color_counts[i][[j for j in range(k) if j != current_color_index]])
        conflict_min[i] = min(color_counts[i][[j for j in range(k) if j != current_color_index]])

    return least_common_colors, [conflict_sums, conflict_avg, conflict_max, conflict_min], neighbor_dist


def calc_dist_in_iteratoin(adj, features):
    n = adj.shape[0]
    dist = np.zeros(n)
    for i in range(n):
        neighbors = torch.where(adj[i] == 1)[0]
        dist[i] = np.mean(np.linalg.norm(features[i] - features[neighbors]))
    return dist

def calc_dist_over_iteartions(adj, features_iteartions):
    dist = []
    for features in features_iteartions:
        dist.append(calc_dist_in_iteratoin(adj, features))
    return dist

def confidance_pair(adj, local_assignments, distances_over_time, closesed_index, supports):
    local_assignments = local_assignments[:closesed_index+1]
    distances_over_time = distances_over_time[:closesed_index+1]
    n = adj.shape[0]
    percentile_range = [10, 20, 30, 50, 70, 80, 90]


    below_pairs = {(percentile_range[p1],percentile_range[p2]):0 for p1 in range(len(percentile_range)) for p2 in range(p1, len(percentile_range))}
    above_pairs = {(percentile_range[p1],percentile_range[p2]):0 for p1 in range(len(percentile_range)) for p2 in range(p1, len(percentile_range))}

    below_pairs_count = {(percentile_range[p1],percentile_range[p2]):0 for p1 in range(len(percentile_range)) for p2 in range(p1, len(percentile_range))}
    above_pairs_count = {(percentile_range[p1],percentile_range[p2]):0 for p1 in range(len(percentile_range)) for p2 in range(p1, len(percentile_range))}

    nieghboring_nodes = [(i,j) for i in range(n) for j in range(i+1, n) if adj[i,j]==1]

    for iteration, (local_assignment, distance_over_time) in enumerate(zip(local_assignments, distances_over_time)):
        if iteration <= 0.2*closesed_index: continue
        support = supports[iteration][-1]

        current_percentiles = np.array([np.percentile(distance_over_time, p) for p in percentile_range])
        current_support_percentiles = np.array([np.percentile(support, p) for p in percentile_range])

        for u,v in nieghboring_nodes:
            # du = percentileofscore(distance_over_time, distance_over_time[u])
            du = distance_over_time[u]
            # dv = percentileofscore(distance_over_time, distance_over_time[v])
            dv = distance_over_time[v]
            # su = percentileofscore(support, support[u])
            su = support[u]
            # sv = percentileofscore(support, support[v])
            sv = support[v]

            for p in range(len(percentile_range)):
                if du <= current_percentiles[p] and su <= current_support_percentiles[p] and dv <= current_percentiles[p] and sv <= current_support_percentiles[p]:
                    below_pairs_count[(percentile_range[p],percentile_range[p])] += 1
                    if local_assignment[u] == local_assignment[v]:
                        below_pairs[(percentile_range[p],percentile_range[p])] += 1
                if du >= current_percentiles[p] and su<= current_support_percentiles[p] and dv >= current_percentiles[p] and sv <= current_support_percentiles[p]:
                    above_pairs_count[(percentile_range[p], percentile_range[p])] += 1
                    if local_assignment[u] == local_assignment[v]:
                        above_pairs[(percentile_range[p],percentile_range[p])] += 1

    above= {
        str(k): v/above_pairs_count[k] if above_pairs_count[k] > 0 else -1 for k,v in above_pairs.items()
    }
    below={
        str(k): v/below_pairs_count[k] if below_pairs_count[k] > 0 else -1 for k,v in below_pairs.items()
    }

    return above, below

def max_clique_sdp():

    # Function to generate a planted clique graph G(n, p, k) as a numpy adjacency matrix
    def generate_planted_clique(n, p, k):
        # Generate a random adjacency matrix for G(n, p)
        A = (np.random.rand(n, n) < p).astype(float)
        np.fill_diagonal(A, 0)  # No self-loops

        # Add a planted clique of size k
        clique_nodes = np.random.choice(n, k, replace=False)
        for i in clique_nodes:
            for j in clique_nodes:
                if i != j:
                    A[i, j] = 1
                    A[j, i] = 1  # Ensure symmetry

        return A, clique_nodes

    # Function to solve the SDP relaxation of the max clique problem
    def solve_sdp_max_clique(A):
        n = A.shape[0]

        # Define the SDP problem
        X = cp.Variable((n, n), symmetric=True)

        # Objective: Maximize the trace of X (trace maximization helps find the largest submatrix)
        # objective = cp.Maximize(cp.trace(X))
        objective = cp.Maximize(cp.sum(X))

        # Constraints
        constraints = [
                          X >> 0,  # X must be positive semi-definite
                          cp.trace(X) == 1  # Diagonal elements must be 1 (each node has to correlate with itself)
                      ] + [X[i, j] == 0 for i in range(n) for j in range(n) if A[i, j] == 0 and i != j
                           # Non-edges have zero correlation
                           ]

        # Solve the SDP
        prob = cp.Problem(objective, constraints)
        prob.solve()
        X_value = X.value
        print(X_value)
        eigvals, eigvecs = np.linalg.eigh(X_value)

        # Find the largest eigenvalue and corresponding eigenvector
        idx = np.argmax(eigvals)
        v = eigvecs[:, idx]

        # Recover the clique by selecting the nodes with large projections
        threshold = 1e-3
        clique_nodes = np.where(v > threshold)[0]

        # Return the solution matrix X
        return X_value, len(clique_nodes), clique_nodes, v

    # PCA and Plotting
    def plot_sdp_solution(X, planted, found):
        # Apply PCA to the SDP solution matrix
        pca = PCA(n_components=2)
        pos = pca.fit_transform(X)

        # Plot using Plotly
        fig = px.scatter(x=pos[:, 0], y=pos[:, 1], title='2D Embedding of SDP Solution color planted cliques',
                         color=["0" if i in planted else "1" for i in range(n)],
                         labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
                         size_max=10)

        # Show the plot
        fig.show()
        fig = px.scatter(x=pos[:, 0], y=pos[:, 1], title='2D Embedding of SDP Solution color found cliques',
                         color=["0" if i in found else "1" for i in range(n)],
                         labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
                         size_max=10)

        # Show the plot
        fig.show()

    n = 50  # Number of nodes
    p = 0.3  # Probability for edge creation
    k = 10  # Size of the planted clique

    # Generate a planted clique graph as an adjacency matrix
    A, planted_clique = generate_planted_clique(n, p, k)

    # Solve for maximum clique using SDP
    X, clique_size, clique_nodes, v = solve_sdp_max_clique(A)

    print(f"SDP Estimated Clique Size: {clique_size}")
    print(f"SDP Estimated Clique Nodes: {clique_nodes}, {planted_clique}")
    print(f"number of non zeros: {sum(X.sum(0) != 0)}")
    print("v", v)

    # Visualize the graph in 2D PCA space
    plot_sdp_solution(X, planted_clique, clique_nodes)

