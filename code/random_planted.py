import numpy as np
import torch


def create_random_k(n, k, p=0.5):
    assignment = torch.randint(0,k, size=(n,))
    assignment += 1
    # Step 1: Create a comparison matrix
    # Expands assignment to (n, 1) and (1, n), then compares them to generate a boolean matrix where True indicates different groups
    comparison_matrix = assignment.unsqueeze(1) != assignment.unsqueeze(0)

    # Step 2: Generate random edges
    # random_edges = torch.randint(0, 2, (n, n), dtype=torch.int)
    random_edges = torch.bernoulli(torch.full((n, n), p)).int()

    # Step 3: Apply the group constraint
    # Uses the comparison matrix as a mask to zero out edges within the same group
    adj_matrix = random_edges * comparison_matrix.int()

    # Step 4: Ensure symmetry
    # Averages the matrix with its transpose and rounds to ensure the matrix is symmetric
    adj_matrix = ((adj_matrix + adj_matrix.t()) / 2).round().int()

    # Zero out the diagonal to avoid self-loops
    torch.Tensor.fill_diagonal_(adj_matrix, 0)

    return assignment, adj_matrix

def create_gnp(n, k, c=1.0, p=None):
    p = c/n if p is None else p
    same = []
    while len(same) == 0:
        upper_triangular = torch.bernoulli(torch.full((n, n), p)).triu(diagonal=1)
        adj_matrix = upper_triangular + upper_triangular.t()
        torch.Tensor.fill_diagonal_(adj_matrix, 0)
        # assignment = find_k_coloring(adj_matrix, k)
        assignment = [-1]*(n-1)

        bad_adj_matrix = adj_matrix.clone()
        n_array = torch.arange(0,len(assignment))
        same = np.array([0,0])
        # for i in range(1, k+1):
        #     same = n_array[assignment==i]
        #     if len(same) > 1:
        #         break
        #     else:
        #         same = []
        # if len(same) > 1:
        #     bad_adj_matrix[same[0], same[1]] = bad_adj_matrix[same[1], same[0]] = 1
        #     break
    return assignment, adj_matrix, bad_adj_matrix, tuple(same[:2].tolist())

def create_planted(n=1000, k=3, p=0.5):
    same = []
    while len(same) == 0:
        assignment, adj_matrix = create_random_k(n, k,p=p)

        bad_adj_matrix = adj_matrix.clone()
        n_array = torch.arange(0,len(assignment))
        for i in range(1, k+1):
            same = n_array[assignment==i]
            if len(same) > 1:
                break
            else:
                same = []
        if len(same) > 1:
            bad_adj_matrix[same[0], same[1]] = bad_adj_matrix[same[1], same[0]] = 1
            break
    return assignment, adj_matrix, bad_adj_matrix, tuple(same[:2].tolist())


def create_one_side(n=1000, k=3, p=0.5):
    same = []
    while len(same) == 0:
        assignment, adj_matrix = create_random_k(n, k, p=p)

        color = assignment[0]
        same_color_nodes = torch.where(assignment == color)[0]
        different_color_nodes = torch.where(assignment != color)[0]
        #connect all same color nodes to all different color nodes
        adj_matrix[same_color_nodes, :] = 0
        adj_matrix[:, same_color_nodes] = 0
        same_color_nodes, different_color_nodes = torch.meshgrid(same_color_nodes, different_color_nodes, indexing='ij')
        adj_matrix[same_color_nodes, different_color_nodes] = 1
        adj_matrix[different_color_nodes, same_color_nodes] = 1


        bad_adj_matrix = adj_matrix.clone()
        n_array = torch.arange(0,len(assignment))
        for i in range(1, k+1):
            same = n_array[assignment==i]
            if len(same) > 1:
                break
            else:
                same = []
        if len(same) > 1:
            bad_adj_matrix[same[0], same[1]] = bad_adj_matrix[same[1], same[0]] = 1
            break
    return assignment, adj_matrix, bad_adj_matrix, tuple(same[:2].tolist())


def create_planted_with_one_side_single(n=1000, k=3):
    same = []
    while len(same) == 0:
        assignment, adj_matrix = create_random_k(n, k)

        adj_matrix[0, :] = 0
        adj_matrix[:, 0] = 0

        adj_matrix[1, :] = 0
        adj_matrix[:, 1] = 0
        adj_matrix[2, :] = 0
        adj_matrix[:, 2] = 0
        assignment[0] = torch.randint(1, k+1, (1,)).item()

        color_of_node_0 = assignment[0]
        same_color_nodes = torch.where(assignment == color_of_node_0)[0]
        num_nodes_to_connect = len(same_color_nodes)
        # nodes_to_connect = same_color_nodes[:num_nodes_to_connect]
        nodes_to_connect = same_color_nodes[:8]
        adj_matrix[0, nodes_to_connect] = 1
        adj_matrix[nodes_to_connect, 0] = 1
        # nodes_to_connect = same_color_nodes[:num_nodes_to_connect//2]
        nodes_to_connect = same_color_nodes[:7]
        adj_matrix[1, nodes_to_connect] = 1
        adj_matrix[nodes_to_connect, 1] = 1
        nodes_to_connect = same_color_nodes[:4]
        adj_matrix[2, nodes_to_connect] = 1
        adj_matrix[nodes_to_connect, 2] = 1

        assignment[0] = ((assignment[0] + 1) % k) + 1
        color_of_node_0 = assignment[0]
        same_color_nodes = torch.where(assignment == color_of_node_0)[0]
        # num_nodes_to_connect = min(len(same_color_nodes), 8)
        num_nodes_to_connect = len(same_color_nodes)
        # nodes_to_connect = same_color_nodes[:num_nodes_to_connect//2]
        nodes_to_connect = same_color_nodes[:1]
        adj_matrix[1, nodes_to_connect] = 1
        adj_matrix[nodes_to_connect, 1] = 1
        nodes_to_connect = same_color_nodes[:4]
        adj_matrix[2, nodes_to_connect] = 1
        adj_matrix[nodes_to_connect, 2] = 1

        assignment[0] = ((assignment[0] + 1) % k) + 1
        assignment[1] = assignment[0]
        assignment[2] = assignment[0]

        adj_matrix[0, 0] = adj_matrix[1, 1] = adj_matrix[2, 2] = adj_matrix[1, 0] = adj_matrix[0, 1] = 0
        adj_matrix[2, 0] = adj_matrix[0, 2] = adj_matrix[2, 1] = adj_matrix[1, 2] = 0


        bad_adj_matrix = adj_matrix.clone()
        n_array = torch.arange(0,len(assignment))
        for i in range(1, k+1):
            same = n_array[assignment==i]
            if len(same) > 1:
                break
            else:
                same = []
        if len(same) > 1:
            bad_adj_matrix[same[0], same[1]] = bad_adj_matrix[same[1], same[0]] = 1
            break
    return assignment, adj_matrix, bad_adj_matrix, tuple(same[:2].tolist())


def create_planted_with_one_side(n=1000, k=3):
    same = []
    while len(same) == 0:
        assignment, adj_matrix = create_random_k(n, k)

        assignment[2] = -1
        adj_matrix[2, :] = 0
        adj_matrix[:, 2] = 0

        adj_matrix[0, :] = 0
        adj_matrix[:, 0] = 0
        assignment[0] = torch.randint(1, k+1, (1,)).item()
        color_of_node_0 = assignment[0]
        same_color_nodes = torch.where(assignment == color_of_node_0)[0]
        num_nodes_to_connect = min(6, len(same_color_nodes))
        nodes_to_connect = same_color_nodes[:num_nodes_to_connect]
        adj_matrix[0, nodes_to_connect] = 1
        adj_matrix[nodes_to_connect, 0] = 1
        assignment[0] = ((assignment[0] + 1) % k) + 1
        assignment[2] = assignment[0]
        adj_matrix[2, nodes_to_connect[1:]] = 1
        adj_matrix[nodes_to_connect[1:], 2] = 1

        adj_matrix[1, :] = 0
        adj_matrix[:, 1] = 0
        same_color_nodes1 = torch.where(assignment == assignment[0])[0]
        num_nodes_to_connect1 = min(6, len(same_color_nodes1))
        nodes_to_connect1 = same_color_nodes1[:num_nodes_to_connect1]
        adj_matrix[1, nodes_to_connect1] = 1
        adj_matrix[nodes_to_connect1, 1] = 1
        adj_matrix[1, nodes_to_connect] = 1
        adj_matrix[nodes_to_connect, 1] = 1
        adj_matrix[0, 0] = 0
        adj_matrix[1, 1] = 0
        adj_matrix[2, 2] = 0
        assignment[1] = ((assignment[0] + 1 )% k) + 1

        bad_adj_matrix = adj_matrix.clone()
        n_array = torch.arange(0,len(assignment))
        for i in range(1, k+1):
            same = n_array[assignment==i]
            if len(same) > 1:
                break
            else:
                same = []
        if len(same) > 1:
            bad_adj_matrix[same[0], same[1]] = bad_adj_matrix[same[1], same[0]] = 1
            break
    return assignment, adj_matrix, bad_adj_matrix, tuple(same[:2].tolist())



def create_planted_with_one_contradicting(n=1000, k=3):
    same = []
    while len(same) == 0:
        assignment, adj_matrix = create_random_k(n, k)

        adj_matrix[0, :] = 0
        adj_matrix[:, 0] = 0
        for i in range(1, k+1):
            same = torch.where(assignment == i)[0]
            if same[0] == 0:
                same= same[1:]
            if len(same) > 1:
                same = same[0:3]
                adj_matrix[0, same] = adj_matrix[same, 0] = 1


        bad_adj_matrix = adj_matrix.clone()
        n_array = torch.arange(0,len(assignment))
        for i in range(1, k+1):
            same = n_array[assignment==i]
            if len(same) > 1:
                break
            else:
                same = []
        if len(same) > 1:
            bad_adj_matrix[same[0], same[1]] = bad_adj_matrix[same[1], same[0]] = 1
            break
    return assignment, adj_matrix, bad_adj_matrix, tuple(same[:2].tolist())

