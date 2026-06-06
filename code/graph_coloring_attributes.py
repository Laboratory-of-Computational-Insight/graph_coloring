import numpy as np

def support_for_color(adj_matrix, color_assignment, color):
    """
    Calculate the support for a given color for all vertices.
    """
    n = len(adj_matrix)
    support = np.zeros(n, dtype=int)
    for v in range(n):
        if color_assignment[v] == -1:  # uncolored vertex
            if not any(adj_matrix[v, u] == 1 and color_assignment[u] == color for u in range(n)):
                support[v] = 1
    return support

def support_for_vertex_color_assignment(adj_matrix, color_assignment, color):
    """
    Calculate the support for assigning a color to each vertex.The support for assigning a particular color to a vertex could be defined as the number of neighboring vertices that would still have at least one valid color option remaining if this assignment were made.
    """
    n = len(adj_matrix)
    support = np.zeros(n, dtype=int)
    for vertex in range(n):
        for u in range(n):
            if adj_matrix[vertex, u] == 1 and color_assignment[u] == -1:
                if any(color_assignment[v] != color for v in range(n) if adj_matrix[u, v] == 1 and v != vertex):
                    support[vertex] += 1
    return support

def color_support_count(adj_matrix, color_assignment, num_colors=None):
    """
    Calculate the color support count for each vertex. Returns an array where each element is the number of available colors for the corresponding vertex.
    """
    if num_colors is None:
        num_colors = len(set([i for i in color_assignment]))
    n = len(adj_matrix)
    support = np.zeros(n, dtype=int)
    for vertex in range(n):
        used_colors = set(color_assignment[u] for u in range(n) if adj_matrix[vertex, u] == 1 and color_assignment[u] != -1)
        support[vertex] = num_colors - len(used_colors)  # Assuming n colors are available
    return support

def support_for_partial_coloring(adj_matrix, color_assignment):
    """
    Calculate the support for a partial coloring for each vertex.We could define the support for a partial coloring as the number of vertices that have been successfully colored without conflicts.
    """
    n = len(adj_matrix)
    support = np.zeros(n, dtype=int)
    for v in range(n):
        if color_assignment[v] != -1:
            if not any(adj_matrix[v, u] == 1 and color_assignment[u] == color_assignment[v] for u in range(n)):
                support[v] = 1
    return support
