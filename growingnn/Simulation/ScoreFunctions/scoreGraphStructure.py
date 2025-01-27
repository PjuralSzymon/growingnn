import time
from ...structure import *
import networkx as nx

def to_networkx_graph(M):
    G = nx.DiGraph()
    for layer in M.input_layers + M.hidden_layers:
        G.add_node(layer.id)
        for input_layer_id in layer.input_layers_ids:
            G.add_edge(input_layer_id, layer.id)
        for output_layer_id in layer.output_layers_ids:
            G.add_edge(layer.id, output_layer_id)
    return G


def scoreDiameter(M, epochs, X_train, Y_train):
    # The smaller the normalized value, the smaller the graph's diameter, 
    # indicating nodes are more concentrated around a central point. Conversely, 
    # a larger value suggests a larger graph diameter, 
    # indicating nodes are more spread out.
    G = to_networkx_graph(M)
    max_distance = len(G.nodes()) - 1
    try:
        diameter = nx.diameter(G)
        normalized_diameter = diameter / max_distance
    except nx.NetworkXError:
        normalized_diameter = 1.0  # Graf niespójny, najgorszy przypadek
    return 1.0 - normalized_diameter

def scoreRadius(M, epochs, X_train, Y_train):
    # The smaller the normalized value, the smaller the graph's radius,
    # suggesting the graph is more centralized around a single node. 
    # A larger value indicates a larger graph radius, 
    # meaning the graph is more extensive.
    G = to_networkx_graph(M)
    max_distance = len(G.nodes()) - 1
    try:
        radius = nx.radius(G)
        normalized_radius = radius / max_distance
    except nx.NetworkXError:
        normalized_radius = 1.0  # Graf niespójny, najgorszy przypadek
    return 1.0 - normalized_radius

def scoreEccentricity(M, epochs, X_train, Y_train):
    # The smaller the normalized value, the smaller the eccentricity of nodes, 
    # indicating nodes are closer together and more integrated. 
    # A larger value suggests greater eccentricity of nodes, 
    # meaning nodes are more distant from each other.
    G = to_networkx_graph(M)
    max_distance = len(G.nodes()) - 1
    # Eccentricity
    try:
        eccentricity = nx.eccentricity(G)
        normalized_eccentricity = {n: e / max_distance for n, e in eccentricity.items()}
    except nx.NetworkXError:
        normalized_eccentricity = 0.0  # Graf niespójny, najgorszy przypadek
    return normalized_eccentricity


def scoreChromaticNumber(M, epochs, X_train, Y_train):
    # The smaller the normalized value, the greater the chromatic number, 
    # indicating the graph requires more distinct colors to color the graph without conflicts. 
    # Conversely, a larger value suggests a smaller chromatic number, 
    # meaning the graph requires fewer distinct colors.
    G = to_networkx_graph(M)
    chromatic_number = nx.coloring.greedy_color(G.to_undirected(), strategy='largest_first')
    max_chromatic_number = max(chromatic_number.values()) + 1
    normalized_chromatic_number = 1 / max_chromatic_number  
    return normalized_chromatic_number


def scoreMatchingNumber(M, epochs, X_train, Y_train):
    # The smaller the normalized value, the greater the matching number, 
    # indicating more edges of the graph can be paired without conflict. 
    # A larger value suggests a smaller matching number, 
    # meaning fewer edges can be paired.
    G = to_networkx_graph(M)
    matching_number = len(nx.max_weight_matching(G.to_undirected()))
    normalized_matching_number = 1 / matching_number if matching_number > 0 else 1.0  
    return normalized_matching_number

def scoreIndependenceNumber(M, epochs, X_train, Y_train):
    # The smaller the normalized value, the greater the independence number, 
    # indicating more vertices of the graph can be independent (not connected by edges).
    # A larger value suggests a smaller independence number, 
    # meaning fewer vertices can be independent.
    G = to_networkx_graph(M)
    independence_number = len(nx.maximal_independent_set(G.to_undirected()))
    normalized_independence_number = 1 / independence_number if independence_number > 0 else 1.0  
    return normalized_independence_number