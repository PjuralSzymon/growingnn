from pyvis.network import Network
from .helpers import *
import numpy as np

DEBUG = False
def draw(model, path = 'mygraph1.html'):
    net = Network(directed =True)
    edges = []
    if DEBUG: print("1 + ")
    for input_layer in model.input_layers:
        edges += draw_rec(model, input_layer.id, None, net, [])
    if DEBUG: print("2 + ")
    edges = delete_repetitions(edges)
    if DEBUG: print("3 + ")
    for edge in edges:
        net.add_edge(edge[0],edge[1])
    if DEBUG: print("4 + ")
    website = net.generate_html(path, False, False)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(website)

def draw_rec(model, layer_id, parent_id, net, edges, deepth = 0):
    deepth += 1
    layer = model.get_layer(layer_id)
        
    # Default node color (low E)
    node_color = "#a6dced"  # Light blue in hexadecimal

    # Check if the layer has the matrix E and calculate its mean
    if hasattr(layer, 'E') and layer.E is not None:
        E_min = 0 
        E_max = np.prod(layer.E.shape)
        E_sum = np.sum(np.abs(layer.E))
        
        # Normalize E_mean to a range of 0 (light blue) to 1 (red)
        normalized_E = min(max((E_sum - E_min) / (E_max - E_min), 0), 1)
        
        # Light blue RGB values
        base_red = 166
        base_green = 220
        base_blue = 237

        # Calculate interpolated color
        red = int(base_red + (255 - base_red) * normalized_E)
        green = int(base_green + (0 - base_green) * normalized_E)
        blue = int(base_blue + (0 - base_blue) * normalized_E)
        
        # Construct the hexadecimal color
        node_color = f"#{red:02X}{green:02X}{blue:02X}"


    net.add_node(layer_id, layer.get_paint_label(), color=node_color)
    if parent_id!=None:
        edges.append([parent_id, layer_id])
    for child_layer_id in layer.output_layers_ids:
        edges += draw_rec(model, child_layer_id, layer_id, net,[], deepth)
    return edges


