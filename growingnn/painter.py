#from turtle import color
from pyvis.network import Network
from .helpers import *

DEBUG = False
def draw(model, path = 'mygraph1.html'):
    net = Network(directed =True)
    edges = []
    #for key in model.input_receptors:
    if DEBUG: print("1 + ")
    for input_layer in model.input_layers:
        edges += draw_rec(model, input_layer.id, None, net, [])
    # for layer in model.input_layers:
    #     edges += draw_rec(model, layer.id, None, net, [])
    if DEBUG: print("2 + ")
    edges = delete_repetitions(edges)
    if DEBUG: print("3 + ")
    for edge in edges:
        net.add_edge(edge[0],edge[1])
    if DEBUG: print("4 + ")
    #net.save_graph(path)
    #net.write_html(path, notebook=False)
    website = net.generate_html(path, False, False)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(website)

def draw_rec(model, layer_id, parent_id, net, edges, deepth = 0):
    deepth += 1
    if DEBUG: 
        if deepth > 20:
            print("cycle detected ? ")
            print("layer_id: ", layer_id)
            print("parent_id: ", parent_id)
            return edges
    layer = model.get_layer(layer_id)
    if DEBUG: print("draw_rec 1")
#    net.add_node(layer_id, label=str(layer_id) + "[" +str(layer.input_size)+","+str(layer.neurons)+"]")
    net.add_node(layer_id, layer.get_paint_label())
    if DEBUG: print("draw_rec 2")
    if parent_id!=None:
        edges.append([parent_id, layer_id])
    if DEBUG: print("draw_rec 3")
    for child_layer_id in layer.output_layers_ids:
        edges += draw_rec(model, child_layer_id, layer_id, net,[], deepth)
    if DEBUG: print("draw_rec 4")
    return edges


