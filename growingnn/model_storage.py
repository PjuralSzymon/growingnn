import numpy as np
import json
from .structure import *
from .helpers import *

def save_model(M, path):
    dict = {}
    dict['batch_size'] = M.batch_size
    dict['loss_function'] = loss_to_json(M.loss_function)
    dict['input_size'] = M.input_size
    dict['output_size'] = M.output_size 
    dict['hidden_size'] = M.hidden_size
    dict['avaible_id'] = M.avaible_id
    dict['activation_fun'] = activation_to_json(M.activation_fun)
    #dict['input_layer'] = layer_to_json(M.input_layer)
    dict['output_layer'] = layer_to_json(M.output_layer)
    dict['convolution'] = M.convolution
    dict['input_shape'] = M.input_shape
    dict['kernel_size'] = M.kernel_size
    dict['depth'] = M.depth
    dict['hidden_layers'] = []
    dict['input_layers'] = []
    for layer in M.input_layers:
        dict['input_layers'].append(layer_to_json(layer))
    for layer in M.hidden_layers:
        dict['hidden_layers'].append(layer_to_json(layer))
    with open(path, 'w+') as f:
        f.write(json.dumps(dict, cls=NumpyArrayEncoder))

def layer_to_json(L):
    if type(L) == Conv:
        return conv_layer_to_json(L)
    elif type(L) == Layer:
        return dense_layer_to_json(L)

def dense_layer_to_json(L):
    dict={}
    dict['type'] = 'dense'
    dict['id'] = L.id
    dict['input_size'] = L.input_size
    dict['neurons'] = L.neurons
    dict['act_fun'] = activation_to_json(L.act_fun)
    dict['is_ending'] = L.is_ending
    dict['input_layers_ids'] = L.input_layers_ids
    dict['output_layers_ids'] = L.output_layers_ids
    dict['W'] = L.W
    dict['B'] = L.B
    return dict

def conv_layer_to_json(L):
    dict={}
    dict['type'] = 'conv'
    dict['id'] = L.id
    dict['act_fun'] = activation_to_json(L.act_fun)
    dict['is_ending'] = L.is_ending
    dict['input_layers_ids'] = L.input_layers_ids
    dict['output_layers_ids'] = L.output_layers_ids
    dict['input_shape'] = L.input_shape
    dict['input_height'] = L.input_height
    dict['input_width'] = L.input_width
    dict['input_depth'] = L.input_depth 
    dict['input_flatten'] = L.input_flatten
    dict['depth'] = L.depth
    dict['kernel_size'] = L.kernel_size
    dict['output_shape'] = L.output_shape
    dict['output_flatten'] = L.output_flatten
    dict['kernels_shape'] = L.kernels_shape
    dict['kernels'] = L.kernels
    dict['biases'] = L.biases
    return dict

def loss_to_json(loo):
    return str(loo.__name__)

def activation_to_json(act):
    return str(act.__name__)

def load_model(path):
    with open(path, "r") as f:
        data = json.load(f)
    M = Model(data['input_size'], data['output_size'], data['hidden_size'])
    M.input_layers.clear()
    if not 'input_layers' in data:
        layer = json_to_layer(data['input_layer'])
        layer.model = M
        M.input_layers.append(layer)
    else:
        for layer_data in data['input_layers']:
            #print("loading ...")
            layer = json_to_layer(layer_data)
            layer.model = M
            M.input_layers.append(layer)
    #M.input_layer = json_to_layer(data['input_layer'])
    #M.input_layer.model = M      
    for layer_data in data['hidden_layers']:
        layer = json_to_layer(layer_data)
        layer.model = M
        M.hidden_layers.append(layer)
    M.avaible_id = data['avaible_id']
    M.output_layer = json_to_layer(data['output_layer'])
    M.output_layer.model = M
    M.batch_size = data['batch_size']
    M.loss_function = json_to_loss(data['loss_function'])
    M.activation_fun = json_to_activation(data['activation_fun'])
    M.convolution = data['convolution']
    M.input_shape = data['input_shape']
    M.kernel_size = data['kernel_size']
    M.depth = data['depth']
    return M

def json_to_layer(data):
    if data['type'] == 'conv':
        return json_to_conv_layer(data)
    elif data['type'] == 'dense':
        return json_to_dense_layer(data)
    
def json_to_dense_layer(data):
    layer = Layer(data['id'], None, data['input_size'], data['neurons'], json_to_activation(data['act_fun']))
    layer.is_ending = data['is_ending']
    layer.input_layers_ids = data['input_layers_ids']
    layer.output_layers_ids = data['output_layers_ids']
    layer.f_input = []
    layer.b_input = []
    layer.W = np.asarray(data['W'])
    layer.B = np.asarray(data['B'])
    return layer

def json_to_conv_layer(data):
    layer = Conv(data['id'], None, data['input_shape'], data['kernel_size'], data['depth'], json_to_activation(data['act_fun']))
    layer.is_ending = data['is_ending']
    layer.input_layers_ids = data['input_layers_ids']
    layer.output_layers_ids = data['output_layers_ids']
    layer.f_input = []
    layer.b_input = []
    layer.input_shape = data['input_shape']
    layer.input_depth = data['input_depth']
    layer.input_height = data['input_height']
    layer.input_width = data['input_width']
    layer.depth = data['depth']
    layer.output_shape = data['output_shape']
    layer.kernels_shape = data['kernels_shape']
    layer.kernels = np.asarray(data['kernels'])
    layer.biases = np.asarray(data['biases'])
    return layer

def json_to_loss(loo):
    if 'MSE' in loo:
        return Loss.MSE
    if 'multiclass_cross_entropy' in loo:
        return Loss.multiclass_cross_entropy

def json_to_activation(act):
    if 'ReLu' in act: return Activations.ReLu
    if 'leaky_ReLu' in act: return Activations.leaky_ReLu
    if 'SoftMax' in act: return Activations.SoftMax
    if 'Sigmoid' in act: return Activations.Sigmoid

# a = np.random.uniform(0, 10, (2,2,2))
# b = np.random.uniform(0, 10, (2,2))
# c = np.random.uniform(0, 10, (2))
# print("a: ", a)
# print("b: ", b)
# print("c: ", c)

# dict = {}
# dict['a'] = a
# dict['b'] = b
# dict['c'] = c
# dict['costam'] = "zwykly text"
# dict['obiekt'] = {'cos1': 123, 'cos2': 456}
# dict['lista'] = ['qww', {'as': 1}, 1]
# dict['tupla'] = 'asd', 'dddd', 1

# print("dict: ", dict)

# with open('tmp.txt', 'w+') as f:
#     f.write(json.dumps(dict, cls=NumpyArrayEncoder))

# with open('tmp.txt', "r") as f:
#     data = json.load(f)

# print("data: ", data)
# print("data a: ", np.asarray(data['a']))
# print("data b: ", np.asarray(data['b']))
# print("data c: ", np.asarray(data['c']))
# print("data ", data['costam'])
# print("data ", data['obiekt'])
# print("data ", data['lista'])
# print("data ", data['tupla'])