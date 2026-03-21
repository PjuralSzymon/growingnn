"""
Model storage and serialization utilities.
"""
import json
import numpy as np
from ..helpers import get_numpy_array
from ..optimizers import OptimizerFactory
from .loss import Loss
from .activations import Activations


class Storage:
    """Model storage and serialization utilities."""
    
    def __init__(self):
        """Initialize storage utilities."""
        pass

    @staticmethod
    def loadModel(path):
        """Load model from JSON file."""
        with open(path, 'r') as json_file:
            model_dict = json.load(json_file)
        return Storage.dictToModel(model_dict)

    @staticmethod
    def saveModel(model, path):
        """Save model to JSON file."""
        with open(path, 'w') as json_file:
            json.dump(Storage.modelToDict(model), json_file, indent=4)

    @staticmethod
    def modelToDict(model):
        """Convert model to dictionary for serialization."""
        dict_main = {}
        dict_main['settings'] = {}
        dict_main['settings']['batch_size'] = model.batch_size
        dict_main['settings']['loss_function'] = str(model.loss_function.__name__)
        dict_main['settings']['output_size'] = model.output_size
        dict_main['settings']['hidden_size'] = model.hidden_size
        dict_main['settings']['avaible_id'] = model.avaible_id
        dict_main['settings']['activation_fun'] = str(model.activation_fun.__name__)
        dict_main['settings']['convolution'] = model.convolution
        dict_main['settings']['input_size'] = model.input_size
        dict_main['settings']['input_shape'] = model.input_shape
        dict_main['settings']['kernel_size'] = model.kernel_size
        dict_main['settings']['depth'] = model.depth
        dict_main['settings']['optimizer'] = OptimizerFactory.ToDict(model.optimizer)
        dict_main['hidden_layers'] = {}
        dict_main['input_layers'] = {}
        dict_main['output_layer'] = {}
        
        for layer in model.hidden_layers:
            dict_main['hidden_layers'][layer.id] = Storage.layerToDict(layer)
        for layer in model.input_layers:
            dict_main['input_layers'][layer.id] = Storage.layerToDict(layer)
        dict_main['output_layer'][layer.id] = Storage.layerToDict(model.output_layer)
        return dict_main

    @staticmethod
    def layerToDict(layer):
        """Convert layer to dictionary for serialization."""
        dict_main = {}
        dict_main['id'] = str(layer.id)
        dict_main['act_fun'] = str(layer.act_fun.__name__)
        dict_main['is_ending'] = layer.is_ending
        dict_main['is_starting'] = layer.is_starting
        dict_main['input_layers_ids'] = [str(i) for i in layer.input_layers_ids] 
        dict_main['output_layers_ids'] = [str(i) for i in layer.output_layers_ids]
        dict_main['optimizer'] = OptimizerFactory.ToDict(layer.optimizer)
        dict_reshapers = {}
        for resheper_key in layer.reshspers.keys():
            reshsper_id = str(resheper_key)
            (resheper_from, resharper_to) = resheper_key
            dict_reshapers[reshsper_id] = {}
            dict_reshapers[reshsper_id]['from'] = resheper_from
            dict_reshapers[reshsper_id]['to'] = resharper_to
            dict_reshapers[reshsper_id]["matrix"] = get_numpy_array(layer.reshspers[resheper_key]).tolist()
        dict_main['reshapers'] = dict_reshapers
        dict_main['weights'] = {}
        
        if type(layer).__name__ == 'Layer':
            dict_main['neurons'] = layer.neurons
            dict_main['input_size'] = layer.input_size
            dict_main['weights']['W'] = get_numpy_array(layer.W).tolist()
            dict_main['weights']['B'] = get_numpy_array(layer.B).tolist()
        if type(layer).__name__ == 'Conv':
            dict_main['weights']['kernels'] = get_numpy_array(layer.kernels).tolist()
            dict_main['weights']['biases'] = get_numpy_array(layer.biases).tolist()
            dict_main['conv'] = {}
            dict_main['conv']['input_shape'] = layer.input_shape
            dict_main['conv']['input_height'] = layer.input_height
            dict_main['conv']['input_width'] = layer.input_width
            dict_main['conv']['input_depth'] = layer.input_depth
            dict_main['conv']['input_flatten'] = layer.input_flatten
            dict_main['conv']['depth'] = layer.depth
            dict_main['conv']['kernel_size'] = layer.kernel_size
            dict_main['conv']['output_shape'] = layer.output_shape
            dict_main['conv']['output_flatten'] = layer.output_flatten
            dict_main['conv']['kernels_shape'] = layer.kernels_shape
        return dict_main

    @staticmethod
    def dictToModel(dict_main):
        """Convert dictionary to model."""
        from ..structure import Model, Layer, Conv, Layer_Type
        
        batch_size = dict_main['settings']['batch_size']
        loss_function_name = dict_main['settings']['loss_function']
        output_size = dict_main['settings']['output_size']
        hidden_size = dict_main['settings']['hidden_size']
        avaible_id = dict_main['settings']['avaible_id']
        activation_fun_name = dict_main['settings']['activation_fun']
        convolution = dict_main['settings']['convolution']
        input_shape = dict_main['settings']['input_shape']
        kernel_size = dict_main['settings']['kernel_size']
        depth = dict_main['settings']['depth']
        input_size = dict_main['settings']['input_size']
        loss_function = Loss.getByName(loss_function_name)
        activation_fun = Activations.getByName(activation_fun_name)
        optimizer = OptimizerFactory.FromDict(dict_main['settings']['optimizer'])
        model = Model(input_size, hidden_size, output_size, loss_function, activation_fun, 1, optimizer)
        model.batch_size = batch_size
        model.avaible_id = avaible_id
        model.convolution = convolution
        model.kernel_size = kernel_size
        model.depth = depth
        model.input_layers = []
        model.hidden_layers = []

        for layer_id, layer_dict in dict_main['input_layers'].items():
            layer = Storage.dictToLayer(layer_dict, model)
            model.input_layers.append(layer)

        for layer_id, layer_dict in dict_main['hidden_layers'].items():
            layer = Storage.dictToLayer(layer_dict, model)
            model.hidden_layers.append(layer)

        for layer_id, layer_dict in dict_main['output_layer'].items():
            layer = Storage.dictToLayer(layer_dict, model)
            model.output_layer = layer
        return model

    @staticmethod
    def dictToLayer(layer_dict, model):
        """Convert dictionary to layer."""
        from ..structure import Layer, Conv, Layer_Type
        
        layer_id = str(layer_dict['id'])
        act_fun_name = layer_dict['act_fun']
        is_ending = layer_dict['is_ending']
        is_starting = layer_dict['is_starting']
        input_layers_ids = layer_dict['input_layers_ids']
        output_layers_ids = layer_dict['output_layers_ids']
        reshapers = layer_dict['reshapers']
        act_fun = Activations.getByName(act_fun_name)
        optimizer = OptimizerFactory.FromDict(layer_dict['optimizer'])
        
        if 'conv' in layer_dict:
            conv_dict = layer_dict['conv']
            layer = Conv(layer_id, model, conv_dict['input_shape'], conv_dict['kernel_size'], conv_dict['depth'], act_fun, optimizer)
            layer.input_height = conv_dict['input_height']
            layer.input_width = conv_dict['input_width']
            layer.input_depth = conv_dict['input_depth']
            layer.input_flatten = conv_dict['input_flatten']
            layer.output_shape = conv_dict['output_shape']
            layer.output_flatten = conv_dict['output_flatten']
            layer.kernels_shape = conv_dict['kernels_shape']
        else:
            input_size = layer_dict['input_size']
            neurons = layer_dict['neurons']
            layer = Layer(layer_id, model, input_size, neurons, act_fun, Layer_Type.RANDOM, optimizer.getDense())
        
        if is_ending: 
            layer.set_as_ending()
        if is_starting: 
            layer.set_as_starting()
        layer.input_layers_ids = input_layers_ids
        layer.output_layers_ids = output_layers_ids
        reshapers_dict = {}
        for reshaper_id, reshaper_info in reshapers.items():
            reshaper_from = reshaper_info['from']
            reshaper_to = reshaper_info['to']
            reshaper_matrix = np.array(reshaper_info['matrix'])
            reshapers_dict[(reshaper_from, reshaper_to)] = reshaper_matrix
        layer.reshspers = reshapers_dict

        if 'weights' in layer_dict:
            weights_dict = layer_dict['weights']
            if 'W' in weights_dict:
                layer.W = np.array(weights_dict['W'])
            if 'B' in weights_dict:
                layer.B = np.array(weights_dict['B'])
            if 'kernels' in weights_dict:
                layer.kernels = np.array(weights_dict['kernels'])
            if 'biases' in weights_dict:
                layer.biases = np.array(weights_dict['biases'])
        return layer
