from .structure import *
import numpy as np
import math
from math import floor, ceil
from .config import config


class Action:
    def __init__(self, _params):
        self.params = _params
        pass

    def execute(self, model):
        pass

    def can_be_infulenced(self, by_action):
        pass

    @staticmethod
    def generate_all_actions(model):
        # Pre-allocate list with estimated size to avoid multiple resizing
        result = []
        
        # Generate all actions in one pass
        if config.ACTIONS_ENABLE_ADD_SEQ_LAYER:
            adding_layer_seq_actions = Add_Seq_Layer.generate_all_actions(model)
            result.extend(adding_layer_seq_actions)
        if config.ACTIONS_ENABLE_ADD_RES_LAYER:
            adding_layer_res_actions = Add_Res_Layer.generate_all_actions(model)
            result.extend(adding_layer_res_actions)
        if config.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER:
            adding_layer_conv_seq_actions = Add_Seq_Conv_Layer.generate_all_actions(model)
            result.extend(adding_layer_conv_seq_actions)
        if config.ACTIONS_ENABLE_ADD_RES_CONV_LAYER:
            adding_layer_conv_res_actions = Add_Res_Conv_Layer.generate_all_actions(model)
            result.extend(adding_layer_conv_res_actions)
        if config.ACTIONS_ENABLE_DEL_LAYER:
            delete_layer_actions = Del_Layer.generate_all_actions(model)
            result.extend(delete_layer_actions)

        if config.ACTIONS_ENABLE_DEL_NEURONS_01:
            delete_neurons_actions_01 = Del_neurons.generate_all_actions(model,0.1)
            result.extend(delete_neurons_actions_01)
        if config.ACTIONS_ENABLE_DEL_NEURONS_05:
            delete_neurons_actions_05 = Del_neurons.generate_all_actions(model,0.5)
            result.extend(delete_neurons_actions_05)
        if config.ACTIONS_ENABLE_DEL_NEURONS_09:
            delete_neurons_actions_09 = Del_neurons.generate_all_actions(model,0.9)
            result.extend(delete_neurons_actions_09)

        if config.ACTIONS_ENABLE_ADD_NEURONS_15:
            add_neurons_actions_15 = Add_neurons.generate_all_actions(model,1.5)
            result.extend(add_neurons_actions_15)
        if config.ACTIONS_ENABLE_ADD_NEURONS_20:
            add_neurons_actions_20 = Add_neurons.generate_all_actions(model,2.0)
            result.extend(add_neurons_actions_20)

        if len(result) == 0:
            result.append(Empty_action(None))
        return result

class Add_Seq_Layer(Action):
    def execute(self, model):
        model.add_norm_layer(self.params[0], self.params[1], self.params[2])
        model.forward_blank()
    
    def can_be_infulenced(self, by_action):
        if type(by_action) is Del_Layer:
            if by_action.params == self.params[0]: return True
            if by_action.params == self.params[1]: return True
        return False

    @staticmethod
    def generate_all_actions(model):
        pairs = model.get_sequence_connection()
        pairs = delete_repetitions(pairs)
        actions = []
        for pair in pairs:
            if type(model.get_layer(pair[1])) != Conv:
                layer_from = model.get_layer(pair[0])
                layer_to = model.get_layer(pair[1])
                additional_params = _get_layer_params("Layer", layer_from.get_output_size(), layer_to.input_size)
                if _check_parameter_limit(model, additional_params):
                    actions.append(Add_Seq_Layer([pair[0], pair[1], Layer_Type.EYE]))
        return actions
    
    def __str__(self):
        return " ( Add Seq Layer Action: " + str(self.params) + " ) "

class Empty(Action):
    def execute(self, model):
        pass

    def can_be_infulenced(self, by_action):
        return False

    def generate_all_actions(model):
        return [Empty(None)]

    def __str__(self):
        return " ( Empty Action )"
    
class Add_Res_Layer(Action):
    def execute(self, model):
        model.add_res_layer(self.params[0], self.params[1], self.params[2])
        model.forward_blank()
    
    def can_be_infulenced(self, by_action):
        if type(by_action) is Del_Layer:
            if by_action.params == self.params[0]: return True
            if by_action.params == self.params[1]: return True
        return False

    @staticmethod
    def generate_all_actions(model):
        pairs = model.get_all_childrens_connections()
        pairs = delete_repetitions(pairs)
        actions = []
        layer_types = [Layer_Type.ZERO, Layer_Type.RANDOM, Layer_Type.EYE]
        
        for pair in pairs:
            if type(model.get_layer(pair[1])) != Conv:
                layer_from = model.get_layer(pair[0])
                layer_to = model.get_layer(pair[1])
                additional_params = _get_layer_params("Layer", layer_from.get_output_size(), layer_to.input_size)
                if _check_parameter_limit(model, additional_params):
                    for layer_type in layer_types:
                        actions.append(Add_Res_Layer([pair[0], pair[1], layer_type]))
        return actions
    
    def __str__(self):
        return " ( Add Res Layer Action: " + str(self.params) + " ) "


class Add_Seq_Conv_Layer(Action):
    def execute(self, model):
        model.add_conv_norm_layer(self.params[0], self.params[1])
        model.forward_blank()

    def can_be_infulenced(self, by_action):
        if type(by_action) is Del_Layer:
            if by_action.params == self.params[0]: return True
            if by_action.params == self.params[1]: return True
        return False

    @staticmethod
    def generate_all_actions(model):
        pairs = model.get_sequence_connection()
        pairs = delete_repetitions(pairs)
        actions = []
        for pair in pairs:
            layer_from = model.get_layer(pair[0])
            layer_to = model.get_layer(pair[1])
            if type(layer_from) == Conv:
                if type(model.get_layer(pair[1])) == Conv:
                    i = layer_from.output_shape[0]
                    d = clip(layer_to.depth, 1, 3)
                    k = clip(layer_to.kernel_size, 1, i)
                    additional_params = _get_layer_params("Conv", i, i, d, k)
                    if _check_parameter_limit(model, additional_params):
                        actions.append(Add_Seq_Conv_Layer(pair))                
                elif type(layer_to) == Layer:
                    o = layer_to.input_size
                    i = layer_from.output_shape[0]
                    d = clip(math.floor(o ** ( 1 / 3)), 1, 3)
                    k = clip(i + 1 - math.ceil(o ** ( 1 / 3)), 1, i)
                    additional_params = _get_layer_params("Conv", i, i, d, k)
                    if _check_parameter_limit(model, additional_params):
                        actions.append(Add_Seq_Conv_Layer(pair))
        return actions
    
    def __str__(self):
        return " ( Add Seq Conv Layer Action: " + str(self.params) + " ) "
    

class Add_Res_Conv_Layer(Action):
    def execute(self, model):
        model.add_conv_res_layer(self.params[0], self.params[1])
        model.forward_blank()
    
    def can_be_infulenced(self, by_action):
        if type(by_action) is Del_Layer:
            if by_action.params == self.params[0]: return True
            if by_action.params == self.params[1]: return True
        return False

    @staticmethod
    def generate_all_actions(model):
        pairs = model.get_all_childrens_connections()
        pairs = delete_repetitions(pairs)
        actions = []
        for pair in pairs:
            layer_from = model.get_layer(pair[0])
            layer_to = model.get_layer(pair[1])
            if type(layer_from) == Conv:
                if type(model.get_layer(pair[1])) == Conv:
                    d = layer_to.depth
                    i = layer_from.output_shape[0]
                    k = clip(layer_to.kernel_size, 1, i)
                    additional_params = _get_layer_params("Conv", i, i, d, k)
                    if _check_parameter_limit(model, additional_params):
                        actions.append(Add_Res_Conv_Layer(pair))                
                elif type(layer_to) == Layer:
                    o = layer_to.input_size
                    i = layer_from.output_shape[0]
                    d = clip(math.floor(o ** ( 1 / 3)), 1, 3)
                    k = clip(i + 1 - math.ceil(o ** ( 1 / 3)), 1, i)
                    additional_params = _get_layer_params("Conv", i, i, d, k)
                    if _check_parameter_limit(model, additional_params):
                        actions.append(Add_Res_Conv_Layer(pair))
        return actions
    
    def __str__(self):
        return " ( Add Res Conv Layer Action: " + str(self.params) + " ) "
    
class Del_Layer(Action):
    def execute(self, model):
        model.forward_blank()
        model.remove_layer(self.params)

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model):
        actions = []
        for layer_hidden in model.hidden_layers:
            actions.append(Del_Layer(layer_hidden.id))
        return actions

    def __str__(self):
        return " ( Del Layer Action: " + str(self.params) + " ) "
    

# class Scale_neurons(Action):

#     def execute(self, model):
#         model.forward_blank()
#         model.get_layer(self.params[0]).scale_neurons(self.params[1])

#     def can_be_infulenced(self, by_action):
#         return False

#     @staticmethod
#     def generate_all_actions(model, scale_neurons_ratio = 0.5):
#         actions = []
#         for layer_hidden in model.hidden_layers + model.input_layers:
#             if type(model.get_layer(layer_hidden.id)) != Conv:
#                 if floor(model.get_layer(layer_hidden.id).neurons * scale_neurons_ratio) < config.MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL:
#                     continue
#                 params = [layer_hidden.id, scale_neurons_ratio]
#                 actions.append(Del_neurons(params))
#         return actions

#     def __str__(self):
#         return " ( Change Amount ofNeurons base Action: " + str(self.params) + " ) "
    

class Del_neurons(Action):
    def execute(self, model):
        model.forward_blank()
        model.get_layer(self.params[0]).scale_neurons(self.params[1])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model, scale_neurons_ratio=0.5):
        actions = []
        for layer_hidden in model.hidden_layers + model.input_layers:
            if type(model.get_layer(layer_hidden.id)) != Conv:
                # Check if the new layer size doesn't go below minimum
                new_neurons = floor(model.get_layer(layer_hidden.id).neurons * scale_neurons_ratio)
                if new_neurons >= config.MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL:
                    params = [layer_hidden.id, scale_neurons_ratio]
                    actions.append(Del_neurons(params))
        return actions

    def __str__(self):
        return " ( Del Neurons Action: " + str(self.params) + " ) "

class Add_neurons(Action):

    def execute(self, model):
        model.forward_blank()
        model.get_layer(self.params[0]).scale_neurons(self.params[1])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model, scale_neurons_ratio=1.5):
        actions = []
        for layer_hidden in model.hidden_layers + model.input_layers:
            if type(model.get_layer(layer_hidden.id)) != Conv:
                new_neurons = floor(model.get_layer(layer_hidden.id).neurons * scale_neurons_ratio)
                if new_neurons <= config.MAXIMUM_MATRIX_NEURONS_SIZE_FOR_NEURONS_ADDITION:
                    layer = model.get_layer(layer_hidden.id)
                    additional_params = (new_neurons - layer.neurons) * layer.input_size
                    if _check_parameter_limit(model, additional_params):
                        params = [layer_hidden.id, scale_neurons_ratio]
                        actions.append(Add_neurons(params))
        return actions

    def __str__(self):
        return " ( Add Neurons Action: " + str(self.params) + " ) "

class Empty_action(Action):

    def execute(self, model):
        pass

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model):
        return [Empty_action(None)]

    def __str__(self):
        return " ( Empty action ) "



def _check_parameter_limit(model, additional_params=0):
    """Check if adding additional parameters would exceed the model parameter limit"""
    current_params = model.get_parametr_count()
    total_params = current_params + additional_params
    return total_params <= config.MAX_MODEL_PARAMETERS

def _get_layer_params(layer_type, input_size, output_size, depth=None, kernel_size=None):
    """Calculate how many parameters would be added by a new layer"""
    if layer_type == "Layer":
        return input_size * output_size
    elif layer_type == "Conv":
        if depth is None or kernel_size is None:
            return 0
        return int(depth) * int(input_size) * int(kernel_size) * int(kernel_size)
    return 0