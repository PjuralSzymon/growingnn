from .structure import *
import numpy as np
from math import floor
from .config import config

class Action:
    def __init__(self, _params):
        self.params = _params
        pass

    def execute(self, Model):
        pass

    def can_be_infulenced(self, by_action):
        pass

    @staticmethod
    def generate_all_actions(Model):
        # Pre-allocate list with estimated size to avoid multiple resizing
        result = []
        
        # Generate all actions in one pass
        if config.ACTIONS_ENABLE_ADD_SEQ_LAYER:
            adding_layer_seq_actions = Add_Seq_Layer.generate_all_actions(Model)
            result.extend(adding_layer_seq_actions)
        if config.ACTIONS_ENABLE_ADD_RES_LAYER:
            adding_layer_res_actions = Add_Res_Layer.generate_all_actions(Model)
            result.extend(adding_layer_res_actions)
        if config.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER:
            adding_layer_conv_seq_actions = Add_Seq_Conv_Layer.generate_all_actions(Model)
            result.extend(adding_layer_conv_seq_actions)
        if config.ACTIONS_ENABLE_ADD_RES_CONV_LAYER:
            adding_layer_conv_res_actions = Add_Res_Conv_Layer.generate_all_actions(Model)
            result.extend(adding_layer_conv_res_actions)
        if config.ACTIONS_ENABLE_DEL_LAYER:
            delete_layer_actions = Del_Layer.generate_all_actions(Model)
            result.extend(delete_layer_actions)
        if config.ACTIONS_ENABLE_DEL_NEURONS_01:
            delete_neurons_actions_01 = Del_neurons.generate_all_actions(Model,0.1)
            result.extend(delete_neurons_actions_01)
        if config.ACTIONS_ENABLE_DEL_NEURONS_05:
            delete_neurons_actions_05 = Del_neurons.generate_all_actions(Model,0.5)
            result.extend(delete_neurons_actions_05)
        if config.ACTIONS_ENABLE_DEL_NEURONS_09:
            delete_neurons_actions_09 = Del_neurons.generate_all_actions(Model,0.9)
            result.extend(delete_neurons_actions_09)

        return result

class Add_Seq_Layer(Action):
    def execute(self, Model):
        Model.add_norm_layer(self.params[0], self.params[1], self.params[2])
    
    def can_be_infulenced(self, by_action):
        if type(by_action) is Del_Layer:
            if by_action.params == self.params[0]: return True
            if by_action.params == self.params[1]: return True
        return False

    @staticmethod
    def generate_all_actions(Model):
        pairs = Model.get_sequence_connection()
        pairs = delete_repetitions(pairs)
        actions = []
        for pair in pairs:
            if type(Model.get_layer(pair[1])) != Conv:
                actions.append(Add_Seq_Layer([pair[0], pair[1], Layer_Type.EYE]))
        return actions
    
    def __str__(self):
        return " ( Add Seq Layer Action: " + str(self.params) + " ) "

class Empty(Action):
    def execute(self, Model):
        pass

    def can_be_infulenced(self, by_action):
        return False

    def generate_all_actions(Model):
        return [Empty(None)]

    def __str__(self):
        return " ( Empty Action )"
    
class Add_Res_Layer(Action):
    def execute(self, Model):
        Model.add_res_layer(self.params[0], self.params[1], self.params[2])
    
    def can_be_infulenced(self, by_action):
        if type(by_action) is Del_Layer:
            if by_action.params == self.params[0]: return True
            if by_action.params == self.params[1]: return True
        return False

    @staticmethod
    def generate_all_actions(Model):
        pairs = Model.get_all_childrens_connections()
        pairs = delete_repetitions(pairs)
        actions = []
        layer_types = [Layer_Type.ZERO, Layer_Type.RANDOM, Layer_Type.EYE]
        
        for pair in pairs:
            if type(Model.get_layer(pair[1])) != Conv:
                for layer_type in layer_types:
                    actions.append(Add_Res_Layer([pair[0], pair[1], layer_type]))
        return actions
    
    def __str__(self):
        return " ( Add Res Layer Action: " + str(self.params) + " ) "


class Add_Seq_Conv_Layer(Action):
    def execute(self, Model):
        Model.add_conv_norm_layer(self.params[0], self.params[1])

    def can_be_infulenced(self, by_action):
        if type(by_action) is Del_Layer:
            if by_action.params == self.params[0]: return True
            if by_action.params == self.params[1]: return True
        return False

    @staticmethod
    def generate_all_actions(Model):
        pairs = Model.get_sequence_connection()
        pairs = delete_repetitions(pairs)
        actions = []
        for pair in pairs:
            layer_from = Model.get_layer(pair[0])
            layer_to = Model.get_layer(pair[1])
            if type(layer_from) == Conv:
                if type(Model.get_layer(pair[1])) == Conv:
                    actions.append(Add_Seq_Conv_Layer(pair))                
                elif type(layer_to) == Layer:
                    actions.append(Add_Seq_Conv_Layer(pair))
        return actions
    
    def __str__(self):
        return " ( Add Seq Conv Layer Action: " + str(self.params) + " ) "
    

class Add_Res_Conv_Layer(Action):
    def execute(self, Model):
        Model.add_conv_res_layer(self.params[0], self.params[1])
    
    def can_be_infulenced(self, by_action):
        if type(by_action) is Del_Layer:
            if by_action.params == self.params[0]: return True
            if by_action.params == self.params[1]: return True
        return False

    @staticmethod
    def generate_all_actions(Model):
        pairs = Model.get_all_childrens_connections()
        pairs = delete_repetitions(pairs)
        actions = []
        for pair in pairs:
            layer_from = Model.get_layer(pair[0])
            layer_to = Model.get_layer(pair[1])
            if type(layer_from) == Conv:
                if type(Model.get_layer(pair[1])) == Conv:
                    actions.append(Add_Res_Conv_Layer(pair))                
                elif type(layer_to) == Layer:
                    actions.append(Add_Res_Conv_Layer(pair))
        return actions
    
    def __str__(self):
        return " ( Add Res Conv Layer Action: " + str(self.params) + " ) "
    
class Del_Layer(Action):
    def execute(self, Model):
        Model.remove_layer(self.params)

    def can_be_infulenced(self, by_action):
        return False

    def generate_all_actions(Model):
        actions = []
        for layer_hidden in Model.hidden_layers:
            actions.append(Del_Layer(layer_hidden.id))
        return actions

    def __str__(self):
        return " ( Del Layer Action: " + str(self.params) + " ) "
    

class Del_neurons(Action):

    def execute(self, Model):
        Model.get_layer(self.params[0]).remove_neurons(self.params[1])

    def can_be_infulenced(self, by_action):
        return False

    def generate_all_actions(Model, remove_neurons_ratio = 0.5):
        actions = []
        for layer_hidden in Model.hidden_layers:
            if type(Model.get_layer(layer_hidden.id)) != Conv:
                if floor(Model.get_layer(layer_hidden.id).neurons * remove_neurons_ratio) < config.MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL:
                    continue
                params = [layer_hidden.id, remove_neurons_ratio]
                actions.append(Del_neurons(params))
        return actions

    def __str__(self):
        return " ( Del Neurons Action: " + str(self.params) + " ) "