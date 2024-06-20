from .structure import *
import numpy as np
import math

class Action:
    def __init__(self, _params):
        self.params = _params
        pass

    def execute(self, Model):
        pass

    def can_be_infulenced(self, by_action):
        pass

    def generate_all_actions(Model):
        result = []
        adding_layer_seq_actions = Add_Seq_Layer.generate_all_actions(Model)
        adding_layer_res_actions = Add_Res_Layer.generate_all_actions(Model)
        adding_layer_conv_seq_actions = Add_Seq_Conv_Layer.generate_all_actions(Model)
        adding_layer_conv_res_actions = Add_Res_Conv_Layer.generate_all_actions(Model)
        #adding_layer_seq_output_action = Add_seq_output_layer.generate_all_actions(Model)
        delete_layer_actions = Del_Layer.generate_all_actions(Model)
        #result += adding_layer_seq_output_action
        result += adding_layer_seq_actions
        result += adding_layer_res_actions
        result += adding_layer_conv_seq_actions
        result += adding_layer_conv_res_actions
        result += delete_layer_actions
        return result

class Add_Seq_Layer(Action):
    def execute(self, Model):
        Model.add_norm_layer(self.params[0], self.params[1])
    
    def can_be_infulenced(self, by_action):
        if type(by_action) is Del_Layer:
            if by_action.params == self.params[0]: return True
            if by_action.params == self.params[1]: return True
        return False

    def generate_all_actions(Model):
        pairs = Model.get_sequence_connection()
        pairs = delete_repetitions(pairs)
        actions = []
        for pair in pairs:
            if type(Model.get_layer(pair[1])) != Conv:
                actions.append(Add_Seq_Layer(pair))
        return actions
    
    def __str__(self):
        return " ( Add Seq Layer Action: " + str(self.params) + " ) "

class Add_seq_output_layer(Action):
    def execute(self, Model):
        Model.add_sequential_output_Layer()

    def can_be_infulenced(self, by_action):
        return False

    def generate_all_actions(Model):
        return [Add_seq_output_layer(None)]

    def __str__(self):
        return " ( Add sequential output Action )"
    
class Add_Res_Layer(Action):
    def execute(self, Model):
        Model.add_res_layer(self.params[0], self.params[1], self.params[2])
    
    def can_be_infulenced(self, by_action):
        if type(by_action) is Del_Layer:
            if by_action.params == self.params[0]: return True
            if by_action.params == self.params[1]: return True
        return False

    def generate_all_actions(Model):
        pairs = Model.get_all_childrens_connections()
        pairs = delete_repetitions(pairs)
        actions = []
        for pair in pairs:
            if type(Model.get_layer(pair[1])) != Conv:
                actions.append(Add_Res_Layer([pair[0], pair[1], Layer_Type.ZERO]))
        for pair in pairs:
            if type(Model.get_layer(pair[1])) != Conv:
                actions.append(Add_Res_Layer([pair[0], pair[1], Layer_Type.RANDOM]))
        for pair in pairs:
            if type(Model.get_layer(pair[1])) != Conv:
                actions.append(Add_Res_Layer([pair[0], pair[1], Layer_Type.EYE]))
        return actions
    
    def __str__(self):
        return " ( Add Res Layer Action: " + str(self.params) + " ) "

class Del_Layer(Action):
    def execute(self, Model):
        #print("Add_Layer: ", self.params)
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
    

class Add_Seq_Conv_Layer(Action):
    def execute(self, Model):
        Model.add_conv_norm_layer(self.params[0], self.params[1])

    def can_be_infulenced(self, by_action):
        if type(by_action) is Del_Layer:
            if by_action.params == self.params[0]: return True
            if by_action.params == self.params[1]: return True
        return False

    def generate_all_actions(Model):
        pairs = Model.get_sequence_connection()
        pairs = delete_repetitions(pairs)
        actions = []
        for pair in pairs:
            layer_from = Model.get_layer(pair[0])
            layer_to = Model.get_layer(pair[1])
            #print("type(layer_from): ", type(layer_from))
            if type(layer_from) == Conv:
                if type(Model.get_layer(pair[1])) == Conv:
                    i = layer_from.output_shape[0]
                    d = np.clip(layer_to.depth, 1, 10)
                    k = np.clip(layer_to.kernel_size, 1, i)  
                    output_shape = (i-k+1,i-k+1, d)  
                    output_flatten = output_shape[0] * output_shape[1] * output_shape[2]
                    #if output_shape == layer_to.input_shape:
                    actions.append(Add_Seq_Conv_Layer(pair))                
                elif type(layer_to) == Layer:
                    o = layer_to.input_size
                    i = layer_from.output_shape[0]
                    d = np.clip(math.floor(o ** ( 1 / 3)), 1, 10)
                    k = np.clip(i + 1 - math.ceil(o ** ( 1 / 3)), 1, i)
                    output_shape = (i-k+1,i-k+1, d)  
                    output_flatten = output_shape[0] * output_shape[1] * output_shape[2]
                    #if output_flatten == layer_to.input_size:
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

    def generate_all_actions(Model):
        pairs = Model.get_all_childrens_connections()
        pairs = delete_repetitions(pairs)
        actions = []
        for pair in pairs:
            layer_from = Model.get_layer(pair[0])
            layer_to = Model.get_layer(pair[1])
            if type(layer_from) == Conv:
                if type(Model.get_layer(pair[1])) == Conv:
                    i = layer_from.output_shape[0]
                    d = np.clip(layer_to.depth, 1, 10)
                    k = np.clip(layer_to.kernel_size, 1, i)  
                    output_shape = (i-k+1,i-k+1, d)  
                    output_flatten = output_shape[0] * output_shape[1] * output_shape[2]
                    #if output_shape == layer_to.input_shape:
                    actions.append(Add_Res_Conv_Layer(pair))                
                elif type(layer_to) == Layer:
                    o = layer_to.input_size
                    i = layer_from.output_shape[0]
                    d = np.clip(math.floor(o ** ( 1 / 3)), 1, 10)
                    k = np.clip(i + 1 - math.ceil(o ** ( 1 / 3)), 1, i)
                    output_shape = (i-k+1,i-k+1, d)  
                    output_flatten = output_shape[0] * output_shape[1] * output_shape[2]
                    #if output_flatten == layer_to.input_size:
                    actions.append(Add_Res_Conv_Layer(pair))
        return actions
    
    def __str__(self):
        return " ( Add Res Conv Layer Action: " + str(self.params) + " ) "