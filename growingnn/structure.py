import math
import sys
from enum import Enum
import json
import threading
import os
from .painter import *
from .config import *
from .optimizers import *
from .quaziIdentity import *

def switch_to_gpu():
    global np, IS_CUPY, correlate, convolve
    import cupy as np
    IS_CUPY = True
    from cupyx.scipy.ndimage import correlate
    from cupyx.scipy.ndimage import convolve

def switch_to_cpu():
    global np, IS_CUPY, correlate2d, convolve2d
    import numpy as np
    IS_CUPY = False
    from scipy.signal import correlate2d
    from scipy.signal import convolve2d

class Loss:
    
    def getByName(name):
        if name == Loss.MSE.__name__:
            return Loss.MSE
        elif name == Loss.multiclass_cross_entropy.__name__:
            return Loss.multiclass_cross_entropy

    class MSE:
        __name__ = 'MSE'
        def exe(Y_true, Y_pred):
            return np.sum((Y_pred - Y_true)**2)/Y_pred.shape[0]
        def der(Y_true, Y_pred):
            return Y_pred - Y_true
    class multiclass_cross_entropy:
        __name__ = 'multiclass_cross_entropy'
        def exe(Y_true, Y_pred):
            error = 0.0
            for i in range(0, Y_true.shape[1]):
                error -= np.dot(Y_true[:,i].T, np.log(Y_pred[:,i]))
            #print("Y_true.shape: ", Y_true.shape)
            return error / Y_true.shape[1]
        def der(Y_true, Y_pred):
            grad = np.zeros(Y_true.shape)
            for i in range(0, Y_true.shape[1]):
                #print("Y_true[:, i]: ", Y_true[:, i].shape)
                #print("Y_pred[:, i]: ", Y_pred[:, i].shape)
                partial_grad = -Y_true[:, i] / Y_pred[:, i]
                A = np.tile(np.reshape(Y_pred[:, i], (Y_pred.shape[0], 1)), (1, Y_pred.shape[0]))
                grad[:, i] = (A * (np.identity(Y_pred.shape[0]) - A.T)) @ partial_grad
            return grad
               
class Activations:

    def getByName(name):
        if name == Activations.ReLu.__name__:
            return Activations.ReLu
        elif name == Activations.leaky_ReLu.__name__:
            return Activations.leaky_ReLu
        elif name == Activations.SoftMax.__name__:
            return Activations.SoftMax
        elif name == Activations.Sigmoid.__name__:
            return Activations.Sigmoid
    class ReLu:
        __name__ = 'ReLu'

        @staticmethod
        @jit(nopython=True)
        def exe(X):
            return numpy.maximum(X,0)
        
        @staticmethod
        @jit(nopython=True)
        def der(X):
            return X > 0
        
    class leaky_ReLu:
        __name__ = 'leaky_ReLu'

        @staticmethod
        @jit(nopython=True)
        def exe(X):
            return  np.where(X > 0, X, X * 0.001)
        
        @staticmethod
        @jit(nopython=True)
        def der(X):
            return  np.where(X > 0, 1, 0.001)
        
    class SoftMax:
        __name__ = 'SoftMax'

        @staticmethod
        def exe(X):
            result = np.zeros(X.shape)
            for i in range(0, X.shape[1]):
                exp = np.exp(X[:, i] - np.nanmax(X[:, i]))
                result[:, i] = np.nan_to_num(exp / np.sum(exp))
            return clip(result, 0.0001, 0.999)
        
        @staticmethod
        @jit(nopython=True)
        def der(X):
            return 1.0
        
    class Sigmoid:
        __name__ = 'Sigmoid'

        @staticmethod
        @jit(nopython=True)
        def exe(X):
            return  1/(1 + np.exp(-X))
        
        @staticmethod
        @jit(nopython=True)
        def der(X):
            sigm = 1/(1 + np.exp(-X))
            return sigm * (1.0 - sigm)
        
class LearningRateScheduler:
    CONSTANT = 0
    PROGRESIVE = 1
    PROGRESIVE_PARABOIDAL = 2

    def __init__(self, mode, alpha, steepness = 0.2):
        self.mode = mode
        self.alpha = alpha
        self.steepness = steepness

    def alpha_scheduler(self, i, iterations):
        thresh = float(self.steepness * iterations)
        i = float(i)
        iterations = float(iterations)
        result = self.alpha
        if self.mode == LearningRateScheduler.CONSTANT:
            result =  self.alpha
        elif self.mode == LearningRateScheduler.PROGRESIVE:
            if i < thresh:
                return self.alpha * ((i+1) / (thresh + 2))
            result = self.alpha * (1 - (i - thresh) / (iterations - thresh + 2))
        else:
            thresh = self.steepness * iterations
            if i < thresh:
                return self.alpha * ( -1 * ((1) / (pow(thresh, 2))) * pow((i) - thresh, 2) + 1)
            result = self.alpha * (-1 * ((1) / (pow(iterations - thresh, 2))) * pow((i) - thresh, 2) + 1)
        return max(0, result)

class SimulationScheduler:
    CONSTANT = 0
    PROGRESS_CHECK = 1
    OVERFIT_CHECK = 2

    def __init__(self, mode, simulation_time, simulation_epochs, min_grow_rate = 0.65):
        self.mode = mode
        self.simulation_time = simulation_time
        self.simulation_epochs = simulation_epochs
        self.min_grow_rate = min_grow_rate
        self.flatten_check_length = 20
        self.progres_delta = 0.01

    def can_simulate(self, i, hist_detail):
        new_acc = 0#hist_detail.Y['iteration_acc_train'][-1]
        prev_acc = 0#hist_detail.Y['iteration_acc_train'][-2]
        if len(hist_detail.Y['iteration_acc_train']) > 0: new_acc = hist_detail.Y['iteration_acc_train'][-1]
        if len(hist_detail.Y['iteration_acc_train']) > 1: prev_acc = hist_detail.Y['iteration_acc_train'][-2]
        new_acc = get_numpy_array(new_acc)
        prev_acc = get_numpy_array(prev_acc)
        if self.mode == SimulationScheduler.CONSTANT: 
            print("[iteration: "+str(i)+"] Constant frequency od simulaiton acc: " + str(new_acc)+ " starting simulation." )
            return True
        elif self.mode == SimulationScheduler.PROGRESS_CHECK:
            if new_acc - prev_acc < self.progres_delta:
                print("[iteration: "+str(i)+"] No correction detected acc: " + str(new_acc)+ "(prev: " + str(prev_acc) + ") starting simulation."  )
                return True
            else:
                print("[iteration: "+str(i)+"] correction detected acc: " + str(new_acc)+ "(prev: " + str(prev_acc) + ") training continues.")
                return False
        elif self.mode == SimulationScheduler.OVERFIT_CHECK:
            if hist_detail.learning_capable():
                print("[iteration: "+str(i)+"] Overfit detected: " + str(new_acc)+ " starting simulation."  )
                return True
            else:
                print("[iteration: "+str(i)+"] No overfit detected: " + str(new_acc)+ " training continues.")
                return False
        return False
    
    def get_mode_label(self):
        if self.mode == SimulationScheduler.CONSTANT: return "simconstant"
        elif self.mode == SimulationScheduler.PROGRESS_CHECK: return "simprogres"
        elif self.mode == SimulationScheduler.OVERFIT_CHECK: return "simoverfit"


class History:
    def __init__(self, keys):
        self.Y = {}
        self.last_img_id = 0
        self.description = "\ndescription_of_training_process: \n"
        self.best_train_acc = 0.0
        self.best_test_acc = 0.0
        for key in keys:
            self.Y[key] = []
    def get_length(self):
        return len(self.Y[list(self.Y.keys())[0]])
    def append(self, key, value):
        if not key in self.Y.keys():
            self.Y[key] = []
        self.Y[key].append(value)
    def merge(self, new_hist):
        for key in self.Y.keys():
            if key in self.Y.keys() and key in new_hist.Y.keys():
                self.Y[key] += new_hist.Y[key]
    def learning_capable(self, patience=10, verbose=0.5):
        if len(self.Y['loss']) < patience: return True
        diff = 0
        last_values = self.Y['loss'][::-1]
        for i in range(1, len(last_values)):
            diff += abs(last_values[i] - last_values[i - 1])
        return diff > verbose
    def get_last(self, key):
        return self.Y[key][-1]
    def draw_hist(self, label, path):
        for key in self.Y.keys():
            #print("len(self.Y[key]): ", len(self.Y[key]), type(self.Y[key][0]))
            xc = range(0, len(self.Y[key]))
            plt.figure()
            plt.plot(xc, get_list_as_numpy_array(self.Y[key]), label = key)
            plt.legend()
            plt.savefig(path + "/" + label + "_" + key + ".png")
            plt.close()
    def save(self, path):
        dict = {}
        dict['keys'] = {}
        for key in self.Y.keys():
            dict['keys'][key] = np.array(get_list_as_numpy_array(self.Y[key]))
        dict['last_img_id'] = str(self.last_img_id)
        dict['description'] = str(self.description)
        dict['best_train_acc'] = str(self.best_train_acc)
        dict['best_test_acc'] = str(self.best_test_acc)
        with open(path, 'w+') as f:
            f.write(json.dumps(dict, cls=NumpyArrayEncoder))
        with open(path+"_description.txt", 'w') as f:
            f.write(self.description)
    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.Y = {}
        self.last_img_id = int(data['last_img_id'])
        self.description = str(data['description'])
        if 'best_train_acc' in data.keys():
            self.best_train_acc = float(data['best_train_acc'])
            self.best_test_acc = float(data['best_test_acc'])
        for key in data['keys'].keys():
            self.Y[key] = list(np.asarray(data['keys'][key]))

class Layer_Type(Enum):
    ZERO = 1
    RANDOM = 2
    EYE = 3

class Layer:
    def __init__(self, _id, _model, input_size, neurons, act_fun, layer_type = Layer_Type.RANDOM, _optimizer = SGDOptimizer()):
        self.id = _id
        self.model = _model
        self.input_size = input_size
        self.neurons = neurons
        self.act_fun = act_fun
        self.is_ending = False
        self.input_layers_ids = []
        self.output_layers_ids = []
        self.f_input = [] # forwarind input
        self.b_input = [] # backwarding input
        self.reshspers = {}
        self.optimizer = _optimizer
        self.optimizer_W = OptimizerFactory.copy(self.optimizer)
        self.optimizer_B = OptimizerFactory.copy(self.optimizer)
        self.connections = {}
        if layer_type == Layer_Type.EYE:
            self.W = np.asarray(eye_stretch(neurons, input_size))
            self.B = np.asarray(np.zeros((neurons, 1)))
        elif layer_type == Layer_Type.ZERO:
            self.W = np.asarray(np.zeros((neurons, input_size)))
            self.B = np.asarray(np.zeros((neurons, 1)))
        else:
            self.W = np.asarray(np.random.randn(neurons, input_size))
            self.B = np.asarray(np.random.randn(neurons, 1))
    
    def get_output_size(self):
        return self.neurons
    
    def connect_input(self, layer_id):
        if layer_id == self.id: 
            print("error I")
            return
        target_layer = self.model.get_layer(layer_id)
        if not layer_id in self.input_layers_ids:
            self.input_layers_ids.append(layer_id)
        if not self.id in target_layer.output_layers_ids:
            target_layer.connect_output(self.id)

    def connect_output(self, layer_id):
        if layer_id == self.id: 
            print("error O")
            print("self.id: "+ str(self.id)+ " layer_id: "+ str(layer_id))
            return
        target_layer = self.model.get_layer(layer_id)
        if not layer_id in self.output_layers_ids:
            self.output_layers_ids.append(layer_id)
        if not self in target_layer.input_layers_ids:
            target_layer.connect_input(self.id)
        # neurons, input_size = self.W.shape
        # self.connections[layer_id] = np.zeros((neurons, input_size))
        # print(f"Connected new layer with ID {layer_id}.")
        
    
    def disconnect(self, to_remove_layer_id):
        if to_remove_layer_id in self.input_layers_ids:
            self.input_layers_ids.remove(to_remove_layer_id)
        if to_remove_layer_id in self.output_layers_ids:
            self.output_layers_ids.remove(to_remove_layer_id)
        if self.id == to_remove_layer_id:
            self.input_layers_ids = []
            self.output_layers_ids = []
        # if to_remove_layer_id in self.connections:
        #     del self.connections[to_remove_layer_id]
        #     print(f"Disconnected layer with ID {to_remove_layer_id}.")
        # else:
        #     print(f"Layer with ID {to_remove_layer_id} not found.")

    def update_weights_shape(self, input_size):
        """
        Sprawdza, czy rozmiar wag `self.W` jest zgodny z rozmiarem wejścia. Jeśli jest za mały, dodaje wiersze zerowe.
        Jeśli jest za duży, obcina nadmiarowe wiersze.
        """
        current_weight_size = self.W.shape[1]  # Liczba kolumn w macierzy wag (rozmiar wejścia dla wag)
        
        if current_weight_size < input_size:
            # Dodajemy brakujące kolumny zerowe, aby dopasować macierz wag do większego wejścia
            extra_columns = input_size - current_weight_size
            zero_padding = np.zeros((self.W.shape[0], extra_columns))
            
            #print("BEFORE self.W.shape: ", self.W.shape)
            self.W = np.hstack([self.W, zero_padding])
            #print("AFTER self.W.shape: ", self.W.shape)
            
            #print("zero_padding: ", zero_padding.shape)
            #print(f"Dodano {extra_columns} kolumn zerowych do macierzy wag.")
        
        elif current_weight_size > input_size:
            # Obcinamy nadmiarowe kolumny z macierzy wag
            self.W = self.W[:, :input_size]
            print(f"Obcięto {current_weight_size - input_size} nadmiarowych kolumn z macierzy wag.")
    
    
    def forward_prop(self, X, deepth = 0):
        self.f_input.append(X)
        if len(self.f_input) < len(self.input_layers_ids): 
            return None
        input_list = []
        for layer_input in self.f_input:
            input_list.append(layer_input)  # Zbiera dane wejściowe
        self.I = np.vstack(input_list)
        self.update_weights_shape(self.I.shape[0])
        self.Z = np.dot(self.W, self.I)
        self.A = self.act_fun.exe(self.Z)
        result = None
        if self.is_ending:
            result = self.A
        else:
            threads=[]
            results=[]
            for layer_id in self.output_layers_ids:
                layer = self.model.get_layer(layer_id)
                if type(layer) == Layer:
                    new_input = Reshape(self.A.copy(), layer.input_size, get_reshsper(self.A.shape[0], layer.input_size))
   
                if threading.active_count() < MAX_THREADS:
                    thread = threading.Thread(
                        target=lambda: results.append(self.model.get_layer(layer_id).forward_prop(new_input, deepth + 1)),
                    )
                    thread.start()
                    threads.append(thread)
                else:
                    print(f"No available threads, continuing in the current thread: {threading.current_thread().name}")
                    r = self.model.get_layer(layer_id).forward_prop(new_input, deepth + 1)
                    results.append(r)
            
            for thread in threads:
                thread.join()

            for recived_result in results:
                if recived_result is not None:
                    if result is None:
                        result = recived_result
                    else:
                        raise ValueError("More than one result has value")
        self.f_input = []
        return result


    def back_prop(self,E,m,alpha):
        if E.shape[0] <=0:
            raise ValueError("Error with 0 shape can't be backpropagated E.shape:", E.shape)
        m = 1.0
        E = Reshape(E, self.neurons, get_reshsper(E.shape[0], self.neurons))
        self.b_input.append(E)
        if len(self.b_input) < len(self.output_layers_ids): return None
        self.E =  clip(mean_n(self.b_input), -error_clip_range, error_clip_range)
        dZ = self.E * self.act_fun.der(self.Z)
        #self.dW = 1 / m * dZ @ self.I.T
        #self.dB = 1 / m * np.reshape(np.sum(dZ, 1), self.B.shape)
        self.dW = Layer.calcuale_dW(m, dZ, self.I)
        self.dB = Layer.calcuale_dB(m, dZ, self.B)
        self.E = self.W.T @ dZ
        before_iteration = 0  # Start index for slicing self.W
        threads=[]
        for layer_id in self.input_layers_ids:
            neurons = self.input_size#self.model.get_layer(layer_id).get_output_size()
            E_slice = self.W[:, before_iteration:before_iteration + neurons].T @ dZ
            before_iteration += neurons
            if threading.active_count() < MAX_THREADS:
                thread = threading.Thread(
                    target=lambda: self.model.get_layer(layer_id).back_prop(E_slice, m, alpha),
                )
                thread.start()
                threads.append(thread)
            else:
                print(f"No available threads, continuing in the current thread: {threading.current_thread().name} count: {threading.active_count()}")
                self.model.get_layer(layer_id).back_prop(E_slice, m, alpha)
        self.update_params(alpha)
        self.b_input = []
        #for thread in threads:
        #    thread.join()

    def update_params(self, alpha):
        self.W = self.optimizer_W.update(self.W, self.dW, alpha)
        self.B = self.optimizer_B.update(self.B, self.dB, alpha)

    @staticmethod
    @jit(nopython=True)
    def calcuale_Z(W, I, B):
        return np.dot(W, I) + B

    @staticmethod
    @jit(nopython=True)
    def calcuale_dW(m, dZ, I):
        return 1 / m * dZ @ I.T

    def calcuale_dB(m, dZ, B):
        return 1 / m * np.reshape(np.sum(dZ, 1), B.shape)
    
    @staticmethod
    @jit(nopython=True)
    def calcuale_updateW(W, alpha, dw):
        return W - alpha * dw
    

    def get_all_childrens_connections(self, deepth = 0):
        conn = []
        for layer_id in self.output_layers_ids:
            conn.append([self.id, layer_id])
            conn += self.model.get_layer(layer_id).get_all_childrens_connections(deepth + 1)
        return conn

    def is_cyclic(self, visited, additional_pair, depth = 0):
        depth += 1
        to_check = self.output_layers_ids.copy()
        if additional_pair[0] == self.id:
            to_check.append(additional_pair[1])
        for layer_id in to_check:
            if layer_id in visited: return True
            visited.append(layer_id)
            layer = self.model.get_layer(layer_id)
            cycle_found = layer.is_cyclic(visited, additional_pair, depth)
            if cycle_found: return True
            visited.remove(layer_id)
        return False
        
    def deepcopy(self):
        copy = Layer(self.id, None, self.input_size, self.neurons, self.act_fun, Layer_Type.RANDOM, self.optimizer.getDense())
        copy.is_ending = self.is_ending
        copy.input_layers_ids = self.input_layers_ids.copy()
        copy.output_layers_ids = self.output_layers_ids.copy()
        copy.f_input = self.f_input.copy()
        copy.b_input = self.b_input.copy()
        copy.W = self.W.copy()
        copy.B = self.B.copy()
        return copy

    def get_weights_summary(self):
        return "W: "+str(np.mean(self.W))+ " B: "+ str(np.mean(self.B))
    
    def get_paint_label(self):
            return str(self.id) + "[" +str(self.input_size)+","+str(self.neurons)+"]"

    def __str__(self):
        return "[<layer: "+ str(self.id)+ " id: " + str(id(self)) + " model id: "+ str(id(self.model))+" in conn: "+ str(len(self.input_layers_ids)) +" out conn: "+ str(len(self.output_layers_ids))+ ">]" 

class Model:
    def __init__(self, input_size, hidden_size, output_size, loss_function = Loss.multiclass_cross_entropy, activation_fun = Activations.Sigmoid, input_paths = 1, _optimizer = SGDOptimizer()):
        self.batch_size = 128
        self.loss_function = loss_function
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_layers = []
        self.avaible_id = 2
        self.activation_fun = activation_fun
        self.input_layers = []#Layer(0, self, input_size, hidden_size, self.activation_fun)
        self.optimizer = _optimizer
        self.output_layer = Layer(1, self, hidden_size, output_size, Activations.SoftMax, Layer_Type.RANDOM, self.optimizer.getDense())
        for i in range(0, input_paths):
            layer_id = "init_"+str(i)
            self.input_layers.append(Layer(layer_id, self, input_size, hidden_size, self.activation_fun, Layer_Type.RANDOM, self.optimizer.getDense()))
            self.add_connection(layer_id, self.output_layer.id)
        self.output_layer.is_ending = True
        if input_paths > 1: self.add_sequential_output_Layer()
        # in testing:
        self.convolution = False
        self.input_shape = None
        self.kernel_size = None
        self.depth = None
        

    def set_convolution_mode(self, input_shape, kernel_size, depth):
        #import convolution
        self.convolution = True
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.depth = depth # amount of kernels
        for i in range(0, len(self.input_layers)):
            layer_id = "init_"+str(i)
            output_layers_ids = self.input_layers[i].output_layers_ids
            for output_layer_id in output_layers_ids:
                self.input_layers[i].disconnect(output_layer_id)
                self.get_layer(output_layer_id).disconnect(self.input_layers[i].id)
                self.input_layers[i] = Conv(layer_id, self, self.input_shape, self.kernel_size, self.depth, self.activation_fun, self.optimizer.getConv())
                self.add_connection(layer_id, output_layer_id)
        #print("self.input_layer: ", self.input_layers[0].output_shape)
        #print("self.input_layer: ", self.input_layers[0].output_flatten)
        #print("hidden_size: ", self.hidden_size)



    def add_res_layer(self, layer_from_id, layer_to_id, layer_type = Layer_Type.ZERO):
        layer_from = self.get_layer(layer_from_id)
        layer_to = self.get_layer(layer_to_id)
        # if type(layer_from) == Conv:
        #     input_size = layer_from.output_flatten
        # elif type(layer_from) == Layer:
        #     input_size = layer_from.neurons
        input_size = layer_from.get_output_size()
        new_layer = Layer(self.avaible_id, self, input_size, layer_to.input_size, self.activation_fun, layer_type, self.optimizer.getDense())
        self.hidden_layers.append(new_layer)
        self.add_connection(layer_from_id, new_layer.id)
        self.add_connection(new_layer.id, layer_to_id)
        self.avaible_id += 1
        return new_layer.id
    
    def add_norm_layer(self, layer_from_id, layer_to_id, layer_type = Layer_Type.RANDOM):
        layer_from = self.get_layer(layer_from_id)
        layer_to = self.get_layer(layer_to_id)
        if type(layer_from) == Conv:
            input_size = layer_from.output_flatten
        elif type(layer_from) == Layer:
            input_size = layer_from.neurons
        new_layer = Layer(self.avaible_id, self, input_size, layer_to.input_size, self.activation_fun, layer_type, self.optimizer.getDense())
        self.hidden_layers.append(new_layer)
        self.add_connection(layer_from_id, new_layer.id)
        self.add_connection(new_layer.id, layer_to_id)
        layer_from.disconnect(layer_to_id)
        layer_to.disconnect(layer_from_id)
        self.avaible_id += 1
        return new_layer.id
    
    def add_sequential_output_Layer(self):
        new_layer = Layer(self.avaible_id, self, self.output_layer.input_size, self.output_layer.input_size, self.activation_fun, Layer_Type.EYE, self.optimizer.getDense())
        self.hidden_layers.append(new_layer)
        input_layers_ids = self.output_layer.input_layers_ids.copy()
        for input_layer_id in input_layers_ids:
            self.add_connection(input_layer_id, new_layer.id)
            layer_from = self.get_layer(input_layer_id)
            layer_from.disconnect(self.output_layer.id)
            self.output_layer.disconnect(layer_from.id)
        self.add_connection(new_layer.id, self.output_layer.id)
        self.avaible_id += 1
        return new_layer.id
    
    def add_conv_norm_layer(self, layer_from_id, layer_to_id, layer_type = Layer_Type.RANDOM):
        layer_from = self.get_layer(layer_from_id)
        layer_to = self.get_layer(layer_to_id)
        if type(layer_to) == Conv:
            i = layer_from.output_shape[0]
            d = clip(layer_to.depth, 1, 3)
            k = clip(layer_to.kernel_size, 1, i)
            new_layer = Conv(self.avaible_id, self, layer_from.output_shape, k, d, self.activation_fun, self.optimizer.getConv())
        elif type(layer_to) == Layer:
            o = layer_to.input_size
            i = layer_from.output_shape[0]
            d = clip(math.floor(o ** ( 1 / 3)), 1, 3)
            k = clip(i + 1 - math.ceil(o ** ( 1 / 3)), 1, i)
            new_layer = Conv(self.avaible_id, self, layer_from.output_shape, k, d, self.activation_fun, self.optimizer.getConv())
        self.hidden_layers.append(new_layer)
        self.add_connection(layer_from_id, new_layer.id)
        self.add_connection(new_layer.id, layer_to_id)
        layer_from.disconnect(layer_to_id)
        layer_to.disconnect(layer_from_id)
        self.avaible_id += 1
        return new_layer.id
    
    def add_conv_res_layer(self, layer_from_id, layer_to_id, layer_type = Layer_Type.ZERO):
        layer_from = self.get_layer(layer_from_id)
        layer_to = self.get_layer(layer_to_id)
        if type(layer_to) == Conv:
            d = layer_to.depth
            i = layer_from.output_shape[0]
            k = clip(layer_to.kernel_size, 1, i)
            new_layer = Conv(self.avaible_id, self, layer_from.output_shape, k, d, self.activation_fun, self.optimizer.getConv())        
        elif type(layer_to) == Layer:
            o = layer_to.input_size
            i = layer_from.output_shape[0]
            d = clip(math.floor(o ** ( 1 / 3)), 1, 3)
            k = clip(i + 1 - math.ceil(o ** ( 1 / 3)), 1, i)
            new_layer = Conv(self.avaible_id, self, layer_from.output_shape, k, d, self.activation_fun, self.optimizer.getConv())
        self.hidden_layers.append(new_layer)
        self.add_connection(layer_from_id, new_layer.id)
        self.add_connection(new_layer.id, layer_to_id)
        self.avaible_id += 1
        return new_layer.id
    
    def add_connection(self, layer_from_id, layer_to_id):
        L_from = self.get_layer(layer_from_id)
        L_from.connect_output(layer_to_id)

    def get_layer(self, id):
        for input_layer in self.input_layers:
            if input_layer.id == id: 
                return input_layer
        if self.output_layer.id == id: 
            return self.output_layer
        for layer in self.hidden_layers:
            if layer.id == id:
                return layer
            
    def get_predictions(A2):
        return argmax(A2, 0)

    def get_loss(self, x, y):
        A = self.forward_prop(x)
        return self.loss_function.exe(one_hot(y) , A)
    
    def get_accuracy(predictions, Y):
        if predictions.shape != Y.shape:
            sys.exit("ERROR, shape of predictions is diffrent than one_hot_Y: " + str(predictions.shape) + " != " + str(Y.shape))
        #test
        #for i in range(0, len(Y)):
        #    print("predicted: ", predictions[i], " real: ", Y[i])
        return np.sum(predictions == Y) / Y.size

    def forward_prop(self, input):
        if not isinstance(input, np.ndarray):
            input = np.array(input)
        #print("len(self.input_layers): ", len(self.input_layers))
        if len(self.input_layers) == 1:
            return self.input_layers[0].forward_prop(input, 0)
        result = None 
        for i in range(0, len(self.input_layers)):
            result = self.input_layers[i].forward_prop(input[i], 0)
        return result
    
    def gradient_descent(self, X, Y, iterations, lr_scheduler, quiet = False, one_hot_needed = True):
        if not isinstance(X, np.ndarray): X = np.array(X)
        if not isinstance(Y, np.ndarray): Y = np.array(Y)
        if one_hot_needed: one_hot_Y = one_hot(Y)
        else: one_hot_Y = Y
        A = []
        index_axis = 0
        if self.convolution:
            if len(self.input_layers) == 1:
                index_axis = 0
            else:
                index_axis = 1
        else:
            if len(self.input_layers) == 1:
                index_axis = 1
            else:
                index_axis = 2
        indexes = list(range(0, X.shape[index_axis]))
        history = History(['accuracy', 'loss'])
        for i in range(iterations + 1):
            current_alpha = lr_scheduler.alpha_scheduler(i, iterations) #alpha * Model.alpha_scheduler(i, iterations)
            loss = 0
            for x_indx_start in range(0, X.shape[index_axis], self.batch_size): #lwn(x)
                batch_indexes = indexes[x_indx_start:(x_indx_start + self.batch_size)]
                A = self.forward_prop(np.take(X, batch_indexes, index_axis))
                E = self.loss_function.der(np.take(one_hot_Y, batch_indexes, 1) , A)
                self.output_layer.back_prop(E, self.batch_size, current_alpha)                
                loss += self.loss_function.exe(np.take(one_hot_Y, batch_indexes, 1) , A)
            random.shuffle(indexes)
            acc = Model.get_accuracy(Model.get_predictions(self.forward_prop(X)),Y)
            history.append('accuracy', acc)
            history.append('loss', loss)
            if i % 1 == 0 and quiet==False:
                print("Epoch: "+ str(i) + " Accuracy: " + str(round(float(acc),3))+ " loss: " + str(round(float(loss),3)) + " lr: " + str(round(float(current_alpha), 3)) +  " threads: " + str(threading.active_count()))
        A = self.forward_prop(X)
        return history.get_last('accuracy'), history

    def evaluate(self, x, y):
        A = self.forward_prop(x)
        return Model.get_accuracy(Model.get_predictions(A),y)

    def remove_layer(self, layer_id, preserve_flow = True):
        if preserve_flow:
            layer_to_remove = self.get_layer(layer_id)
            for input_layer_id in layer_to_remove.input_layers_ids:
                input_layer = self.get_layer(input_layer_id)
                for output_layer_id in layer_to_remove.output_layers_ids:
                    input_layer.connect_output(output_layer_id)
        
        self.output_layer.disconnect(layer_id)
        for input_layer in self.input_layers:
            input_layer.disconnect(layer_id)
        for layer in self.hidden_layers:
            layer.disconnect(layer_id)
        layer_to_remove = self.get_layer(layer_id)
        self.hidden_layers.remove(layer_to_remove)

    def get_all_childrens_connections(self):
        pairs = []
        for input_layer in self.input_layers:
            pairs += input_layer.get_all_childrens_connections()
        return delete_repetitions(pairs)

    def deepcopy(self):
        copy = Model(self.input_size, self.output_size, self.hidden_size)
        copy.input_layers = []
        for input_layer in self.input_layers:
            copy_layer = input_layer.deepcopy()
            copy_layer.model = copy
            copy.input_layers.append(copy_layer)
        for layer in self.hidden_layers:
            copy_layer = layer.deepcopy()
            copy_layer.model = copy
            copy.hidden_layers.append(copy_layer)
        copy.avaible_id = self.avaible_id
        copy.output_layer = self.output_layer.deepcopy()
        copy.output_layer.model = copy
        copy.convolution = self.convolution
        copy.input_shape = self.input_shape
        copy.kernel_size = self.kernel_size
        copy.depth = self.depth
        return copy
    
    def is_cyclic(self, additional_pair):
        return self.input_layer.is_cyclic([], additional_pair)
 
    def is_connected(self, layer_to_del):
        return self.input_layer.is_connected(layer_to_del)

    def get_sequence_connection(self):
        pairs = []
        for layer in self.hidden_layers + self.input_layers:
            for input_id in layer.input_layers_ids:
                pairs.append([input_id, layer.id])
            for input_id in layer.output_layers_ids:
                pairs.append([layer.id, input_id])
        return pairs
    
    def get_state_representation(self):
        #experimental / temporary
        return str(self.get_sequence_connection())
    
    def show_connection_table(self):
        print("connection_table")
        for layer in self.input_layers + [self.output_layer]:
            print("layer.id: ", layer.id, " layer outputs: ", layer.output_layers_ids, " summary: ", layer.get_weights_summary())
        for layer in self.hidden_layers:
            print("layer.id: ", layer.id, " layer outputs: ", layer.output_layers_ids, " summary: ", layer.get_weights_summary())

    def get_connection_table(self):
        result = ""
        for layer in self.input_layers + [self.output_layer]:
            result += "layer.id: " + str(layer.id) + " layer outputs: " + str(layer.output_layers_ids) + " summary: " + str(layer.get_weights_summary())
        for layer in self.hidden_layers:
            result += "layer.id: " + str(layer.id) + " layer outputs: " + str(layer.output_layers_ids) + " summary: " + str(layer.get_weights_summary())
        return result

class Conv(Layer):
    def __init__(self, _id, _model, input_shape, kernel_size, depth, act_fun, _optimizer = SGDOptimizer()):
        self.id = _id
        self.model = _model
        self.act_fun = act_fun
        self.is_ending = False
        self.input_layers_ids = []
        self.output_layers_ids = []
        self.f_input = [] # forwarind input
        self.b_input = [] # backwarding input
        self.input_shape = input_shape
        self.input_height, self.input_width, self.input_depth = input_shape 
        self.input_flatten = int(self.input_height * self.input_width * self.input_depth)
        self.depth = int(depth) 
        self.kernel_size = int(kernel_size)
        self.output_shape = (int(self.input_height-kernel_size+1),int(self.input_width-kernel_size+1), int(depth))  
        self.output_flatten = int(self.output_shape[0] * self.output_shape[1] * self.output_shape[2])
        self.kernels_shape = (int(self.depth), int(self.input_depth), int(kernel_size), int(kernel_size)) 
        self.reshspers = {}
        self.kernels = np.array(numpy.random.randn(*self.kernels_shape) - 0.5)
        self.biases = np.array(numpy.random.randn(*self.output_shape) - 0.5)
        self.optimizer = _optimizer

    def get_output_size(self):
        return self.output_flatten
    
    def get_reshsper(self, size_from, size_to):
        if not (size_from, size_to) in self.reshspers.keys():
            self.reshspers[(size_from, size_to)] = eye_stretch(size_from, size_to)
        return self.reshspers[(size_from, size_to)]

    def forward_prop(self, X, deepth = 0):
        self.f_input.append(X)
        if len(self.f_input) < len(self.input_layers_ids): 
            return None
        self.I = mean_n_conv(self.f_input, self.input_shape)
        self.Z = np.zeros((self.I.shape[0], self.output_shape[0], self.output_shape[1], self.output_shape[2]))
        for img_id in range(0, self.I.shape[0]):
            for i in range(self.depth): 
                for j in range(self.input_depth): 
                    if IS_CUPY:
                        temp1 = correlate(self.I[img_id,:,:,j], self.kernels[i,j])
                        self.Z[img_id,:,:,i] += np.resize(temp1, self.Z[img_id,:,:,i].shape)
                    else:
                        self.Z[img_id,:,:,i] += correlate2d(self.I[img_id,:,:,j], self.kernels[i,j], "valid") 
                self.Z[img_id,:,:,i] += self.biases[:,:,i]
        self.A = self.act_fun.exe(self.Z)
        result = None
        threads=[]
        results=[]
        for layer_id in self.output_layers_ids:
            layer = self.model.get_layer(layer_id)
            if type(layer) == Conv:
                new_input = Resize(self.A.copy(), layer.input_shape)
            elif type(layer) == Layer:
                new_input = Reshape_forward_prop(self.A.copy(), layer.input_size, get_reshsper(self.output_flatten, layer.input_size))         
            
            if threading.active_count() < MAX_THREADS:
                thread = threading.Thread(
                    target=lambda: results.append(layer.forward_prop(new_input.copy(), deepth + 1)),
                )
                thread.start()
                threads.append(thread)
            else:
                print(f"No available threads, continuing in the current thread: {threading.current_thread().name}")
                r = layer.forward_prop(new_input, deepth + 1)
                results.append(r)
            
        for thread in threads:
            thread.join()

        for recived_result in results:
            if recived_result is not None:
                if result is None:
                    result = recived_result
                else:
                    raise ValueError("More than one result has value")            
        self.f_input = []
        return result

    def back_prop(self, E, m, alpha):
        if len(E.shape) <= 2:
            E = Reshape_back_prop(E, self.output_shape, get_reshsper(E[:, 0].shape[0], self.output_flatten))
        else:
            E = Resize(E, self.output_shape)
        self.b_input.append(E)
        if len(self.b_input) < len(self.output_layers_ids): return None
        self.E =  clip(mean_n_conv(self.b_input, self.input_shape), -error_clip_range, error_clip_range)
        dZ = self.E * self.act_fun.der(self.Z)
        self.error = np.zeros((dZ.shape[1], dZ.shape[2], dZ.shape[3]))
        self.kernels_gradient = np.zeros(self.kernels_shape)
        self.input_gradient = np.zeros((self.I.shape[0], self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        for img_id in range(0, self.I.shape[0]):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    if IS_CUPY:
                        temp1 = correlate(self.I[img_id,:,:,j], dZ[img_id,:,:,i])
                        self.kernels_gradient[i,j] += np.resize(temp1, self.kernels_gradient[i,j].shape)
                        temp2 = convolve(dZ[img_id, :, :, i], self.kernels[i,j])
                        self.input_gradient[img_id, :, :, j] += np.resize(temp2, self.input_gradient[img_id, :, :, j].shape)
                    else:
                        self.kernels_gradient[i,j] += correlate2d(self.I[img_id,:,:,j], dZ[img_id,:,:,i], "valid")
                        self.input_gradient[img_id,:,:,j] += convolve2d(dZ[img_id,:,:,i], self.kernels[i,j], "full")
            self.error += dZ[img_id,:,:,:]
        self.kernels_gradient[i,j] /= self.I.shape[0]
        self.input_gradient[img_id,:,:,j] /= self.I.shape[0]
        self.error /= self.I.shape[0]
        for layer_id in self.input_layers_ids:
            if threading.active_count() < MAX_THREADS:
                thread = threading.Thread(
                    target=lambda: self.model.get_layer(layer_id).back_prop(self.input_gradient.copy, m, alpha),
                )
                thread.start()
            else:
                print(f"No available threads, continuing in the current thread: {threading.current_thread().name} count: {threading.active_count()}")
                self.model.get_layer(layer_id).back_prop(self.input_gradient, m, alpha)
        self.update_params(alpha)
        self.b_input = []

    def update_params(self, alpha):
        self.kernels, self.biases = self.optimizer.update(self.kernels, self.kernels_gradient, self.biases, self.error, alpha)
            
    def deepcopy(self):
        copy = Conv(self.id, None, self.input_shape, self.kernel_size, self.depth, self.act_fun, self.optimizer.getConv())
        copy.is_ending = self.is_ending
        copy.input_layers_ids = self.input_layers_ids.copy()
        copy.output_layers_ids = self.output_layers_ids.copy()
        copy.f_input = self.f_input.copy()
        copy.b_input = self.b_input.copy()
        copy.input_shape = self.input_shape
        copy.input_depth = self.input_depth
        copy.input_height = self.input_height
        copy.input_width = self.input_width
        copy.depth = self.depth
        copy.output_shape = self.output_shape
        copy.kernels_shape = self.kernels_shape
        copy.kernels = self.kernels.copy()
        copy.biases = self.biases.copy()
        return copy
    
    def get_weights_summary(self):
        return "k: "+str(np.mean(self.kernels))+ " B: "+ str(np.mean(self.biases))
    
    def get_paint_label(self):
        return str(self.id) + "[" +str(self.input_shape)+","+str(self.output_shape)+"]"

    def __str__(self):
        return "[<layer: "+ str(self.id)+ " id: " + str(id(self)) + " model id: "+ str(id(self.model))+" in conn: "+ str(len(self.input_layers_ids)) +" out conn: "+ str(len(self.output_layers_ids))+ ">]"

    
def Resize(x, shape):
    if x[0].shape == shape:
        return x
    x_resized = np.zeros((x.shape[0], shape[0], shape[1], shape[2]))
    for i in range(0, x.shape[0]):
        x_resize_tmp = strech(x[i], (shape[0], shape[1]))
        index = min([x_resize_tmp.shape[2], shape[2]])
        x_resized[i, :, :, 0:index] = strech(x[i], (shape[0], shape[1]))[:, :, 0:index]
    return x_resized



class Storage:
    
    def __init__(self):
        pass
    

    def loadModel(path):
        with open(path, 'r') as json_file:
            model_dict = json.load(json_file)
        return Storage.dictToModel(model_dict)

    def saveModel(model, path):
        with open(path, 'w') as json_file:
            json.dump(Storage.modelToDict(model), json_file, indent=4)
        pass
    
    def modelToDict(model):
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

    def layerToDict(layer):
        dict_main = {}
        dict_main['id'] = str(layer.id)
        dict_main['act_fun'] = str(layer.act_fun.__name__)
        dict_main['is_ending'] = layer.is_ending
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
            dict_reshapers[reshsper_id]["matrix"] =  get_numpy_array(layer.reshspers[resheper_key]).tolist()
        dict_main['reshapers'] = dict_reshapers
        dict_main['weights'] = {}
        if type(layer) == Layer:
            dict_main['neurons'] = layer.neurons
            dict_main['input_size'] = layer.input_size
            dict_main['weights']['W'] = get_numpy_array(layer.W).tolist()
            dict_main['weights']['B'] = get_numpy_array(layer.B).tolist()
        if type(layer) == Conv:
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
        

    def dictToModel(dict_main):
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


    def dictToLayer(layer_dict, model):
        layer_id = str(layer_dict['id'])
        act_fun_name = layer_dict['act_fun']
        is_ending = layer_dict['is_ending']
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
            layer = Layer(layer_id, model, input_size, neurons, act_fun, Layer_Type.RANDOM,  optimizer.getDense())
        
        layer.is_ending = is_ending
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
