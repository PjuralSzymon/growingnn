import math
import sys
from enum import Enum
import json
import threading
import os
import time
from numba import jit
from scipy.signal import correlate2d, convolve2d
from .painter import *
from .config import config, DistributionMode
from .optimizers import *
from .quaziIdentity import *
from .utils.stoppers import BaseStopper, AccuracyStopper, ParameterCountStopper, AccuracyAndReductionStopper
from .utils.loss import Loss
from .utils.activations import Activations
from .utils.lr_scheduler import LearningRateScheduler
from .utils.history import History
from .utils.storage import Storage

# GPU/CuPy functionality removed - using CPU only





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

    def can_simulate(self, i, hist_detail, epochsInGeneration=20, quiet = False):
        new_acc = 0#hist_detail.Y['iteration_acc_train'][-1]
        prev_acc = 0#hist_detail.Y['iteration_acc_train'][-2]
        if len(hist_detail.Y['iteration_acc_train']) > 0: new_acc = hist_detail.Y['iteration_acc_train'][-1]
        if len(hist_detail.Y['iteration_acc_train']) > 1: prev_acc = hist_detail.Y['iteration_acc_train'][-2]
        new_acc = get_numpy_array(new_acc)
        prev_acc = get_numpy_array(prev_acc)
        if self.mode == SimulationScheduler.CONSTANT: 
            if not quiet:
                print("[iteration: "+str(i)+"] Constant frequency od simulaiton acc: " + str(new_acc)+ " starting simulation." )
            return True
        elif self.mode == SimulationScheduler.PROGRESS_CHECK:
            if new_acc - prev_acc < self.progres_delta:
                if not quiet:
                    print("[iteration: "+str(i)+"] No correction detected acc: " + str(new_acc)+ "(prev: " + str(prev_acc) + ") starting simulation."  )
                return True
            else:
                if not quiet:
                    print("[iteration: "+str(i)+"] correction detected acc: " + str(new_acc)+ "(prev: " + str(prev_acc) + ") training continues.")
                return False
        elif self.mode == SimulationScheduler.OVERFIT_CHECK:
            if not hist_detail.learning_capable(epochsInGeneration):
                if not quiet:
                    print("[iteration: "+str(i)+"] Model not learning capable: " + str(new_acc)+ " starting simulation."  )
                return True
            else:
                if not quiet:
                    print("[iteration: "+str(i)+"] Model is learning capable: " + str(new_acc)+ " training continues.")
                return False
        return False
    
    def get_mode_label(self):
        if self.mode == SimulationScheduler.CONSTANT: return "simconstant"
        elif self.mode == SimulationScheduler.PROGRESS_CHECK: return "simprogres"
        elif self.mode == SimulationScheduler.OVERFIT_CHECK: return "simoverfit"




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
        self.is_starting = False
        self.input_layers_ids = []
        self.output_layers_ids = []
        self.f_input = [] # forwarind input
        self.b_input = [] # backwarding input
        self.reshspers = {}
        self.optimizer = _optimizer
        self.optimizer_W = OptimizerFactory.copy(self.optimizer)
        self.optimizer_B = OptimizerFactory.copy(self.optimizer)
        self.connections = {}
        self.size_registry = {}
        if layer_type == Layer_Type.EYE:
            self.W = np.asarray(eye_stretch(neurons, input_size))
            self.B = np.asarray(np.zeros((neurons, 1)))
        elif layer_type == Layer_Type.ZERO:
            self.W = np.asarray(np.zeros((neurons, input_size)))
            self.B = np.asarray(np.zeros((neurons, 1)))
        else:
            if config.WEIGHT_DISTRIBUTION_MODE == DistributionMode.UNIFORM:
                self.W = np.random.uniform(low=-config.WEIGHTS_CLIP_RANGE/3, high=config.WEIGHTS_CLIP_RANGE/3, size=(neurons, input_size))
                self.B = np.random.uniform(low=-config.WEIGHTS_CLIP_RANGE/3, high=config.WEIGHTS_CLIP_RANGE/3, size=(neurons, 1))
            elif config.WEIGHT_DISTRIBUTION_MODE == DistributionMode.NORMAL:
                self.W = np.random.normal(loc=0.0, scale=config.WEIGHTS_CLIP_RANGE/3, size=(neurons, input_size))
                self.B = np.random.normal(loc=0.0, scale=config.WEIGHTS_CLIP_RANGE/3, size=(neurons, 1))
            elif config.WEIGHT_DISTRIBUTION_MODE == DistributionMode.GAMMA:
                # Using shape=2 and scale=config.WEIGHTS_CLIP_RANGE for Gamma distribution.
                self.W = np.random.gamma(shape=2.0, scale=config.WEIGHTS_CLIP_RANGE/3, size=(neurons, input_size))
                self.B = np.random.gamma(shape=2.0, scale=config.WEIGHTS_CLIP_RANGE/3, size=(neurons, 1))
            elif config.WEIGHT_DISTRIBUTION_MODE == DistributionMode.REVERSED_GAUSSIAN:
                # Shifting the mean to negative, and controlling spread with config.WEIGHTS_CLIP_RANGE
                self.W = get_reverse_normal_distribution(config.WEIGHTS_CLIP_RANGE/3, (neurons, input_size))
                self.B = get_reverse_normal_distribution(config.WEIGHTS_CLIP_RANGE/3, (neurons, 1))
            else:
                raise ValueError(f"Unsupported distribution mode: {config.WEIGHT_DISTRIBUTION_MODE}")
        self.W =  np.ascontiguousarray(self.W, dtype=config.FLOAT_TYPE)
        self.B =  np.ascontiguousarray(self.B, dtype=config.FLOAT_TYPE)
            
    def set_as_ending(self):
        self.is_ending = True
        #self.done_event = threading.Event()
    
    def set_as_starting(self):
        self.is_starting = True
        #self.done_event = threading.Event()
        
    def get_output_size(self):
        return self.neurons
    
    def scale_neurons(self, reduce_ratio):
        neurons_reduced_amount = max(1, int(self.neurons * reduce_ratio))
        
        # Store old neuron count for weight adjustment
        old_neurons = self.neurons
        
        self.W = np.dot(self.W.T, get_reshsper(self.neurons, neurons_reduced_amount)).T
        self.B = np.dot(self.B.T, get_reshsper(self.neurons, neurons_reduced_amount)).T
        
        # Adjust weights in output layers that receive input from this layer
        for output_layer_id in self.output_layers_ids:
            output_layer = self.model.get_layer(output_layer_id)
            if output_layer is not None:
                if self.id in output_layer.input_layers_ids:
                    start_pos, end_pos = output_layer.get_weight_matrix_indexes_for_layer_id(self.id)
                    if start_pos == 0 and end_pos == 0:
                        print("ERROR NOT WEIGHT ADJUSTMENT")
                        continue
                    # Adjust the weight matrix using the same scaling logic
                    if hasattr(output_layer, 'W') and output_layer.W is not None:                       
                        weights_before = output_layer.W[:, :start_pos] if start_pos > 0 else None
                        weights_middle = output_layer.W[:, start_pos:end_pos]  # Part corresponding to current layer
                        weights_after = output_layer.W[:, end_pos:] if end_pos < output_layer.W.shape[1] else None

                        scaled_weights_middle = np.dot(weights_middle, get_reshsper(old_neurons, neurons_reduced_amount))
                        
                        # Combine the three parts
                        new_W_parts = []
                        if weights_before is not None:
                            new_W_parts.append(weights_before)
                        new_W_parts.append(scaled_weights_middle)
                        if weights_after is not None:
                            new_W_parts.append(weights_after)
                        output_layer.W = np.hstack(new_W_parts)
                        
                        # Update the input_size of the output layer
                        if hasattr(output_layer, 'input_size'):
                            output_layer.input_size = output_layer.W.shape[1]
        self.neurons = neurons_reduced_amount

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
        if target_layer is None:
            raise ValueError(f"Target layer with ID {layer_id} does not exist in the model")
        if not layer_id in self.output_layers_ids:
            self.output_layers_ids.append(layer_id)
        if not self in target_layer.input_layers_ids:
            target_layer.connect_input(self.id)
    
    def disconnect(self, to_remove_layer_id):
        if to_remove_layer_id in self.input_layers_ids:
            self.input_layers_ids.remove(to_remove_layer_id)
        if to_remove_layer_id in self.output_layers_ids:
            self.output_layers_ids.remove(to_remove_layer_id)
        if self.id == to_remove_layer_id:
            self.input_layers_ids = []
            self.output_layers_ids = []

    @staticmethod
    @jit(nopython=True)
    def update_weights_shape(W, input_size):
        current_weight_size = W.shape[1]
        if current_weight_size < input_size:
            # Dodawanie brakujących kolumn z zerami
            new_W = np.zeros((W.shape[0], input_size))
            new_W[:, :current_weight_size] = W
            return new_W
        elif current_weight_size > input_size:
            # Usuwanie niepotrzebnych kolumn
            return W[:, :input_size]
        return W
    
    def should_thread_forward(self):
        return (threading.active_count() < config.MAX_THREADS and 
                len(self.f_input) + 1 >= len(self.input_layers_ids))
    
    def append_to_f_input(self, X, sender_id):
        if sender_id == -1:
            self.f_input = [X]
            return
            
        if sender_id not in self.input_layers_ids:
            raise ValueError(f"Sender ID {sender_id} is not in the input layers IDs {self.input_layers_ids}")
            
        #print(self.id, "---sender_id: ", sender_id, " X.shape: ", X.shape, " input_layers_ids: ", self.input_layers_ids)
        self.size_registry[sender_id] = X.shape[0]
        # Pre-allocate if needed
        if len(self.f_input) < len(self.input_layers_ids):
            self.f_input.extend([None] * (len(self.input_layers_ids) - len(self.f_input)))
            
        self.f_input[self.input_layers_ids.index(sender_id)] = X
        

    def forward_prop(self, X, sender_id, deepth = 0):
        if X is None:
            raise ValueError("Layed Dense recived None input")
        self.append_to_f_input(X, sender_id)
        if any(x is None for x in self.f_input):
                return None
        
        #Preparing data
        self.I = np.vstack(self.f_input)
        self.W = Layer.update_weights_shape(self.W, self.I.shape[0])
        #Making data contiguous in memory makes NUMBA work faster
        self.I = np.ascontiguousarray(self.I)
        self.B = np.ascontiguousarray(self.B)
        self.W = np.ascontiguousarray(self.W)
        #Computing forward pass
        self.Z = Layer.compute_forward(self.I, self.W, self.B)
        self.A = self.act_fun.exe(self.Z)

        for layer_id in self.output_layers_ids:
            #Reshape calucualted signal to the input size of the next layer
            layer = self.model.get_layer(layer_id)
            new_input = None
            if type(layer) == Layer:
                #new_input = Reshape(self.A.copy(), layer.input_size, get_reshsper(self.A.shape[0], layer.input_size))
                new_input = self.A.copy()
            elif type(layer) == Conv:
                new_input = Resize(self.A.copy(), layer.input_shape)
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

            if new_input is None:
                raise ValueError("Failed to initialize new_input for layer")

            #Forward prop using threads approach
            if layer.should_thread_forward():
                input_copy = new_input.copy()
                thread = threading.Thread(
                    target=lambda input_copy=input_copy: layer.forward_prop(input_copy, self.id, deepth + 1),
                )
                thread.start()
                self.model.forward_threads.append(thread)
            #Forward prop using single thread approach
            else:
                layer.forward_prop(new_input, self.id, deepth + 1)
        self.f_input = []

    def should_thread_backward(self):
        if threading.active_count() >= config.MAX_THREADS:
            return False
        if len(self.b_input) + 1 < len(self.output_layers_ids): 
            return False
        return True
    
    def get_size_registry(self, layer_id):
        if layer_id not in self.size_registry.keys():
            return self.model.get_layer(layer_id).get_output_size()
        return self.size_registry[layer_id]
    
    def back_prop(self,E,m,alpha):
        if E.shape[0] <=0:
            raise ValueError("Error with 0 shape can't be backpropagated E.shape:", E.shape)
        m = 1.0
        E = Reshape(E, self.neurons, get_reshsper(E.shape[0], self.neurons))
        self.b_input.append(E)
        if len(self.b_input) < len(self.output_layers_ids): return None
        self.E =  clip(mean_n(self.b_input), -config.ERROR_CLIP_RANGE, config.ERROR_CLIP_RANGE)
        dZ = self.E * self.act_fun.der(self.Z)
        self.dW = Layer.calcuale_dW(m, dZ, self.I)
        self.dB = Layer.calcuale_dB(m, dZ, self.B)
        self.E = self.W.T @ dZ
        before_iteration = 0
        for layer_id in self.input_layers_ids:
            #neurons = self.input_size
            neurons = self.get_size_registry(layer_id)
            E_slice = self.W[:, before_iteration:before_iteration + neurons].T @ dZ
            before_iteration += neurons
            layer = self.model.get_layer(layer_id)
            if layer.should_thread_backward():
                thread = threading.Thread(
                    target=lambda: layer.back_prop(E_slice.copy(), m, alpha),
                )
                thread.start()
                self.model.bacward_threads.append(thread)
            else:
                #print(f"No available threads, continuing in the current thread: {threading.current_thread().name} count: {threading.active_count()}")
                layer.back_prop(E_slice, m, alpha)
        self.update_params(alpha)
        self.b_input = []
        # if self.is_starting:
        #     self.done_event.set()

    def update_params(self, alpha):
        self.W = self.optimizer_W.update(self.W, self.dW, alpha)
        self.B = self.optimizer_B.update(self.B, self.dB, alpha)

    @staticmethod
    @jit(nopython=True, cache=False)
    def compute_forward(I: config.FLOAT_TYPE, W: config.FLOAT_TYPE, B: config.FLOAT_TYPE):
        """Compute forward pass with optimized array contiguity"""
        Z = np.dot(W, I) + B
        return Z
    
    @staticmethod
    @jit(nopython=True, cache=False)
    def calcuale_Z(W, I, B):
        return np.dot(W, I) + B

    @staticmethod
    @jit(nopython=True, cache=False)
    def calcuale_dW(m, dZ, I):
        return 1 / m * dZ @ I.T

    @staticmethod
    @jit(nopython=True)
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
        if self.is_ending:
            copy.set_as_ending()
        if self.is_starting:
            copy.set_as_starting()
        copy.input_layers_ids = self.input_layers_ids.copy()
        copy.output_layers_ids = self.output_layers_ids.copy()
        copy.f_input = self.f_input.copy()
        copy.b_input = self.b_input.copy()
        copy.W = self.W.copy()
        copy.B = self.B.copy()
        copy.size_registry = self.size_registry.copy()
        return copy

    def get_weights_summary(self):
        return "W: "+str(np.mean(self.W))+ " B: "+ str(np.mean(self.B))
    
    def get_paint_label(self):
        return str(self.id) + "[" +str(self.W.shape[1])+","+str(self.W.shape[0])+"]" + "mean: "+str( round(np.mean(self.W), 2))

    def __str__(self):
        return "[<layer: "+ str(self.id)+ " id: " + str(id(self)) + " model id: "+ str(id(self.model))+" in conn: "+ str(len(self.input_layers_ids)) +" out conn: "+ str(len(self.output_layers_ids))+ ">]" 

    def get_weight_matrix_indexes_for_layer_id(self, target_layer_id):
        """
        Get the start and end indexes in the weight matrix that correspond to a specific layer ID.
        
        Args:
            target_layer_id: The ID of the layer whose weight indexes we want to find
            
        Returns:
            tuple: (start_index, end_index) representing the column range in the weight matrix
        """
        if target_layer_id not in self.size_registry.keys():
            return 0, 0
        #print(f"\n=== DEBUG: get_weight_matrix_indexes_for_layer_id ===")
        #print(f"Current layer ID: {self.id}")
        #print(f"Target layer ID: {target_layer_id}")
        #print(f"Input layers IDs: {self.input_layers_ids}")
        #print(f"W: {self.W.shape}")
        #print(f"B: {self.B.shape}")
        #print("Current type: ", type(self))
        for idin in self.input_layers_ids:
            if idin not in self.size_registry.keys(): 
                continue
            inlayer = self.model.get_layer(idin)
            #print("inlayer: ", idin, " size_registry: ", self.size_registry[idin])
            if type(inlayer) == Conv:
                #print("inlayer: ", idin, " output_flatten: ", inlayer.output_flatten, " input_flatten ", inlayer.input_flatten)
                pass
            else:
                #print("inlayer: ", idin, " W: ", inlayer.W.shape, " B: ", inlayer.B.shape, " output: ", inlayer.get_output_size())
                pass
        
        if target_layer_id not in self.input_layers_ids:
            print(f"ERROR: Layer ID {target_layer_id} is not in the input layers IDs {self.input_layers_ids}")
            raise ValueError(f"Layer ID {target_layer_id} is not in the input layers IDs {self.input_layers_ids}")
        
        # Find the position of the target layer in input_layers_ids
        input_index = self.input_layers_ids.index(target_layer_id)
        #print(f"Input index of target layer: {input_index}")
        
        # Calculate the start position by summing up the output sizes of previous layers
        start_pos = 0
        #print(f"Calculating start position...")
        for i in range(input_index):
            prev_layer_id = self.input_layers_ids[i]
            if prev_layer_id in self.size_registry.keys():
                prev_output_size = self.size_registry[prev_layer_id]
                start_pos += prev_output_size
                #print(f"  Layer {prev_layer_id}: output_size = {prev_output_size}, start_pos = {start_pos}")
            else:
                #print(f"  WARNING: Layer {prev_layer_id} is not in the size_registry!")
                pass
        
        # Calculate the end position
        target_layer = self.model.get_layer(target_layer_id)
        if target_layer is None:
            print(f"ERROR: Target layer with ID {target_layer_id} does not exist")
            raise ValueError(f"Target layer with ID {target_layer_id} does not exist")
        
        target_output_size = target_layer.get_output_size()
        end_pos = start_pos + target_output_size
        #print(f"Target layer {target_layer_id}: output_size = {target_output_size}")
        #print(f"Final indexes: start_pos = {start_pos}, end_pos = {end_pos}")
        #print(f"=== END DEBUG ===\n")
        
        return start_pos, end_pos

class Model:
    def __init__(self, input_size, hidden_size, output_size, loss_function = Loss.multiclass_cross_entropy, activation_fun = Activations.Sigmoid, input_paths = 1, _optimizer = SGDOptimizer(), output_activation_fun = Activations.SoftMax):
        if input_size <= 0:
            raise ValueError("Input size must be positive")
        if hidden_size <= 0:
            raise ValueError("Hidden size must be positive")
        if output_size <= 0:
            raise ValueError("Output size must be positive")
        if input_paths < 1:
            raise ValueError("Number of input paths must be at least 1")
        if loss_function is None:
            raise ValueError("Loss function cannot be None")
        if activation_fun is None:
            raise ValueError("Activation function cannot be None")
        if _optimizer is None:
            raise ValueError("Optimizer cannot be None")
            
        self.batch_size = 128
        self.loss_function = loss_function
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_size = int(self.input_size * 0.2 + self.output_size * 0.8)
        self.hidden_layers = []
        self.avaible_id = 2
        self.activation_fun = activation_fun
        self.input_layers = []
        self.optimizer = _optimizer
        self.output_activation_fun = output_activation_fun
        self.output_layer = Layer(1, self, hidden_size, output_size, self.output_activation_fun, Layer_Type.RANDOM, self.optimizer.getDense())
        self.output_layer.set_as_ending()
        for i in range(0, input_paths):
            layer_id = "init_"+str(i)
            layer = Layer(layer_id, self, input_size, hidden_size, self.activation_fun, Layer_Type.RANDOM, self.optimizer.getDense())
            layer.set_as_starting()
            self.input_layers.append(layer)
            self.add_connection(layer_id, self.output_layer.id)
        if input_paths > 1: self.add_sequential_output_Layer()
        # in testing:
        self.convolution = False
        self.input_shape = None
        self.kernel_size = None
        self.depth = None
        self.forward_threads = []
        self.bacward_threads = []
        
    @property
    def layers(self):
        """Return all layers in the model for compatibility with tests"""
        return self.input_layers + self.hidden_layers + [self.output_layer]
    
    def get_parametr_count(self):
        counter = 0
        for layer in self.hidden_layers + self.input_layers + [self.output_layer]:
            if type(layer) == Conv:
                counter += int(layer.depth) * int(layer.input_depth) * int(layer.kernel_size) * int(layer.kernel_size)
            elif type(layer) == Layer:
                counter += layer.input_size * layer.neurons
        return counter

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

    def is_regression(self):
        return self.output_layer.act_fun == Activations.Linear
    
    def validate_and_adjust_layer_size(self,input_size, output_size):
        if input_size * output_size > config.MAXIMUM_MATRIX_SIZE_FOR_LAYER_ADDITION:
            input_size = self.hidden_size
        if input_size * output_size > config.MAXIMUM_MATRIX_SIZE_FOR_LAYER_ADDITION:
            output_size = math.floor(math.sqrt(config.MAXIMUM_MATRIX_SIZE_FOR_LAYER_ADDITION / input_size))
        if input_size > config.MAXIMUM_MATRIX_DIMENSTION_FOR_LAYER_ADDITION:
            input_size = config.MAXIMUM_MATRIX_DIMENSTION_FOR_LAYER_ADDITION
        if output_size > config.MAXIMUM_MATRIX_DIMENSTION_FOR_LAYER_ADDITION:
            output_size = config.MAXIMUM_MATRIX_DIMENSTION_FOR_LAYER_ADDITION
        return input_size, output_size

    def add_res_layer(self, layer_from_id, layer_to_id, layer_type = Layer_Type.ZERO):
        layer_from = self.get_layer(layer_from_id)
        layer_to = self.get_layer(layer_to_id)
        input_size = layer_from.get_output_size()
        input_size, _ = self.validate_and_adjust_layer_size(input_size, layer_to.input_size)
        #input_size = min(input_size, self.hidden_size)
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
        input_size, _ = self.validate_and_adjust_layer_size(input_size, layer_to.input_size)
        #input_size = min(input_size, self.hidden_size)
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
        # Check if source layer exists
        L_from = self.get_layer(layer_from_id)
        if L_from is None:
            raise ValueError(f"Source layer with ID {layer_from_id} does not exist in the model")
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
        return None
            
    def get_predictions(A2):
        return argmax(A2, 0)

    def get_accuracy(predictions, Y):
        if predictions.shape != Y.shape:
            sys.exit("ERROR, shape of predictions is diffrent than one_hot_Y: " + str(predictions.shape) + " != " + str(Y.shape))
        return np.sum(predictions == Y) / Y.size

    def forward_prop(self, input):
        if input is None:
            raise ValueError("Input is None")
        if not isinstance(input, np.ndarray):
            input = np.array(input)
        if input.size == 0:
            raise ValueError("Input array is empty")
        if len(self.input_layers) == 0:
            raise ValueError("Model has no input layers")

        input = np.ascontiguousarray(input, dtype=config.FLOAT_TYPE)
        self.output_layer.A = None
        self.output_layer.set_as_ending()
        if len(self.input_layers) == 1:
            self.input_layers[0].forward_prop(input, -1,  0)
        else:
            if len(input) != len(self.input_layers):
                raise ValueError(f"Number of input arrays ({len(input)}) does not match number of input paths ({len(self.input_layers)})")
            for i in range(0, len(self.input_layers)):
                self.input_layers[i].forward_prop(input[i], -1,  0)
        
        #self.output_layer.done_event.wait()
        #print("!!!waiting for forward threads: ", threading.active_count())
        for thread in self.forward_threads:
            thread.join()
        self.forward_threads.clear()
        if self.output_layer.A is None:
            for layer in self.hidden_layers:
                print("layer: ", layer.id, " A: ", layer.A is None)
            raise ValueError("After forward prop A on output layer is None")
        return self.output_layer.A

    def back_prop(self,E,m,alpha):
        for i in range(0, len(self.input_layers)):
            self.input_layers[i].set_as_starting()
        self.output_layer.back_prop(E, m, alpha)   
        # for i in range(0, len(self.input_layers)):
        #     self.input_layers[i].done_event.wait()
        for thread in self.bacward_threads:
            thread.join()
        self.bacward_threads.clear()
    
    def gradient_descent(self, X, Y, iterations, lr_scheduler, quiet = False, one_hot_needed = True, path="."):
        if X is None or Y is None:
            raise ValueError("Training data (X) or labels (Y) cannot be None")
        if not isinstance(X, np.ndarray): X = np.array(X)
        if not isinstance(Y, np.ndarray): Y = np.array(Y)
        if X.size == 0 or Y.size == 0:
            raise ValueError("Training data (X) or labels (Y) cannot be empty")
        if iterations <= 0:
            raise ValueError("Number of iterations must be positive")
        if lr_scheduler is None:
            raise ValueError("Learning rate scheduler cannot be None")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        X = np.ascontiguousarray(X, dtype=config.FLOAT_TYPE)
            
        if not self.is_regression():
            if one_hot_needed: 
                one_hot_Y = one_hot(Y)
            else: 
                one_hot_Y = Y
            
        # Determine the correct axis for indexing based on convolution mode
        if self.is_regression():
            index_axis_x = 0
        else:
            index_axis_x = 0 if self.convolution and len(self.input_layers) == 1 else (1 if not self.convolution and len(self.input_layers) == 1 else 2)

        indexes = np.arange(X.shape[index_axis_x])
        
        # Initialize history
        history = History(['accuracy', 'loss'])
        
        # Main training loop
        for i in range(iterations + 1):
            # Get current learning rate
            current_alpha = lr_scheduler.alpha_scheduler(i, iterations)
            
            # Initialize batch metrics
            total_loss = 0          
            correct_predictions = 0 
            total_samples = Y.shape[0]
            
            # Process batches
            for x_indx_start in range(0, X.shape[index_axis_x], self.batch_size):
                # Get batch indexes
                batch_end = min(x_indx_start + self.batch_size, X.shape[index_axis_x])
                batch_indexes = indexes[x_indx_start:batch_end]
                
                # Forward pass
                batch_X = np.take(X, batch_indexes, index_axis_x)
                if self.is_regression():
                    batch_Y = np.take(Y, batch_indexes, 0)
                else:
                    batch_Y = np.take(one_hot_Y, batch_indexes, 1)
                
                # Forward propagation
                A = self.forward_prop(batch_X)
                
                # Calculate error and backpropagate
                E = self.loss_function.der(batch_Y, A)

                #print("E: ", E)
                self.back_prop(E, len(batch_indexes), current_alpha)
                
                # Calculate loss and accuracy
                batch_loss = self.loss_function.exe(batch_Y, A)
                total_loss += batch_loss
                correct_predictions += np.sum(Model.get_predictions(A) == np.argmax(batch_Y, axis=0))
            
            # Shuffle indexes for next iteration
            np.random.shuffle(indexes)

            history.update_training_progress(correct_predictions, total_samples, total_loss, i, current_alpha, quiet)

            if i % config.PROGRESS_PRINT_FREQUENCY == 0 and not quiet:
                print(f"Epoch: {i} Accuracy: {round(float(history.get_last('accuracy')), 3)} loss: {round(float(history.get_last('loss')), 3)} lr: {round(float(current_alpha), 3)} threads: {threading.active_count()}")

        if self.is_regression():
            return history.get_last('loss'), history
        else:   
            return history.get_last('accuracy'), history

    def evaluate(self, x, y):
        A = self.forward_prop(x)
        return Model.get_accuracy(Model.get_predictions(A),y)

    def remove_layer(self, layer_id, preserve_flow = True):
        # Check if the layer exists
        if self.get_layer(layer_id) == None:
            raise ValueError(f"Layer with ID {layer_id} does not exist in the model")
            
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
        copy.loss_function = self.loss_function
        copy.activation_fun = self.activation_fun
        copy.output_activation_fun = self.output_activation_fun
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

    def backward_prop(self, error, batch_size, learning_rate):
        """Alias for back_prop for compatibility with tests"""
        return self.back_prop(error, batch_size, learning_rate)
        
    def to_json(self):
        """Convert model to JSON format for serialization"""
        return Storage.modelToDict(self)
        
    @classmethod
    def from_json(cls, json_data):
        """Create a model from JSON data"""
        return Storage.dictToModel(json_data)
        

class Conv(Layer):
    def __init__(self, _id, _model, input_shape, kernel_size, depth, act_fun, _optimizer = SGDOptimizer()):
        self.id = _id
        self.model = _model
        self.act_fun = act_fun
        self.is_ending = False
        self.is_starting = False
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
        self.optimizer = _optimizer
        self.size_registry = {}
        if config.WEIGHT_DISTRIBUTION_MODE == DistributionMode.UNIFORM:
            # Uniform Distribution: Generate values in range (-1, 1) and shift by -0.5
            self.kernels = np.array(np.random.uniform(low=-1.0, high=1.0, size=self.kernels_shape) - 0.5)
            self.biases = np.array(np.random.uniform(low=-1.0, high=1.0, size=self.output_shape) - 0.5)
        elif config.WEIGHT_DISTRIBUTION_MODE == DistributionMode.NORMAL:
            # Normal Distribution: Generate values from normal distribution and shift by -0.5
            self.kernels = np.array(np.random.normal(loc=0.0, scale=1/3, size=self.kernels_shape) - 0.5)
            self.biases = np.array(np.random.normal(loc=0.0, scale=1/3, size=self.output_shape) - 0.5)
        elif config.WEIGHT_DISTRIBUTION_MODE == DistributionMode.GAMMA:
            # Gamma Distribution: Generate values from Gamma distribution and shift by -0.5
            self.kernels = np.array(np.random.gamma(shape=2.0, scale=1.0, size=self.kernels_shape) - 0.5)
            self.biases = np.array(np.random.gamma(shape=2.0, scale=1.0, size=self.output_shape) - 0.5)
        elif config.WEIGHT_DISTRIBUTION_MODE == DistributionMode.REVERSED_GAUSSIAN:
            # Reversed Gaussian: Generate values from normal distribution, shift by -0.5, and reverse by multiplying by -1
            self.kernels = np.array(get_reverse_normal_distribution(1/3, self.kernels_shape) - 0.5)
            self.biases = np.array(get_reverse_normal_distribution(1/3, self.output_shape) - 0.5)
        else:
            raise ValueError(f"Unsupported distribution mode: {config.WEIGHT_DISTRIBUTION_MODE}")

        

    def get_output_size(self):
        return self.output_flatten
    
    def get_reshsper(self, size_from, size_to):
        if not (size_from, size_to) in self.reshspers.keys():
            self.reshspers[(size_from, size_to)] = eye_stretch(size_from, size_to)
        return self.reshspers[(size_from, size_to)]

    def forward_prop(self, X, sender_id, deepth = 0):
        if X is None:
            raise ValueError("Layed Conv recived None input")
        self.append_to_f_input(X, sender_id)
        if any(x is None for x in self.f_input):
                return None
        # Combine inputs more efficiently
        self.I = mean_n_conv(self.f_input, self.input_shape)
        self.Z = np.zeros((self.I.shape[0], self.output_shape[0], self.output_shape[1], self.output_shape[2]))
        
        for img_id in range(0, self.I.shape[0]):
            for i in range(self.depth): 
                for j in range(self.input_depth): 
                    self.Z[img_id,:,:,i] += correlate2d(self.I[img_id,:,:,j], self.kernels[i,j], "valid") 
                self.Z[img_id,:,:,i] += self.biases[:,:,i]
        
        self.A = self.act_fun.exe(self.Z)
        
        # Process outputs more efficiently
        for layer_id in self.output_layers_ids:
            layer = self.model.get_layer(layer_id)
            if type(layer) == Conv:
                new_input = Resize(self.A.copy(), layer.input_shape)
            elif type(layer) == Layer:
                new_input = Reshape_forward_prop(self.A.copy(), layer.input_size, get_reshsper(self.output_flatten, layer.input_size))         
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")
            
            if layer.should_thread_forward():
                # Create a copy of new_input to avoid the closure issue
                input_copy = new_input.copy()
                thread = threading.Thread(
                    target=lambda input_copy=input_copy: layer.forward_prop(input_copy, self.id, deepth + 1),
                )
                thread.start()
                self.model.forward_threads.append(thread)
            else:
                layer.forward_prop(new_input.copy(), self.id, deepth + 1)
        self.f_input = []

    def back_prop(self, E, m, alpha):
        if len(E.shape) <= 2:
            E = Reshape_back_prop(E, self.output_shape, get_reshsper(E[:, 0].shape[0], self.output_flatten))
        else:
            E = Resize(E, self.output_shape)
        self.b_input.append(E)
        if len(self.b_input) < len(self.output_layers_ids): return None
        self.E =  clip(mean_n_conv(self.b_input, self.input_shape), -config.ERROR_CLIP_RANGE, config.ERROR_CLIP_RANGE)
        dZ = self.E * self.act_fun.der(self.Z)
        self.error = np.zeros((dZ.shape[1], dZ.shape[2], dZ.shape[3]))
        self.kernels_gradient = np.zeros(self.kernels_shape)
        self.input_gradient = np.zeros((self.I.shape[0], self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        for img_id in range(0, self.I.shape[0]):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    self.kernels_gradient[i,j] += correlate2d(self.I[img_id,:,:,j], dZ[img_id,:,:,i], "valid")
                    self.input_gradient[img_id,:,:,j] += convolve2d(dZ[img_id,:,:,i], self.kernels[i,j], "full")
            self.error += dZ[img_id,:,:,:]
        self.kernels_gradient[i,j] /= self.I.shape[0]
        self.input_gradient[img_id,:,:,j] /= self.I.shape[0]
        self.error /= self.I.shape[0]
        for layer_id in self.input_layers_ids:
            layer = self.model.get_layer(layer_id)
            if layer.should_thread_backward():
                thread = threading.Thread(
                    target=lambda: layer.back_prop(self.input_gradient.copy(), m, alpha),
                )
                thread.start()
                self.model.bacward_threads.append(thread)
            else:
                #print(f"No available threads, continuing in the current thread: {threading.current_thread().name} count: {threading.active_count()}")
                layer.back_prop(self.input_gradient, m, alpha)
        self.update_params(alpha)
        self.b_input = []
        #if self.is_starting:
        #    self.done_event.set()
            
    def update_params(self, alpha):
        self.kernels, self.biases = self.optimizer.update(self.kernels, self.kernels_gradient, self.biases, self.error, alpha)
            
    def deepcopy(self):
        copy = Conv(self.id, None, self.input_shape, self.kernel_size, self.depth, self.act_fun, self.optimizer.getConv())
        if self.is_ending:
            copy.set_as_ending()
        if self.is_starting:
            copy.set_as_starting()
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
        return str(self.id) + "[" +str(self.input_shape)+","+str(self.output_shape)+"]" + "mean: "+str( round(np.mean(self.kernels), 2))

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



