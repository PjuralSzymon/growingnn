from abc import abstractmethod
from numba import jit
import numpy as np

from growingnn.config import WEIGHTS_CLIP_RANGE

class Optimizer:
    def __init__(self, weights_clip_range=WEIGHTS_CLIP_RANGE):
        self.weights_clip_range = weights_clip_range
        self.updateType("SGD", "Dense")

    def updateType(self, _optimzierType, _layerType):
        self.optimzierType = _optimzierType
        self.layerType = _layerType
        pass

    @staticmethod
    def clip_and_fix(params, clip_range):
        params = np.clip(params, -clip_range, clip_range)
        params = np.nan_to_num(params, nan=np.nanmean(params))
        return params

    def ToDict(self):
        dict = {}
        dict["optimzierType"] = self.optimzierType
        dict["layerType"] = self.layerType
        dict["weights_clip_range"] = self.weights_clip_range
        return dict
    
    def FromDict(dict):
        return Optimizer(dict["weights_clip_range"])
    
    @abstractmethod
    def update(self, params, grads, alpha):
        raise NotImplementedError("This method should be implemented by subclasses.")

class DenseOptimizer(Optimizer):
    def __init__(self, weights_clip_range=WEIGHTS_CLIP_RANGE):
        super().__init__(weights_clip_range)

    @abstractmethod
    def update(self, params, grads, alpha):
        raise NotImplementedError("This method should be implemented by subclasses.")

class ConvOptimizer(Optimizer):
    def __init__(self, weights_clip_range=WEIGHTS_CLIP_RANGE):
        super().__init__(weights_clip_range)

    @abstractmethod
    def update(self, params, grads, alpha):
        raise NotImplementedError("This method should be implemented by subclasses.")


    
class SGDOptimizer(DenseOptimizer):

    def __init__(self, weights_clip_range=WEIGHTS_CLIP_RANGE):
        super().__init__(weights_clip_range)
        self.updateType("SGD", "Dense")
        
    @staticmethod
    @jit(nopython=True)
    def sgd_update(params, grads, alpha):
         return params.astype(np.float64) - alpha * grads

    def ToDict(self):
        dict = {}
        dict["optimzierType"] = self.optimzierType
        dict["layerType"] = self.layerType
        dict["weights_clip_range"] = self.weights_clip_range
        return dict
    
    def FromDict(dict):
        return SGDOptimizer(dict["weights_clip_range"])
    
    def update(self, params, grads, alpha):
        return SGDOptimizer.clip_and_fix(SGDOptimizer.sgd_update(params, grads, alpha), 
                                         WEIGHTS_CLIP_RANGE)
    
    def getDense(self):
        return self

    def getConv(self):
        return ConvSGDOptimizer(self.weights_clip_range)

class AdamOptimizer(DenseOptimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, weights_clip_range=WEIGHTS_CLIP_RANGE):
        super().__init__(weights_clip_range)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.updateType("Adam", "Dense")

    def ToDict(self):
        dict = {}
        dict["optimzierType"] = self.optimzierType
        dict["layerType"] = self.layerType
        dict["beta1"] = self.beta1
        dict["beta2"] = self.beta2
        dict["epsilon"] = self.epsilon
        dict["weights_clip_range"] = self.weights_clip_range
        return dict
    
    def FromDict(dict):
        return AdamOptimizer(dict["beta1"], dict["beta2"], dict["epsilon"], dict["weights_clip_range"])
    
    @staticmethod
    @jit(nopython=True)
    def adam_update(params, grads, m, v, t, alpha, beta1, beta2, epsilon):
        m = np.copy(beta1 * m + (1 - beta1) * grads)
        v = np.copy(beta2 * v + (1 - beta2) * (grads ** 2))
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params = params - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        return params, m, v

    def update(self, params, grads, alpha):
        if self.m is None or self.v is None or self.m.shape != params.shape:
            reset = True
        else:
            reset = False
        if reset:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        params, self.m, self.v = self.adam_update(
            params, grads, self.m, self.v, self.t, alpha, self.beta1, self.beta2, self.epsilon
        )
        return AdamOptimizer.clip_and_fix(params, self.weights_clip_range)

    def getDense(self):
        return self
    
    def getConv(self):
        return ConvAdamOptimizer(self.beta1, self.beta2, self.epsilon, self.weights_clip_range)
    
class ConvSGDOptimizer(ConvOptimizer):

    def __init__(self, weights_clip_range=WEIGHTS_CLIP_RANGE):
        super().__init__(weights_clip_range)
        self.updateType("SGD", "Conv")

    def ToDict(self):
        dict = {}
        dict["optimzierType"] = self.optimzierType
        dict["layerType"] = self.layerType
        dict["weights_clip_range"] = self.weights_clip_range
        return dict
    
    def FromDict(dict):
        return ConvSGDOptimizer(dict["weights_clip_range"])
    
    @staticmethod
    @jit(nopython=True)
    def conv_sgd_update(kernels, kernel_grads, biases, bias_grads, alpha):
        kernels = kernels - alpha * kernel_grads
        biases = biases - alpha * bias_grads
        return kernels, biases

    def update(self, kernels, kernel_grads, biases, bias_grads, alpha):
        kernels, biases = self.conv_sgd_update(kernels, kernel_grads, biases, bias_grads, alpha)
        kernels = ConvSGDOptimizer.clip_and_fix(kernels, self.weights_clip_range)
        biases = ConvSGDOptimizer.clip_and_fix(biases, self.weights_clip_range)
        return kernels, biases
    
    def getDense(self):
        return SGDOptimizer(self.weights_clip_range)
    
    def getConv(self):
        return self
    
class ConvAdamOptimizer(ConvOptimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, weights_clip_range=WEIGHTS_CLIP_RANGE):
        super().__init__(weights_clip_range)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_kernels = None
        self.v_kernels = None
        self.m_biases = None
        self.v_biases = None
        self.t = 0
        self.updateType("Adam", "Conv")

    def ToDict(self):
        dict = {}
        dict["optimzierType"] = self.optimzierType
        dict["layerType"] = self.layerType
        dict["beta1"] = self.beta1
        dict["beta2"] = self.beta2
        dict["epsilon"] = self.epsilon
        dict["weights_clip_range"] = self.weights_clip_range
        return dict
    
    def FromDict(dict):
        return ConvAdamOptimizer(dict["beta1"], dict["beta2"], dict["epsilon"], dict["weights_clip_range"])
    
    @staticmethod
    @jit(nopython=True)
    def conv_adam_update(kernels, kernel_grads, biases, bias_grads, m_kernels, v_kernels, m_biases, v_biases, t, alpha, beta1, beta2, epsilon):
        # Adam for kernels
        m_kernels = beta1 * m_kernels + (1 - beta1) * kernel_grads
        v_kernels = beta2 * v_kernels + (1 - beta2) * (kernel_grads ** 2)
        m_hat_kernels = m_kernels / (1 - beta1 ** t)
        v_hat_kernels = v_kernels / (1 - beta2 ** t)
        kernels = kernels - alpha * m_hat_kernels / (np.sqrt(v_hat_kernels) + epsilon)
        
        # Adam for biases
        m_biases = beta1 * m_biases + (1 - beta1) * bias_grads
        v_biases = beta2 * v_biases + (1 - beta2) * (bias_grads ** 2)
        m_hat_biases = m_biases / (1 - beta1 ** t)
        v_hat_biases = v_biases / (1 - beta2 ** t)
        biases = biases - alpha * m_hat_biases / (np.sqrt(v_hat_biases) + epsilon)
        
        return kernels, biases, m_kernels, v_kernels, m_biases, v_biases

    def update(self, kernels, kernel_grads, biases, bias_grads, alpha):
        if self.m_kernels is None:
            self.m_kernels = np.zeros_like(kernels)
        if self.v_kernels is None:
            self.v_kernels = np.zeros_like(kernels)
        if self.m_biases is None:
            self.m_biases = np.zeros_like(biases)
        if self.v_biases is None:
            self.v_biases = np.zeros_like(biases)

        self.t += 1
        kernels, biases, self.m_kernels, self.v_kernels, self.m_biases, self.v_biases = self.conv_adam_update(
            kernels, kernel_grads, biases, bias_grads, self.m_kernels, self.v_kernels, self.m_biases, self.v_biases,
            self.t, alpha, self.beta1, self.beta2, self.epsilon
        )
        kernels = ConvAdamOptimizer.clip_and_fix(kernels, self.weights_clip_range)
        biases = ConvAdamOptimizer.clip_and_fix(biases, self.weights_clip_range)
        return kernels, biases
    
    def getDense(self):
        return AdamOptimizer(self.beta1, self.beta2, self.epsilon, self.weights_clip_range)
    
    def getConv(self):
        return self
    
class OptimizerFactory:
    SGD = "SGD"
    Adam = "Adam"

    def __init__(self, weights_clip_range=WEIGHTS_CLIP_RANGE, **kwargs):
        self.weights_clip_range = weights_clip_range
        self.kwargs = kwargs

    def create_optimizer(self, optimizer_type, layer_type):
        if layer_type == "Dense":
            if optimizer_type == OptimizerFactory.SGD:
                return SGDOptimizer(self.weights_clip_range)
            elif optimizer_type == OptimizerFactory.Adam:
                return AdamOptimizer(**self.kwargs)
        elif layer_type == "Conv":
            if optimizer_type == OptimizerFactory.SGD:
                return ConvSGDOptimizer(self.weights_clip_range)
            elif optimizer_type == OptimizerFactory.Adam:
                # Assuming ConvOptimizer can also support Adam-like optimizers, or you could implement a ConvAdamOptimizer class
                return ConvAdamOptimizer(self.weights_clip_range)  # or ConvAdamOptimizer(self.alpha, **self.kwargs)
        else:
            raise ValueError("Unsupported layer type or optimizer type")
        
    def copy(optimizer):
        return OptimizerFactory.FromDict(OptimizerFactory.ToDict(optimizer))
    
    def ToDict(optimizer):
        return optimizer.ToDict()
    
    def FromDict(dict):
        if dict["layerType"] ==  "Conv":
            if dict["optimzierType"] ==  "SGD":
                return ConvSGDOptimizer.FromDict(dict)
            elif dict["optimzierType"] ==  "Adam":
                return ConvAdamOptimizer.FromDict(dict)
        elif dict["layerType"] ==  "Dense":
            if dict["optimzierType"] ==  "SGD":
                return SGDOptimizer.FromDict(dict)
            elif dict["optimzierType"] ==  "Adam":
                return AdamOptimizer.FromDict(dict)    