from numba import jit
import numpy as np

class Optimizer:
    def __init__(self, alpha, weights_clip_range=1.0):
        self.alpha = alpha
        self.weights_clip_range = weights_clip_range

    @staticmethod
    @jit(nopython=True)
    def clip_and_fix(params, clip_range):
        params = np.clip(params, -clip_range, clip_range)
        params = np.nan_to_num(params, nan=np.nanmean(params))
        params[params == -np.inf] = -1.0
        params[params == np.inf] = 1.0
        return params

    def update(self, params, grads):
        raise NotImplementedError("This method should be implemented by subclasses.")

class SGDOptimizer(Optimizer):
    @staticmethod
    @jit(nopython=True)
    def sgd_update(params, grads, alpha):
        return params - alpha * grads

    def update(self, params, grads):
        params = self.sgd_update(params, grads, self.alpha)
        return self.clip_and_fix(params, self.weights_clip_range)

class AdamOptimizer(Optimizer):
    def __init__(self, alpha, beta1=0.9, beta2=0.999, epsilon=1e-8, weights_clip_range=1.0):
        super().__init__(alpha, weights_clip_range)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    @staticmethod
    @jit(nopython=True)
    def adam_update(params, grads, m, v, t, alpha, beta1, beta2, epsilon):
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params = params - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        return params, m, v

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
        if self.v is None:
            self.v = np.zeros_like(params)

        self.t += 1
        params, self.m, self.v = self.adam_update(
            params, grads, self.m, self.v, self.t, self.alpha, self.beta1, self.beta2, self.epsilon
        )
        return self.clip_and_fix(params, self.weights_clip_range)

class ConvSGDOptimizer(Optimizer):
    @staticmethod
    @jit(nopython=True)
    def conv_sgd_update(kernels, kernel_grads, biases, bias_grads, alpha):
        kernels = kernels - alpha * kernel_grads
        biases = biases - alpha * bias_grads
        return kernels, biases

    def update(self, kernels, kernel_grads, biases, bias_grads):
        kernels, biases = self.conv_sgd_update(kernels, kernel_grads, biases, bias_grads, self.alpha)
        kernels = self.clip_and_fix(kernels, self.weights_clip_range)
        biases = self.clip_and_fix(biases, self.weights_clip_range)
        return kernels, biases

class ConvAdamOptimizer(Optimizer):
    def __init__(self, alpha, beta1=0.9, beta2=0.999, epsilon=1e-8, weights_clip_range=1.0):
        super().__init__(alpha, weights_clip_range)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_kernels = None
        self.v_kernels = None
        self.m_biases = None
        self.v_biases = None
        self.t = 0

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

    def update(self, kernels, kernel_grads, biases, bias_grads):
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
            self.t, self.alpha, self.beta1, self.beta2, self.epsilon
        )
        kernels = self.clip_and_fix(kernels, self.weights_clip_range)
        biases = self.clip_and_fix(biases, self.weights_clip_range)
        return kernels, biases
    
class OptimizerFactory:
    SGD = "SGD"
    Adam = "Adam"

    def __init__(self, alpha, weights_clip_range=1.0, **kwargs):
        self.alpha = alpha
        self.weights_clip_range = weights_clip_range
        self.kwargs = kwargs

    def create_optimizer(self, optimizer_type, layer_type):
        if layer_type == "Dense":
            if optimizer_type == OptimizerFactory.SGD:
                return SGDOptimizer(self.alpha, self.weights_clip_range)
            elif optimizer_type == OptimizerFactory.Adam:
                return AdamOptimizer(self.alpha, **self.kwargs)
        elif layer_type == "Conv":
            if optimizer_type == OptimizerFactory.SGD:
                return ConvOptimizer(self.alpha, self.weights_clip_range)
            elif optimizer_type == OptimizerFactory.Adam:
                # Assuming ConvOptimizer can also support Adam-like optimizers, or you could implement a ConvAdamOptimizer class
                return ConvOptimizer(self.alpha, self.weights_clip_range)  # or ConvAdamOptimizer(self.alpha, **self.kwargs)
        else:
            raise ValueError("Unsupported layer type or optimizer type")


# class DenseLayer:
#     def __init__(self, optimizer):
#         self.weights = np.random.randn(10, 10)
#         self.biases = np.random.randn(10, 1)
#         self.optimizer = optimizer

#     def update_params(self, weight_grads, bias_grads):
#         self.weights = self.optimizer.update(self.weights, weight_grads)
#         self.biases = self.optimizer.update(self.biases, bias_grads)

# class ConvolutionLayer:
#     def __init__(self, optimizer):
#         self.kernels = np.random.randn(3, 3, 3, 3)
#         self.biases = np.random.randn(3)
#         self.optimizer = optimizer

#     def update_params(self, kernel_grads, bias_grads):
#         self.kernels, self.biases = self.optimizer.update(self.kernels, kernel_grads, self.biases, bias_grads)


# # For a dense layer using SGD
# dense_optimizer = SGDOptimizer(alpha=0.01, weights_clip_range=1.0)
# dense_layer = DenseLayer(dense_optimizer)

# # During training
# weight_grads = np.random.randn(10, 10)  # Example gradients
# bias_grads = np.random.randn(10, 1)
# dense_layer.update_params(weight_grads, bias_grads)

# # For a convolutional layer using Adam
# conv_optimizer = ConvOptimizer(alpha=0.001, weights_clip_range=1.0)
# conv_layer = ConvolutionLayer(conv_optimizer)

# # During training
# kernel_grads = np.random.randn(3, 3, 3, 3)  # Example gradients
# bias_grads = np.random.randn(3)
# conv_layer.update_params(kernel_grads, bias_grads)
