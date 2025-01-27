from ..helpers import *
from ..action import *
from ..structure import *
#from sklearn.decomposition import PCA

# Simulation set

def create_simulation_set_SAMLE(x,y, amount = 20):
    #set_size_ratio = float(amount) / float(max(np.shape(y)))
    x_sim, y_sim = protected_sampling(x, y, amount)
    #_, x_sim, _, y_sim = train_test_split(x, y, test_size=set_size_ratio)
    return x_sim, y_sim

# def create_simulation_set_PCA(x,y, amount = 10):
#     #print("1 x shape: ", x.shape)
#     #print("22: x: ", x.shape[0], " y: ", y.shape[0])
#     reverse_axes = False
#     if x.shape[0] != y.shape[0]:
#         x = np.swapaxes(x, 0, 1)
#         reverse_axes = True
#         #print("2 x shape: ", x.shape)
#     if y.shape[0] < 100: return create_simulation_set_SAMLE(x,y,amount)
#     if x.shape[0] != y.shape[0]: print("error wrong dimensions !")
#     pca = PCA(amount)
#     classes = {}
#     x_result = []
#     y_result = []
#     for i in range(x.shape[0]):
#         if not y[i] in classes.keys():
#             classes[y[i]] = []
#         classes[y[i]].append(x[i])
#     for c in classes:
#         x_per_class = np.array(classes[c])
#         needed_shape = (amount,) + x_per_class.shape[1:]
#         n = np.prod(x_per_class.shape[1:]) # obliczamy długość wektora spłaszczonego
#         x_per_class_flat = x_per_class.reshape(x_per_class.shape[0], n)
#         x_per_class_flat = x_per_class_flat.T
#         sim_x_per_class_flat = pca.fit_transform(x_per_class_flat)
#         sim_x_per_class = sim_x_per_class_flat.reshape(needed_shape)
#         x_result.append(sim_x_per_class)
#         y_result.append(np.ones((amount,)) * int(c))
#     x_result = np.array(x_result)
#     y_result = np.array(y_result).astype(int)
#     x_result = np.concatenate(x_result, axis=0)
#     y_result = np.concatenate(y_result, axis=0)
#     print("3 x shape: ", x.shape)
#     if reverse_axes:
#         x = np.swapaxes(x, 0, 1)
#         print("4 x shape: ", x.shape)
#         print("x_result: ", x_result.shape)
#         print("y_result: ", y_result.shape)
#         x_result = np.swapaxes(x_result, 0, 1)
#     print("x_result: ", x_result.shape)
#     print("y_result: ", y_result.shape)
#     return x_result, y_result