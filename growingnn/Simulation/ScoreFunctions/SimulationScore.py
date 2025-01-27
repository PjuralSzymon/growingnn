from ...structure import *
from .scoreByLearning import *
from .scoreEfficiency import *
from .scoreGraphStructure import *

class Simulation_score:
    #ACCURACY = 0
    #LOSS = 1
    # def __init__(self, mode = 0):
    #     self.mode = mode

    ScoreFunctions = {
        'weight_acc': scoreAcc,
        'weight_loss': scoreLoss,
        'weight_time': scoreTime,
        'weight_countW': scoreCountWeights,
        'weight_diameter': scoreDiameter,
        'weight_radius': scoreRadius,
        'weight_eccentricity': scoreEccentricity,
        'weight_chromaticN': scoreChromaticNumber,
        'weight_machingN': scoreMatchingNumber,
        'weight_independenceN': scoreIndependenceNumber,
    }
    def __init__(self, 
                 weight_acc =  1.0, 
                 weight_loss = 0.0,
                 weight_time = 0.0,
                 weight_countW = 0.0,
                 weight_diameter = 0.0,
                 weight_radius = 0.0,
                 weight_eccentricity = 0.0,
                 weight_chromaticN = 0.0,
                 weight_machingN = 0.0,
                 weight_independenceN = 0.0):
        self.weights = {}
        self.weights['weight_acc'] = weight_acc
        self.weights['weight_loss'] = weight_loss
        self.weights['weight_time'] = weight_time
        self.weights['weight_countW'] = weight_countW
        self.weights['weight_diameter'] = weight_diameter
        self.weights['weight_radius'] = weight_radius
        self.weights['weight_eccentricity'] = weight_eccentricity
        self.weights['weight_chromaticN'] = weight_chromaticN
        self.weights['weight_machingN'] = weight_machingN
        self.weights['weight_independenceN'] = weight_independenceN

        #     self.mode = mode
    # def new_max_loss(self, global_history):
    #     self.max_loss = numpy.max(get_list_as_numpy_array(global_history.Y['loss']))
        
    # def grade(self, acc, history):
    #     if self.mode == Simulation_score.ACCURACY:
    #         return acc
    #     else:
    #         return max(1.e-17, self.max_loss - history.get_last('loss'))
            
    def weightSum(self):
        sum = 0.0
        for key in self.weights.keys():
            sum += self.weights[key]
        return sum

    def scoreFun(self, M, epochs, X_train, Y_train):
        score = 0.0
        for score_weight in Simulation_score.ScoreFunctions.keys():
            if self.weights[score_weight] > 0.0:
                score  += self.weights[score_weight] * Simulation_score.ScoreFunctions[score_weight](M, epochs, X_train, Y_train)
        return score / self.weightSum()