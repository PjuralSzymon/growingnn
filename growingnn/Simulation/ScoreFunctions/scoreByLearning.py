from ...structure import *

def scoreAcc(M, epochs, X_train, Y_train):
    acc, history = M.gradient_descent(X_train, Y_train, epochs, LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.1) , True)
    return acc

def scoreLoss(M, epochs, X_train, Y_train):
    _, history = M.gradient_descent(X_train, Y_train, epochs, LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.1) , True)
    return min(1 / history.get_last('loss'), 1.0)