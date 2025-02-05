import time
import random
from ..action import Action
#from ..structure import *

# def scoreFun(M, epochs, X_train, Y_train, simulation_score):
#     acc, history = M.gradient_descent(X_train, Y_train, epochs, LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.1) , True)
#     return simulation_score.grade(acc, history)
    #return max(1.e-17, max_loss - history.get_last('loss'))

async def get_action(M, max_time_for_dec, epochs, X_train, Y_train, simulation_score):
    all_actions = Action.generate_all_actions(M)
    size_of_changes = len(all_actions)
    if size_of_changes == 0:
        print("Error no actions avaible")
    best_action = None
    best_score = float("-inf")

    deadline = time.time() + max_time_for_dec
    deepth = 0
    rollouts = 0

    while time.time() < deadline:
        if len(all_actions) <= 0: break
        action = random.choice(all_actions)
        new_M = M.deepcopy()
        #new_M.apply_action(action)
        action.execute(new_M)
        all_actions.remove(action)

        score = simulation_score.scoreFun(new_M, epochs, X_train, Y_train)

        if score > best_score:
            best_score = score
            best_action = action

        rollouts += 1

    if time.time() > deadline:
        print("More time was needed to analyze all possibilities at least once")

    return best_action, deepth, rollouts

