import random
from ..action import *
from ..structure import *

async def get_action(M, max_time_for_dec, epochs, X_train, Y_train, simulation_score):
    all_action_seq = Action.generate_all_actions(M)
    if len(all_action_seq) == 0:
        print("Error no actions avaible")
    return random.choice(all_action_seq), 0, 0