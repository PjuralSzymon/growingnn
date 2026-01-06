import random
from ..action import Action
from ..quaziIdentity import clear_reshepers_cache
#from ..structure import *

async def get_action(M, max_time_for_dec, epochs, X_train, Y_train, simulation_score):
    M.disable_threading = True
    all_action_seq = Action.generate_all_actions(M)
    if len(all_action_seq) == 0:
        print("Error no actions avaible")
    clear_reshepers_cache()
    return random.choice(all_action_seq), 0, 0