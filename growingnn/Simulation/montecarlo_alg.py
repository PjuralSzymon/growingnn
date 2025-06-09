import time
import random
import math
import numpy as np
from ..action import Action
#from ..structure import *

UCB1_CONTS = 2
DEEPTH = 2

def protected_divide(a,b):
    LARGE_MAX = 2**128
    if a>LARGE_MAX or b > LARGE_MAX:
        return a//b 
    return a/b

class TreeNode:
    def __init__(self, _parent, _action, _M, _epochs, _X_train, _Y_train, _simulation_score):
        self.parent = _parent
        self.action = _action
        self.X_train = _X_train
        self.Y_train = _Y_train
        self.epochs = _epochs
        self.action = _action
        self.simulation_score = _simulation_score
        self.M = _M
        self.childNodes = []
        self.value = 0
        self.visit_counter = 0

    def expand(self):
#        all_action_seq = self.M.generate_all_possible_new_layers()
        all_action_seq = Action.generate_all_actions(self.M)
        for action in all_action_seq:
            M_copy = self.M.deepcopy()
            action.execute(M_copy)
            #M_copy.add_layer(action[0], action[1])
            new_node = TreeNode(self, action, M_copy, self.epochs, self.X_train, self.Y_train, self.simulation_score)
            self.childNodes.append(new_node)
    
    def rollout(self): 
        M_copy = self.M.deepcopy()
        deepth = DEEPTH
        while deepth > 0:
            all_action_seq = Action.generate_all_actions(M_copy)
            if not all_action_seq:
                break
                
            # Choose action and execute it
            choosen_action = random.choice(all_action_seq)
            choosen_action.execute(M_copy)
            
            # Filter actions more efficiently using list comprehension
            all_action_seq = [action for action in all_action_seq 
                            if action != choosen_action and 
                            not action.can_be_infulenced(choosen_action)]
            deepth -= 1

        return self.simulation_score.scoreFun(M_copy, self.epochs, self.X_train, self.Y_train)

    def get_best_child(self):
        USB1 = lambda node : node.value + UCB1_CONTS*protected_divide(math.log(node.parent.visit_counter),node.visit_counter)
        gradeChild = lambda node : float('inf') if node.visit_counter==0 else USB1(node)
        bestScore = 0
        bestChild = None
        for child in self.childNodes:
            score = gradeChild(child)
            if score > bestScore:
                bestScore = score
                bestChild = child
        return bestChild
        
    def is_leaf(self): 
        return len(self.childNodes)==0

    def get_depth(self):
        res = 0 
        for child in self.childNodes:
            res = max(res,child.get_depth())
        return res + 1

    def kill(self):
        for child in self.childNodes:
            child.kill()
        del self

    def __str__(self):
        result = f"node: value:{self.value} visit_counter: {self.visit_counter}\n"
        for child in self.childNodes:
            result += child.__str__() + "\n"
        return result
    
async def get_action(M, max_time_for_dec, epochs, X_train, Y_train, simulation_score):
    size_of_changes = len(Action.generate_all_actions(M))
    if size_of_changes == 0: 
        print("Error")
        return None, 0, 0
        
    root = TreeNode(None, None, M, epochs, X_train, Y_train, simulation_score)
    deadline = time.time() + max_time_for_dec
    deepth = 0
    rollouts = 0
    while time.time() < deadline or rollouts <= size_of_changes:
        _, deepth, r = simulate(root)
        rollouts += r
    if time.time() > deadline: 
        print("More time was needed to analyze all possibilities at least once")
        
    best_action = root.get_best_child().action
    root.kill()
    return best_action, deepth, rollouts

def simulate(node, deepth = 0, rollouts = 0):
    if node.is_leaf():
        if node.visit_counter==0:
            new_value = node.rollout()
            node.value = new_value
            node.visit_counter += 1
            rollouts += 1
            return node.value, deepth, rollouts
        else:
            node.expand()
    child = node.get_best_child()
    if child == None: return node.value, deepth, rollouts
    new_value, deepth, rollouts = simulate(child, deepth + 1, rollouts)        
    node.value += new_value
    node.visit_counter += 1
    return node.value, deepth, rollouts


