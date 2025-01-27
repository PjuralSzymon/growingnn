import time
import random
import math
from ..action import *
from ..structure import *

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

    # def scoreFun(M, epochs, X_train, Y_train, simulation_score):
    #     #print("X_train: ", X_train.shape)
    #     #print("Y_train: ", Y_train.shape)
    #     acc, history = M.gradient_descent(X_train, Y_train, epochs, LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.1) , True)
    #     #print("loss in MCTS: ", history.get_last('loss'), " off: ", max_loss - history.get_last('loss'))
    #     #return max(1.e-17, max_loss - history.get_last('loss'))
    #     return simulation_score.grade(acc, history)
    
    def rollout(self): 
        M_copy = self.M.deepcopy()
        #all_action_seq = M_copy.generate_all_possible_new_layers()
        
        #Action.generate_all_actions(self.M)
        deepth = DEEPTH
        while deepth > 0:
            all_action_seq = Action.generate_all_actions(M_copy)
            if len(all_action_seq) == 0: break
            choosen_action = random.choice(all_action_seq)
            choosen_action.execute(M_copy)
            #M_copy.add_layer(action_seq[0], action_seq[1])
            all_action_seq.remove(choosen_action)
            new_action_seq = []
            for check_action in all_action_seq:
                if check_action.can_be_infulenced(choosen_action) == False:
                    new_action_seq.append(check_action)
            all_action_seq = new_action_seq
            deepth -= 1

        score = self.simulation_score.scoreFun(M_copy, self.epochs, self.X_train, self.Y_train)
        return score

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
        result = "node: value:"+ str(self.value) + " visit_counter: "+str(self.visit_counter)+ "\n"
        for child in self.childNodes:
            result += child.__str__()+ "\n"
        return result
    
async def get_action(M, max_time_for_dec, epochs, X_train, Y_train, simulation_score):
    size_of_changes = len(Action.generate_all_actions(M))
    #size_of_changes = len(M.generate_all_possible_new_layers())
    if size_of_changes == 0: 
        print("Error")
        return None,0
    root = TreeNode(None, None, M, epochs, X_train, Y_train, simulation_score)
    deadline = time.time() + max_time_for_dec
    deepth = 0
    rollouts = 0
    while time.time() < deadline or rollouts <= size_of_changes:
        _, deepth, r = simulate(root)
        rollouts += r
    if time.time() > deadline: 
        print("More time was needed to analise all possiblites at least once")
    best_action = root.get_best_child().action
    root.kill()
    return best_action, deepth, rollouts

def simulate(node, deepth = 0, rollouts = 0):
    if node.is_leaf():
        if node.visit_counter==0:
            new_value = node.rollout()
            #print("deepth: ", deepth, " rollout score: ", new_value, )
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


