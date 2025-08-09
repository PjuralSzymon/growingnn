import time
import random
from ..action import Action
#from ..structure import *

# Global configuration for logging
ENABLE_LOGGING = False
LOG_DIRECTORY = "./logs"

# def scoreFun(M, epochs, X_train, Y_train, simulation_score):
#     acc, history = M.gradient_descent(X_train, Y_train, epochs, LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.1) , True)
#     return simulation_score.grade(acc, history)
    #return max(1.e-17, max_loss - history.get_last('loss'))

# Global index for tracking actions
global_action_index = 0

async def get_action(M, max_time_for_dec, epochs, X_train, Y_train, simulation_score):
    global global_action_index
    global_action_index += 1
    
    all_actions = Action.generate_all_actions(M)
    size_of_changes = len(all_actions)
    if size_of_changes == 0:
        print("Error no actions avaible")
    best_action = None
    best_score = float("-inf")

    deadline = time.time() + max_time_for_dec
    deepth = 0
    rollouts = 0
    
    # Create log text
    log_text = f"=== Action Analysis #{global_action_index} ===\n"
    log_text += f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_text += f"Model ID: {id(M)}\n"
    log_text += f"Available actions: {size_of_changes}\n"
    log_text += f"Max time for decision: {max_time_for_dec}s\n"
    log_text += f"Epochs: {epochs}\n"
    log_text += f"Simulation score function: {simulation_score.__class__.__name__}\n"
    log_text += f"Deadline: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(deadline))}\n\n"
    
    action_results = []

    while time.time() < deadline:
        if len(all_actions) <= 0: break
        action = random.choice(all_actions)
        new_M = M.deepcopy()
        #new_M.apply_action(action)
        action.execute(new_M)
        all_actions.remove(action)

        score = simulation_score.scoreFun(new_M, epochs, X_train, Y_train)
        
        # Log action details
        action_info = {
            'action_type': action.__class__.__name__,
            'action_params': str(action),
            'score': score,
            'timestamp': time.strftime('%H:%M:%S')
        }
        action_results.append(action_info)
        
        log_text += f"Action #{rollouts + 1}:\n"
        log_text += f"  Type: {str(action)}\n"
        log_text += f"  Score: {score}\n"
        log_text += f"  Time: {action_info['timestamp']}\n\n"
        rollouts += 1
        if score > best_score:
            best_score = score
            best_action = action

    # Final summary
    log_text += "=== FINAL SUMMARY ===\n"
    log_text += f"Total rollouts: {rollouts}\n"
    log_text += f"Best score: {best_score}\n"
    log_text += f"Best action: {best_action}\n"
    log_text += f"Execution time: {time.time() - (deadline - max_time_for_dec):.2f}s\n"
    log_text += f"=== END ANALYSIS #{global_action_index} ===\n\n"
    
    # Save to file only if logging is enabled
    if ENABLE_LOGGING:
        file_path = f"{LOG_DIRECTORY}/action_analysis_{global_action_index:06d}.txt"
        try:
            import os
            import asyncio
            
            # Create directory synchronously (this is usually fast)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Write file asynchronously using thread pool
            async def write_file():
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(log_text)
            await asyncio.to_thread(write_file)
            print(f"Action analysis saved to: {file_path}")
        except Exception as e:
            print(f"Error saving action analysis: {e}")

    return best_action, deepth, rollouts

