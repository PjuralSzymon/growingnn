from .painter import *
from .structure import *
from .action import *
import asyncio
from .Simulation import *
from .Simulation.ScoreFunctions import *
import os
from .helpers import convert_to_desired_type

def train(x_train, x_test, y_train, y_test, labels, path, model_name, epochs, generations, input_size, hidden_size, output_size, input_shape, kernel_size, deepth, batch_size = 128, simulation_set_size = 20, simulation_alg = montecarlo_alg, sim_set_generator = create_simulation_set_SAMLE, simulation_scheduler = SimulationScheduler(SimulationScheduler.PROGRESS_CHECK, simulation_time = 60, simulation_epochs = 20), lr_scheduler = LearningRateScheduler(LearningRateScheduler.PROGRESIVE, 0.03, 0.8), loss_function = Loss.multiclass_cross_entropy, activation_fun = Activations.Sigmoid, input_paths = 1, sample_sub_generator = None, simulation_score = Simulation_score(), optimizer = SGDOptimizer()):
    # Convert data types once at the beginning
    x_train = convert_to_desired_type(x_train)
    x_test = convert_to_desired_type(x_test)
    y_train = convert_to_desired_type(y_train)
    y_test = convert_to_desired_type(y_test)
    
    # Create directory if it doesn't exist (removed redundant check)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    # Initialize history tracking
    hist_detail = History(['accuracy', 'loss'])
    model_path = path + model_name
    hist_path = path + model_name + "_hist"
    
    # Create and configure model
    M = Model(input_size, hidden_size, output_size, loss_function, activation_fun, input_paths, optimizer)
    if input_shape is not None:
        M.set_convolution_mode(input_shape, kernel_size, deepth)
    M.batch_size = batch_size
    
    # Calculate initial accuracy
    acc = Model.get_accuracy(Model.get_predictions(M.forward_prop(x_test)), y_test)
    print(f"Model is ready, starting accuracy: {acc}")
    
    # Generate simulation set once
    sim_x, sim_y = sim_set_generator(x_train, y_train, simulation_set_size)
    
    # Draw initial model
    draw(M, model_path + 'cifar_init.html')
    
    # Pre-allocate event loop for simulations
    # Always create a new event loop to avoid issues with closed loops from previous tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Main training loop
    for i in range(generations):
        # Draw model before generation
        draw(M, model_path + '_graph_' + str(hist_detail.last_img_id) + "bef.html")
        
        # Run gradient descent
        new_acc, new_hist = M.gradient_descent(x_train, y_train, epochs, lr_scheduler, False, True, model_path + "_gen_" + str(i))
        
        # Update history
        hist_detail.merge(new_hist)
        hist_detail.append('iteration_acc_train', new_acc)
        test_acc = M.evaluate(x_test, y_test)
        hist_detail.append('iteration_acc_test', test_acc)
        
        # Check if simulation is needed
        if simulation_scheduler.can_simulate(i, hist_detail):
            # Log simulation start
            hist_detail.description += f"[iteration: {i}] No correction detected acc: {new_acc} starting simulation.\n"
            
            # Run simulation
            action, deepth, rollouts = loop.run_until_complete(
                simulation_alg.get_action(
                    M.deepcopy(), 
                    simulation_scheduler.simulation_time, 
                    simulation_scheduler.simulation_epochs, 
                    sim_x, 
                    sim_y, 
                    simulation_score
                )
            )
            
            # Log simulation results
            size_of_changes = len(Action.generate_all_actions(M))
            hist_detail.description += f"[iteration: {i}] Best action found after simulation: {action} deepth of tree searched: {deepth} number of rollouts: {rollouts} size_of_changes: {size_of_changes}\n"
            
            # Execute the action
            action.execute(M)
        
        # Save model and history
        hist_detail.save(hist_path)
        Storage.saveModel(M, model_path + "epoch_" + str(i) + "save.json")
        
        # Draw model after generation
        draw(M, model_path + '_graph_' + str(hist_detail.last_img_id) + ".html")
        hist_detail.last_img_id += 1
        
        # Draw history
        hist_detail.draw_hist(model_path + "_history_detail", ".")
        
        # Check for best training accuracy
        if hist_detail.get_last('iteration_acc_train') > hist_detail.best_train_acc:
            hist_detail.description += f'[iteration: {i}] Rewriting best model for train acc prev: {hist_detail.best_train_acc} new: {hist_detail.get_last("iteration_acc_train")}\n'
            hist_detail.best_train_acc = hist_detail.get_last('iteration_acc_train')
            if sample_sub_generator is not None:
                sample_sub_generator(M, model_path + "train_", labels, x_train, y_train, x_test, y_test)
        
        # Check for best test accuracy
        if hist_detail.get_last('iteration_acc_test') > hist_detail.best_test_acc:
            hist_detail.description += f'[iteration: {i}] Rewriting best model for test acc prev: {hist_detail.best_test_acc} new: {hist_detail.get_last("iteration_acc_test")}\n'
            hist_detail.best_test_acc = hist_detail.get_last('iteration_acc_test')
            if sample_sub_generator is not None:
                sample_sub_generator(M, model_path + "test_", labels, x_train, y_train, x_test, y_test)
    
    # Draw final model
    draw(M, model_path + "_graph.html")
    
    # Always close the loop we created
    if not loop.is_closed():
        loop.close()
    
    return M
