from .painter import *
from .structure import *
from .action import *
import asyncio
from .model_storage import *
from .Simulation import *
import os
from .helpers import convert_to_desired_type

def train(x_train, x_test, y_train, y_test, labels, path, model_name, epochs, generations, input_size, hidden_size, output_size, input_shape, kernel_size, deepth, batch_size = 128, simulation_set_size = 20, simulation_alg = montecarlo_alg, sim_set_generator = create_simulation_set_SAMLE, simulation_scheduler = SimulationScheduler(SimulationScheduler.PROGRESS_CHECK, simulation_time = 60, simulation_epochs = 20), lr_scheduler = LearningRateScheduler(LearningRateScheduler.PROGRESIVE, 0.03, 0.8), loss_function = Loss.multiclass_cross_entropy, activation_fun = Activations.Sigmoid, input_paths = 1, sample_sub_generator = None, augmentor = Augmentor(), simulation_score = Simulation_score()):
    x_train = convert_to_desired_type(x_train)
    x_test = convert_to_desired_type(x_test)
    y_train = convert_to_desired_type(y_train)
    y_test = convert_to_desired_type(y_test)
    print("x_train: ", type(x_train))
    print("x_test: ", type(x_test))
    print("y_train: ", type(y_train))
    print("y_test: ", type(y_test))
    if not os.path.exists(path):
        os.mkdir(path, 0o777)
    if not os.path.exists(path): os.mkdir(path, 0o777)
    hist = {'train': [], 'test': []}
    hist_detail = History(['accuracy', 'loss'])
    model_path = path+model_name
    hist_path = path+model_name+"_hist"
    if os.path.isfile(hist_path) and False:
        hist_detail.load(hist_path) 
        print("hist_detail loaded")
    if os.path.isfile(model_path) and False:
        M = model_storage.load_model(model_path)  
        hist_detail.description += 'Reloading model ... \n'
        print("Model was loaded from: ", model_path)
    else:
        M = Model(input_size, hidden_size, output_size, loss_function, activation_fun, input_paths)
        if input_shape != None:
                M.set_convolution_mode(input_shape, kernel_size, deepth)
        M.batch_size = batch_size
    acc = Model.get_accuracy(Model.get_predictions(M.forward_prop(x_test)),y_test)
    print("model is ready, starting accuracy: ", acc)

    sim_x, sim_y = sim_set_generator(x_train, y_train, simulation_set_size)
    draw(M, model_path+ 'cifar_init.html')
    for i in range(0,generations):
        new_acc, new_hist = M.gradient_descent(x_train, y_train, epochs, lr_scheduler, False, True, augmentor)
        hist_detail.merge(new_hist)
        hist_detail.append('iteration_acc_train', new_acc)
        hist_detail.append('iteration_acc_test', M.evaluate(x_test, y_test))
        if simulation_scheduler.can_simulate(i, hist_detail):
            correc_desc = "[iteration: "+str(i)+"] No correction detected acc: " + str(new_acc)+ " starting simulation." 
            hist_detail.description += correc_desc+'\n'            
            loop = asyncio.get_event_loop()
            simulation_score.new_max_loss(hist_detail)
            action, deepth, rollouts = loop.run_until_complete(simulation_alg.get_action(M.deepcopy(), simulation_scheduler.simulation_time, simulation_scheduler.simulation_epochs, sim_x, sim_y, simulation_score))
            size_of_changes = len(Action.generate_all_actions(M))
            action_desc = "[iteration: "+str(i)+"] Best action found after simulation: "+ str(action)+ " deepth of tree searched: "+ str(deepth) + " number of rollouts: "+ str(rollouts) + " size_of_changes: " + str(size_of_changes)
            hist_detail.description += action_desc+"\n"
            action.execute(M)
        #model_storage.save_model(M, model_path)
        hist_detail.save(hist_path)
        draw(M, model_path+'_graph_'+ str(hist_detail.last_img_id)+".html")
        hist_detail.last_img_id += 1
        #helpers.draw_hist(hist, model_path+"_history_global", ".")
        hist_detail.draw_hist(model_path+"_history_detail", ".")
        if hist_detail.get_last('iteration_acc_train') > hist_detail.best_train_acc:
            #model_storage.save_model(M, model_path+"_best_train_acc") # TURN OFF FOR LINUX
            hist_detail.description += '[iteration: '+str(i)+'] Rewriting best model for train acc prev: ' + str(hist_detail.best_train_acc) + " new: " + str(hist_detail.get_last('iteration_acc_train')) + "\n"
            hist_detail.best_train_acc = hist_detail.get_last('iteration_acc_train')
            if sample_sub_generator != None:
                sample_sub_generator(M, model_path+"train_", labels, x_train, y_train, x_test, y_test)
        if hist_detail.get_last('iteration_acc_test') > hist_detail.best_test_acc:
            #model_storage.save_model(M, model_path+"_best_test_acc") # TURN OFF FOR LINUX
            hist_detail.description += '[iteration: '+str(i)+'] Rewriting best model for test acc prev: ' + str(hist_detail.best_test_acc) + " new: " + str(hist_detail.get_last('iteration_acc_test')) + "\n"
            hist_detail.best_test_acc = hist_detail.get_last('iteration_acc_test')
            if sample_sub_generator != None:
                sample_sub_generator(M, model_path+"test_", labels, x_train, y_train, x_test, y_test)
    draw(M, model_path+"_graph.html")
    return M
