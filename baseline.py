import CMGDB
import math
import os
import ast
import argparse
import time
import matplotlib

# def f(x):
#   th1 = 23.5
#   th2 = 23.5
#   return [(th1 * x[0] + th2 * x[1]) * math.exp(-0.1 * (x[0] + x[1])), 0.7 * x[0]]

# def f(x):
#     p1=0.8
#     p2=0.6
#     p3=0.1
#     th1 = 55
#     th2 = 55
#     th3 = 55
#     th4 = 55
#     return [(th1 * x[0] + th2 * x[1] + th3 * x[2] + th4 * x[3]) * math.exp(-0.1 * (x[0] + x[1] + x[2] + x[3])), p1 * x[0], p2 * x[1], p3 * x[2]]

def f(x):
    theta_1 = 28.9 # 26.27 # 21.05 # 19.6
    theta_2 = 29.8 # 23.68 # 24.15 # 45.77 # 23.68
    theta_3 = 22.0 # 21.05 # 26.27 # 24.15 # 23.68
    return [(theta_1 * x[0] + theta_2 * x[1] + theta_3 * x[2]) * math.exp(-0.1 * (x[0] + x[1] + x[2])),
            0.7 * x[0], 0.7 * x[1]]
 

def F(rect):
    return CMGDB.BoxMap(f, rect, padding=False)

class Config:
   def __init__(self, config_fname):
    with open(config_fname) as f:
        config = ast.literal_eval(f.read())
    
        self.output_dir = config['output_dir']
        self.subdiv_min = 36#config['subdiv_min']
        self.subdiv_max = 42#config['subdiv_max']
        self.subdiv_init = 0#config['subdiv_init']
        self.subdiv_limit = 10000#config['subdiv_limit']
        self.lower_bounds = [0, 0, 0]
        self.upper_bounds = [220, 154, 108]

if __name__ == "__main__":

    start_time = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='baseline1.txt')

    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    print(args.config)
    config = Config(config_fname)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    model = CMGDB.Model(config.subdiv_min, config.subdiv_max, config.subdiv_init, config.subdiv_limit, config.lower_bounds, config.upper_bounds, F)

    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)

    ''' This color list and scale factor was used for Fig 4 (a)'''
    # clist = ['#E69F00', '#56B4E9', '#009E73', '#F0E442']
    # scale_factor = [1, 1, 10, 1]

    ''' This color list and scale factor was used for Fig 4 (b)'''
    clist = ['#AA4499', '#E69F00', '#CC79A7', '#56B4E9', '#009E73', '#F0E442']
    scale_factor = [1, 1, 1, 1, 1, 1, 1, 1]   
        
    morse_graph_plot = CMGDB.PlotMorseGraph(morse_graph, cmap=matplotlib.cm.cool)
  #  morse_graph_plot.view('morse_graph')
    morse_graph_plot.render(os.path.join(config.output_dir, 'morse_graph'), format='pdf', view=False, cleanup=False)

    filename1 = os.path.join(config.output_dir, 'morse_sets1')
    filename2 = os.path.join(config.output_dir, 'morse_sets2')
    #CMGDB.PlotMorseSets(morse_graph, clist=clist, xlabel='$x_1$', ylabel='$x_2$', fontsize=20, fig_fname=filename1)
    CMGDB.PlotMorseSets(morse_graph, proj_dims=[0, 2], cmap=matplotlib.cm.cool, xlabel='$x_1$', ylabel='$x_3$', fontsize=20, fig_fname=filename2)
    CMGDB.SaveMorseSets(morse_graph, os.path.join(config.output_dir, 'morse_sets'))

    end_time = time.perf_counter()

    duration_mins = round((end_time - start_time)//60)

    filename = os.path.join(config.output_dir, 'computation_log.txt')
    with open(filename, "w") as f:
        f.write("--- Computation Parameters ---\n")
        f.write(f"Lower bounds: {config.lower_bounds}\n")
        f.write(f"Upper bounds: {config.upper_bounds}\n")
        f.write("------------------------------\n")
        f.write(f"Subdivision init: {config.subdiv_init}\n")
        f.write(f"Subdivision min: {config.subdiv_min}\n")
        f.write(f"Subdivision max: {config.subdiv_max}\n")
        f.write(f"Subdivision limit: {config.subdiv_limit}\n")
        f.write("------------------------------\n")
        f.write(f"Program duration: {duration_mins} minutes")