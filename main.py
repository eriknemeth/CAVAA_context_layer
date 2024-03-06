from run_experiment import *
from mplite import TaskManager, Task
import time

if __name__ == '__main__':
    weights = np.array([0.4, 0.4, 0.2])
    complex_experiment(f'/home/eriknemeth/DATA/A2324_2/test/', '2', weights, replay_threshold=0.1)
    experiment_plotter('/home/eriknemeth/DATA/A2324_2/test/',
                       'environment_2.txt', 'agent_2.csv')