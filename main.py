from run_experiment import *
from mplite import TaskManager, Task
import time

if __name__ == '__main__':
    spatial_navigation()
    experiment_plotter('./savedata', 'environment.txt', 'SORB_agent.csv')