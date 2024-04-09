from run_experiment import *
import time

if __name__ == '__main__':
    spatial_navigation()
    experiment_plotter('./savedata', 'environment.txt', 'SORB_agent.csv', 'SEC_agent.csv')