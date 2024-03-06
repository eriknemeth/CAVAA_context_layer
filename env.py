import matplotlib
import numpy as np
from typing import Tuple
import random
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import math
import seaborn as sns
import re

import SORB_agent


class Env:
    """
    The environment class will contain all information regarding the environment, as well as it will be responsible for
    the generation of observations for the agent by reacting to the actions of the former.

    The environment is created independently of the agent, thus it can be easily replaced by a different class, or
    further subclasses may be introduced with ease, as long as the communication between it and the agent remains
    undisturbed.
    """

    def __init__(self, **kwargs):
        """
        General constructor of the environment, simply defining the most basic elements of it
        """
        # The encoding of the entire action space (all possible actions, m dimensions)
        self._act = np.array([])
        # The encoding of the maze -- 0 represents a wall, a number represents an available state
        self._maze = np.array([])
        # What is the value of each state defined above
        self._reward = np.array([])
        # What is the likelihood of getting a reward in each state
        self._reward_prob = np.array([])
        # Where te agent is right now
        self._agent_pos = np.array([])
        # Are there any forbidden actions (following the coding of _act) -- for every state specify a vector of
        # length m. 0 means no restriction, 1 means forbidden action in said state
        self._restrict = np.array([])
        # Probability of slipping (stochastic transitioning)
        self._slip_prob = 0
        # About storing the data
        self._save_env = False
        self._events = None
        return

    # Hidden functions for several upkeep purposes

    def __restrict_walls__(self) -> None:
        """
        Restricts bumping into a wall, let that be explicit 0 or just out of bounds
        """
        for x in range(self._maze.shape[0]):
            for y in range(self._maze.shape[1]):
                if self._maze[x, y] >= 0:
                    for a in range(len(self._act)):
                        [x_prime, y_prime] = self.__next_state__(x, y, a)
                        if self.__check_out_of_bounds__(x_prime, y_prime) or self._maze[x_prime, y_prime] == -1:
                            self._restrict[x, y, a] = 1

    def __check_out_of_bounds__(self, x: int, y: int) -> bool:
        """
        See if an (x, y) coordinate pair is out of bounds on the map, or is if a forbidden filed (i.e. wall)
        :param x: x coordinate
        :param y: y coordinate
        :return: we are out of bounds (True) or not (False)
        """
        if x < 0 or x >= self._maze.shape[0] or y < 0 \
                or y >= self._maze.shape[1] or self._maze[x, y] == -1:
            return True
        return False

    def __next_state__(self, x: int, y: int, a: int) -> np.ndarray:
        """
        Tells us the label and the coordinates of the next state if we take action a in state s (stays in s if the
        action is impossible)
        :param x: the x coordinate we're in
        :param y: the y coordinate we're in
        :param a: the action chosen
        :return: the coordinates of the arrival state
        """
        [x_prime, y_prime] = np.array([x, y]) + self._act[a]
        return np.array([x_prime, y_prime]).astype(int)

    def __slip__(self, x: int, y: int, x_prime: int, y_prime: int) -> np.array:
        """
        In case of a slippery maze, this function will implement how slip should happen. The basic idea is that we take
        a non-forbidden step from s_prime that doesn't lead us back to s. Watch out, this can happen recursively!
        :param x, y: starting state
        :param x_prime, y_prime: arrival state before slipping
        :return: arrival state after slipping in coordinates
        """
        # First let's decide if we slip or not
        if np.random.uniform(0, 1) >= self._slip_prob:
            return np.array([x_prime, y_prime]).astype(int)

        a_poss = self.possible_moves(self._maze[x_prime, y_prime])
        a_poss_filt = np.copy(a_poss)  # This is the actual a_poss without the action(s) that'd take us back

        # Getting rid of a move that would possibly take us back
        for a in a_poss:
            if np.all(np.array([x, y]) == self.__next_state__(x_prime, y_prime, a)):
                np.delete(a_poss_filt, a_poss_filt == a)

        # Taking the random step
        a = np.random.choice(a_poss_filt)
        [x_fin, y_fin] = self.__next_state__(x_prime, y_prime, a)

        # If we were to go out of bounds, or bump into a wall, stay in place instead:
        if self.__check_out_of_bounds__(x_fin, y_fin):
            x_fin, y_fin = x_prime, y_prime

        # And then slip on recursively
        [x_fin, y_fin] = self.__slip__(x_prime, y_prime, x_fin, y_fin)

        return np.array([x_fin, y_fin]).astype(int)

    def __save_step__(self) -> None:
        """
        Saves the current state of the maze by adding a row to the _events memory.
        :return:
        """
        if not self._save_env:
            return

        # 1) Which step are we at
        step = 0
        if self._events.shape[0] > 0:
            step = self._events.at[len(self._events) - 1, 'iter'] + 1

        # 2) Format the event to store (we might have more reward columns than needed)
        # iter, agent_pos_x, agent_pos_y, rew0_pos_x, rew0_pos_y, rew0_val, rew0_proba, ...
        event = {'iter': [step]}
        agent = np.argwhere(self._agent_pos == 1)
        event['agent_pos_x'] = [agent[0, 0]]
        event['agent_pos_y'] = [agent[0, 1]]
        rewards = np.argwhere(self._reward > 0)
        for rew_idx in range(rewards.shape[0]):
            event[f'rew{rew_idx}_pos_x'] = [rewards[rew_idx, 0]]
            event[f'rew{rew_idx}_pos_y'] = [rewards[rew_idx, 1]]
            event[f'rew{rew_idx}_val'] = [self._reward[rewards[rew_idx, 0], rewards[rew_idx, 1]]]
            event[f'rew{rew_idx}_proba'] = [self._reward_prob[rewards[rew_idx, 0], rewards[rew_idx, 1]]]

        # 3) Add it to the table
        events_temp = pd.DataFrame.from_dict(event).fillna(value=np.nan)
        self._events = self._events.copy() if events_temp.empty \
            else events_temp.copy() if self._events.empty \
            else pd.concat([self._events, events_temp], ignore_index=True)
        # If we want to evade a FutureWarning about a possible concatenation between an empty and a non-empty table we
        # need to check for all possibilities
        return

    def __overwrite_step__(self, x: int, y: int) -> None:
        """
        Overwrites the last stored memory in case the agent was moved (teleported) without a step having taken place.
        :param x: New x coordinate of the agent
        :param y: New y coordinate of the agent
        :return:
        """
        if self._save_env:
            self._events.at[len(self._events) - 1, 'agent_pos_x'] = x
            self._events.at[len(self._events) - 1, 'agent_pos_y'] = y
        return

    # Getters that will communicate towards the agent

    def state_num(self) -> int:
        """
        Returns the number of total states possible in the environment, where each state means a separate location in
        the maze

        :return: number of possible states
        """
        return np.max(self._maze) + 1

    def act_num(self) -> int:
        """
        Returns the maximum number of possible actions within the maze for any state.

        :return: max number of possible actions
        """
        return len(self._act)

    def max_rew(self, **kwargs) -> float:
        """
        Returns the maximal obtainable reward from the maze. If 'iter' is defined, it returns the same for a past
        iteration.
        :param kwargs:
            iter: what past iteration we want to know the max reward of
        :return: max of reward
        """
        it = kwargs.get('iter', None)
        if it is not None:
            if it not in self._events['iter'].values:
                raise ValueError(f'Step {it} has not yet been taken')
            curr_row = self._events.loc[self._events['iter'] == it, :].reset_index(drop=True)
            max_rew = 0
            rew_idx = 0
            while f'rew{rew_idx}_val' in self._events.columns:
                if curr_row.at[0, f'rew{rew_idx}_val'] > max_rew:
                    max_rew = curr_row.at[0, f'rew{rew_idx}_val']
                rew_idx += 1
            return max_rew
        return np.max(self._reward)

    # Communication towards the agent

    def curr_state(self) -> int:
        """
        Returns the current state (as per understood by the agent) of the agent

        :return: current state of the agent
        """
        return self._maze[self._agent_pos.astype(bool)][0]

    def possible_moves(self, s: int) -> np.ndarray:
        """
        Given a state of query, it computes all the possible available actions, taking into consideration whether
        movement is restricted or not, and whether the agent can try and bump into walls or not

        :param s: current state label, as understood by the agent
        :return: a numpy array of possible actions to choose from (labels follow those of self._act)
        """

        [x, y] = np.argwhere(self._maze == s)[0]
        moves = np.array(range(len(self._act)))
        moves = moves[~self._restrict[x, y, :].astype(bool)]
        return moves.astype(int)

    # And receiving communication from the agent

    def step(self, s: int, a: int) -> Tuple[int, float]:
        """
        Performs a step from state s (as per designated by the agent), taking action a (as per chosen in advance), and
        returns the observed outcome.
        If the action would drive the agent out of bounds or into the wall, the agent stays in place
        If the environment is slippery, the agent might slip (potentially recursively)
        Every time we take a step, the environment's memory is updated.

        :param s: state label, as per understood by the agent
        :param a: action label, following the indexing of self._act
        :return: new state label as per understood by the agent (int), and corresponding reward (float)
        """
        # Let's remove the agent from the starting state
        [x, y] = np.argwhere(self._maze == s)[0]
        self._agent_pos[x, y] = 0

        # Then see where we land
        [x_prime, y_prime] = self.__next_state__(x, y, a)

        # If we were to go out of bounds, or bump into a wall, stay in place instead:
        if self.__check_out_of_bounds__(x_prime, y_prime):
            x_prime, y_prime = x, y

        # Then we might slip (the function will have no effect if slip_prob == 0)
        [x_prime, y_prime] = self.__slip__(x, y, x_prime, y_prime)

        # Arriving at our final destination in the environment
        s_prime = self._maze[x_prime, y_prime]
        self._agent_pos[x_prime, y_prime] = 1

        # Generating reward
        # TODO stepping onto our out of the rewarding stare?
        rew = 0
        if random.uniform(0, 1) < self._reward_prob[x_prime, y_prime]:  # for any act leading into the rewarded state
            rew = self._reward[x_prime, y_prime]
        # if random.uniform(0, 1) < self._reward_prob[x, y]:  # for any act leading out of the rewarded state
        #     rew = self._reward[x, y]
        #     s_prime = self._maze[x, y]

        # Saving
        self.__save_step__()
        return s_prime, rew

    def place_reward(self, reward_state: int, reward_val: float, reward_prob: float) -> None:
        """
        Places the reward.
        :param reward_state: Where this reward should be placed (state-space representation)
        :param reward_val: Value of this reward
        :param reward_prob: Probability of said reward
        :return: -
        """
        [x, y] = np.argwhere(self._maze == reward_state)[0]
        # Where is the reward and how big is it?
        self._reward[x, y] = reward_val
        # What is the likelihood of getting a reward
        self._reward_prob[x, y] = reward_prob

        # Call the toggle_save, because in case we are saving, adding a new reward means we need to extend the storage
        self.toggle_save(save_on=self._save_env)
        return

    def reset_reward(self) -> None:
        """
        Resets the reward to zero.

        :return: -
        """
        self._reward = np.zeros(self._maze.shape)
        self._reward_prob = np.zeros(self._maze.shape)
        return

    def place_agent(self, init_state: int) -> None:
        """
        A function to place the agent onto state init_state. If saving is on this function will overwrite the location
        of the agent in the last row of the memory.
        :param init_state: the state (understood by the agent) where we should be placed
        """
        [x, y] = np.argwhere(self._maze == init_state)[0]

        # Remove the agent from its old position
        self._agent_pos = np.zeros(self._maze.shape)
        # And then place it onto the new
        self._agent_pos[x, y] = 1

        # Take care of saving by overwriting the last element
        self.__overwrite_step__(x, y)
        return

    # About saving
    def toggle_save(self, **kwargs) -> None:
        """
        Toggles save. If the environment was saving its status so far, it sops doing so. Otherwise, it begins to do so,
        by already storing a snapshot of the current state as well.
        If a new reward has been added recently, we'll increase the size of the memory to accomodate it.
        :param kwargs:
            save_on: If instead of toggling, we want to make sure to turn it on [True] or off [False], we can
        :return:
        """
        save_on = kwargs.get('save_on', not self._save_env)
        if save_on:
            try:
                if f'rew{np.sum(self._reward > 0) - 1}_pos_x' not in self._events.columns:  # We added a new reward
                    for rew_idx in range(np.sum(self._reward > 0)):
                        if f'rew{rew_idx}_pos_x' not in self._events.columns:
                            self._events[f'rew{rew_idx}_pos_x'] = np.full((self._events.shape[0], 1), np.nan)
                            self._events[f'rew{rew_idx}_pos_y'] = np.full((self._events.shape[0], 1), np.nan)
                            self._events[f'rew{rew_idx}_val'] = np.full((self._events.shape[0], 1), np.nan)
                            self._events[f'rew{rew_idx}_proba'] = np.full((self._events.shape[0], 1), np.nan)
            except AttributeError:  # There is no such thing as _events yet
                col_names = [f'rew{variable_num}_{variable_name}' for variable_num in range(np.sum(self._reward > 0))
                             for variable_name in ['pos_x', 'pos_y', 'val', 'proba']]
                self._events = pd.DataFrame(columns=['iter', 'agent_pos_x', 'agent_pos_y', *col_names])
            if not self._save_env:
                self._save_env = True
                self.__save_step__()
        else:
            self._save_env = False

    def dump_env(self, **kwargs) -> None:
        """
        Saves everything that we have stored into 2 different files: one for the environment, and one for the events.
        :param kwargs:
            path: [str] the path to save the document. If no path is defined then the current working folder will be
                used
            label: [str] an additional label to add at the end of the output file name.
        :return:
        """
        path = kwargs.get('path', None)
        if path is not None:
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                os.mkdir(path)
        else:
            path = './'
        label = kwargs.get('label', None)
        if label is not None:
            label = f'_{label}'
        else:
            label = ''

        # 1) Save the whole environment
        file = open(f'{path}environment{label}.txt', 'wb')
        pickle.dump(self.__dict__, file, 2)
        file.close()

        # 2) Save the events
        # try:
        #     self._events.to_csv(f'{path}environment{label}.csv', sep=',', index=False, encoding='utf-8')
        # except AttributeError:
        #     print('Note: This environment does not store the transpired events, no .csv generated.')

    def load_env(self, file_name: str, **kwargs):
        """
        Loads a previously saved environment
        :param file_name: the name of the environment file [txt]
        :param kwargs:
            path: path to the file. If nothing is specified we'll be looking in the working folder
        :return:
        """
        path = kwargs.get('path', None)
        if path is not None:
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                raise FileNotFoundError(f'No directory named {path}')
        else:
            path = './'

        if os.path.isfile(f'{path}{file_name}'):
            file = open(f'{path}{file_name}', 'rb')
            tmp_dict = pickle.load(file)
            file.close()
            self.__dict__.update(tmp_dict)
        else:
            raise FileNotFoundError(f'No file named {file_name}')


class DTMaze(Env):
    """
    A child class to env, where we can specify different mazes, without losing the functions already used in the parent
    class. The maze and all of its properties will have to be initialized as matrices contained in np arrays

    DT Maze stands for double T maze
    """

    def __init__(self, **kwargs):
        """
        Constructor of the double T maze class

        :param kwargs:  forbidden_walls -- is the agent allowed to choose to bump into the wall
                        restricted_dT   -- creates a restricted double-T maze where we can only walk in one direction
                        slip_prob       -- the probability of slipping after a step
        """
        # Handling the potential kwargs
        forbidden_walls = kwargs.get('forbidden_walls', False)
        restricted_dt = kwargs.get('restricted_dT', False)
        slip_prob = kwargs.get('slip_prob', 0)

        # Setting everything up so that we have a double-T maze
        Env.__init__(self)
        # The encoding of the possible actions: {0: up, 1: right, 2: down, 3: left}
        self._act = np.array([np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])])
        # The maze itself
        self._maze = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                               [9, -1, -1, -1, 10, -1, -1, -1, 11],
                               [12, -1, 13, 14, 15, -1, -1, -1, 16],
                               [17, -1, -1, 18, -1, -1, -1, -1, 19],
                               [20, -1, -1, 21, -1, -1, -1, -1, 22],
                               [23, 24, 25, 26, 27, 28, 29, 30, 31]])
        # Transitions
        self._slip_prob = slip_prob
        # Where is the reward
        self._reward = np.zeros(self._maze.shape)
        # What is the likelihood of getting a reward
        self._reward_prob = np.zeros(self._maze.shape)
        # Where do we usually start from
        self._agent_pos = np.zeros(self._maze.shape)
        # Are there any forbidden actions (following the coding of _act)
        if restricted_dt:
            # In this case I want to simply test what happens if I restrict going backwards
            self._restrict = np.array(
                [[[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                  [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                 [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0],
                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
                 [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0],
                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
                 [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
                 [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                  [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
                 [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0],
                  [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]])
        else:
            self._restrict = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))
        # Are the walls restricted
        if forbidden_walls:
            self.__restrict_walls__()
        return


class PlotterEnv(Env):
    """
    A child class of Env that can load in the event history of an agent and plot it
    """

    def __init__(self, file_name: str, **kwargs):
        """
        This class will only ever be used to plot previous results, thus we can only call it by loading a file
        :param file_name: the name of the environment file [csv]
        :param kwargs:
            use_epochs: do I plot in an epoch-based [True] or a step-based [False] manner. If not defined, cannot
                generate cumulative plots.
            with_replay: do I consider replay steps to be valid time steps (and thus appear on the time axis, True),
                or do I ignore replay time altogether [False, default]
            win_size: if use_epochs == False, we will need to perform a convolution on the reward rates, the size of the
                kernel of which can be defined here [int]
            path: path to the file. If nothing is specified we'll be looking in the working folder
            norm_rep: normalize the replay map if True (likelihood of replay instead of expected number of replay steps)
            params: what parameters were used to create the batches. If not None, then cumulative reward rates will be
                stored that can be plotted as a heatmap-like matrix
                win_begin: if we compute the cumulative reward rate (matrix-representation), then where does the
                    interval begin, over which we accumulate the obtained reward [optional int (step/epoch number),
                    default=0]
                win_end: if we compute the cumulative reward rate (matrix-representation), then where does the
                    interval end, over which we accumulate the obtained reward [optional int (step/epoch number),
                    default=None, can be negative integer]
        :return:
        """
        Env.__init__(self)

        # Parameters of plotting
        self._use_epochs = kwargs.get('use_epochs', None)
        if self._use_epochs is not None and not self._use_epochs:  # If we are step-based, we'll need the length of the kernel
            self._win_size = kwargs.get('win_size', None)
            if self._win_size is None:
                raise ValueError(
                    'If we plot in a step-based fashion, win_size needs to be defined for the reward rates.')
        self._with_replay = kwargs.get('with_replay', False)
        self._norm_rep = kwargs.get('norm_rep', False)  # Will we normalize the replay maps based on total replay count?

        # Placeholders for the agent data
        self._agent_events = None
        # The following variables are structured as {batch_name: np.ndarray(x, y, repetition), ...}
        self._stopped_at = {}  # A dict of arrays (of the maze) where each state counts stoppings by the agent
        self._replayed = {}  # A dict of arrays (of the maze) where each state counts replays by the agent
        # The following variables are structured as {batch_name: np.ndarray(t, repetition), ...}
        self._rew_rate = {}  # A dict of vectors containing the reward rate for each step/epoch
        self._replay_rate = {}  # A dict of vectors containing the replay rate for each step/epoch

        # In case of a need for matrix-like representation
        params = kwargs.get('params', None)
        if params is not None:
            params.append('cumul_rew')
            self._cumul_rew = pd.DataFrame(columns=params)
            self._win_begin = kwargs.get('win_begin', 0)
            self._win_end = kwargs.get('win_end', None)

        # The environment itself
        path = kwargs.get('path', None)
        self.load_env(file_name, path=path)
        return

    # Internal methods of the data aggreagtor
    def __count_replay__(self, batch: str) -> None:
        """
        Counts the replay events (stoppings) and the replayed states and stores them
        :param batch: the name of the batch of data we're working on
        :return:
        """
        # First extend the current memory
        if batch not in self._stopped_at:
            # If it's a new batch, then let's just add a corresponding array
            self._replayed[batch] = np.zeros((self._maze.shape[0], self._maze.shape[1], 1))
            self._stopped_at[batch] = np.zeros((self._maze.shape[0], self._maze.shape[1], 1))
        else:
            # If it is a pre-existing batch, let's simply extend it
            self._replayed[batch] = np.append(self._replayed[batch],
                                              np.zeros((self._maze.shape[0], self._maze.shape[1], 1)),
                                              axis=2)
            self._stopped_at[batch] = np.append(self._stopped_at[batch],
                                                np.zeros((self._maze.shape[0], self._maze.shape[1], 1)),
                                                axis=2)

        # Loop through the whole table
        for row_idx in range(len(self._agent_events)):
            if self._agent_events['step'].iloc[row_idx] > 0:  # If we have a replay event
                # We update the replayed states
                x, y = self.__find_state_coord__(row_idx, 's')
                self._replayed[batch][x, y, -1] += 1
                x, y = self.__find_state_coord__(row_idx, 's_prime')
                self._replayed[batch][x, y, -1] += 1

                if self._agent_events['step'].iloc[row_idx] == 1:  # And if this is really the first replay step
                    # We update stopping
                    x, y = self.__find_state_coord__(row_idx - 1, 's_prime')
                    self._stopped_at[batch][x, y, -1] += 1

        # Finally we normalize
        if self._norm_rep:
            sum_of_steps = (self._agent_events['step'] > 0).to_numpy().sum()
            sum_of_replay = (self._agent_events['step'] == 1).to_numpy().sum()
            if sum_of_steps > 0:
                self._replayed[batch][:, :, -1] /= sum_of_steps
            if sum_of_replay > 0:
                self._stopped_at[batch][:, :, -1] /= sum_of_replay
        return

    def __epoch_based_rate_computer__(self, data: dict, batch: str, agent_events: pd.core.frame.DataFrame,
                                      **kwargs) -> None:
        """
        Computes the reward or the replay rates in an epoch-based fashion. The core of the computation is
        (reward or replay number)/(number of steps in the epoch)
        :param data: either the self._rew_rate or the self._replay_rate
        :param batch: what batch we are working on
        :param agent_events: a dataframe with steps, iterations, rewards and replay
        :param kwargs:
            rate: if 'reward' (default) we compute reward rates. If 'replay' we compute replay rates
        :return:
        """
        rate = kwargs.get('rate', 'r')
        if rate == 'reward':
            rate = 'r'
        elif rate not in ['reward', 'r', 'replay']:
            raise ValueError('We can either compute reward or replay rates.')
        if rate == 'replay' and self._with_replay:
            raise ValueError('We cannot compute replay rates if replay is included in the time axis.')

        # If we consider epochs, then the reward rates is a fixed length vector
        # 1) Let's see if we have the requested batch
        if batch not in data:
            # If it's a new batch, then let's just add a corresponding array
            data[batch] = np.zeros((int((agent_events['r'] > 0).sum()), 1))
        else:
            # If it is a pre-existing batch, let's simply extend it
            data[batch] = np.append(data[batch], np.zeros((int((agent_events['r'] > 0).sum()), 1)),
                                    axis=1)

        # 2) Loop through the dataframe and count the steps between 2 rewards
        epoch_idx = 0
        last_rew_idx = 0
        for rew_idx in range(len(agent_events)):
            if agent_events['r'].iloc[rew_idx] > 0:
                # If we find the reward, we save r/(steps it took to get it) for the reward rate
                # And the (sum of all replays)/(steps we took) for the replay rate
                data[batch][epoch_idx, -1] = agent_events.loc[last_rew_idx:rew_idx, rate].sum() / \
                                             (rew_idx - last_rew_idx)
                last_rew_idx = rew_idx
                epoch_idx += 1

    def __step_based_rate_computer__(self, data: dict, batch: str, agent_events: pd.core.frame.DataFrame,
                                     **kwargs) -> None:
        """
        Computes the reward or the replay rates in a step-based fashion. The core of the computation is
        a convolution over the reward or replay rate with a predefined smoothing window
        :param data: either the self._rew_rate or the self._replay_rate
        :param batch: what batch we are working on
        :param agent_events: a dataframe with steps, iterations, rewards and replay
        :param kwargs:
            rate: if 'reward' (default) we compute reward rates. If 'replay' we compute replay rates
        :return:
        """
        rate = kwargs.get('rate', 'r')
        if rate == 'reward':
            rate = 'r'
        elif rate not in ['reward', 'r', 'replay']:
            raise ValueError('We can either compute reward or replay rates.')
        if rate == 'replay' and self._with_replay:
            raise ValueError('We cannot compute replay rates if replay is included in the time axis.')

        # If it is not measured in epochs, then we do a convolution
        conv_win = np.ones(self._win_size) * 1 / self._win_size

        # 1) We compute the convolved reward rate.
        # We will pad the rewards on the left, but not on the right!
        og_rate = np.append(np.zeros(math.floor(self._win_size / 2)), agent_events[rate].to_numpy())
        smooth_rate = np.convolve(og_rate, conv_win, mode='valid')

        # 2) Let's see if we have the requested batch
        if batch not in data:
            # If it's a new batch, then let's just add a corresponding array
            data[batch] = np.reshape(smooth_rate, (len(smooth_rate), 1))
        else:
            # If it is a pre-existing batch, we might need to pad it or the old ones on the right
            if data[batch].shape[0] > len(smooth_rate):
                smooth_rate = np.append(smooth_rate, smooth_rate[-1] * np.ones(data[batch].shape[0] - len(smooth_rate)))
            elif data[batch].shape[0] < len(smooth_rate):
                extension = np.repeat(data[batch][[-1], :], len(smooth_rate) - data[batch].shape[0], axis=0)
                data[batch] = np.append(data[batch], extension, axis=0)

            # And then we add it to the end
            data[batch] = np.append(data[batch], np.reshape(smooth_rate, (smooth_rate.shape[0], 1)), axis=1)

    def __compute_reward_rate__(self, batch: str) -> None:
        """
        Computes the reward rates and replay rates (if not with_replay) for the current agent data and stores it
        :param batch: The name of the batch of data we're working on
        :return:
        """
        # If replay is considered we work on the full dataframe, otherwise it's only the real steps
        # We only consider from step 1 as step 0 contains the initial state (and NaNs)
        agent_events = self._agent_events.loc[1:, ['iter', 'step', 'r']]
        if not self._with_replay:  # Let's count the replay steps
            agent_events['replay'] = -1 * agent_events['step'].diff()  # The number of replay steps *before* a given s
            agent_events = agent_events[agent_events['step'] == 0]
            agent_events['replay'] = agent_events['replay'].shift(-1)  # Now it's *after* a given s
            agent_events['replay'] = agent_events['replay'].fillna(0)  # Bc of the NaN after the last step
        # It's important to only count real rewards, thus we remove virtual ones
        agent_events.loc[agent_events['step'] > 0, 'r'] = 0

        if self._use_epochs:
            self.__epoch_based_rate_computer__(self._rew_rate, batch, agent_events)
            if not self._with_replay:
                self.__epoch_based_rate_computer__(self._replay_rate, batch, agent_events, rate='replay')

        else:
            # If it is not measured in epochs, then we do a convolution
            self.__step_based_rate_computer__(self._rew_rate, batch, agent_events)
            if not self._with_replay:
                self.__step_based_rate_computer__(self._replay_rate, batch, agent_events, rate='replay')
        return

    def __compute_cumul_rew__(self, batch: str) -> None:
        """
        Computes the cumulative rewards for the current agent data within the previously specified window and stores it
        :param batch: The name of the batch of data we're working on
        :return:
        """
        try:
            # 0) Extracting the batch information from the batch name
            curr_row = pd.DataFrame([[None] * len(self._cumul_rew.columns)], columns=self._cumul_rew.columns)
            for col_name in self._cumul_rew.columns:
                if col_name == 'cumul_rew':
                    continue
                # Now we want to find the col_name in the batch name. There are 2 possible formats: name_strvalue, or
                # nameNumvalue. Normally this is preceeded by an underscore (_name_val or _nameVal), except at the very
                # beginning of the batch name. val_idx will be the index where the value begins
                # 0.a) let's see if _name_val exists
                val_idx = batch.find(f'_{col_name}_') + len(col_name) + 2  # Idx where the variable value begins
                if val_idx - len(col_name) - 2 == -1:
                    # 0.b) If it doesn't, that means that the value might be numeric, so we should retry with _nameVal
                    res = re.search('_' + col_name + r'\d', batch)
                    if res is not None:
                        val_idx = res.start() + len(col_name) + 1
                    else:
                        # 0.c) If we still haven't found it that means the name_val/nameVal is at the beginning of the
                        # batch name. Let's see if we can find it as name_val
                        val_idx = batch.find(f'{col_name}_')
                        if val_idx == 0:
                            # If it is at the very beginning of the batch name, we're good
                            val_idx = len(col_name) + 1
                        else:
                            # 0.d) If it isn't, then nameVal has to be at the beginning of the file name
                            val_idx = len(col_name)

                # Now we can find the end index of the value using the start index
                val_end = batch[val_idx:].find('_')
                if val_end == -1:
                    curr_row[col_name].iloc[0] = batch[val_idx:]
                else:
                    curr_row[col_name].iloc[0] = batch[val_idx:val_idx + val_end]

            # 1) Preparing the tab;e to use, including the index by which we'll cut it up
            agent_events = self._agent_events.loc[1:, ['iter', 'step', 'r']]
            agent_events.loc[agent_events['step'] > 0, 'r'] = 0
            if not self._use_epochs:
                # If we're using steps we have to cut the table in a fashion that we take everything within the
                # specified window
                agent_events['idx'] = agent_events['iter']  # We will select based on the simple iter field of the table
            else:
                # If we're epoch based, we need to compute the epoch index, and cut the dataframe based on that
                agent_events['idx'] = (agent_events['r'] > 0).cumsum()  # We will select based on the epoch index

            # 2) Preparing the indices of the cuts whether or not they are epoch or step based
            win_begin = self._win_begin  # Win begin can be positive or negative integer
            if win_begin < 0:
                win_begin = agent_events['idx'].iloc[-1] + win_begin
            elif not self._use_epochs:
                # Since agent_events starts from index 1!
                win_begin += 1
            win_end = self._win_end  # Win end can be None (till the end) positive (iter num) or negative (end-iter num)
            if win_end is None:
                win_end = agent_events['idx'].iloc[-1]
            elif win_end < 0:
                win_end = agent_events['idx'].iloc[-1] + win_end
            elif not self._use_epochs:
                # Since agent_events starts from index 1!
                win_end += 1
            agent_events = agent_events.loc[(agent_events['idx'] >= win_begin) & (agent_events['iter'] < win_end), :]

            # 3) Computing and storing the sum reward
            curr_row['cumul_rew'] = agent_events['r'].sum()
            self._cumul_rew = pd.concat([self._cumul_rew, curr_row], axis=0)
            return

        except AttributeError:
            return

    # Internal methods of the plotter
    def __find_state_coord__(self, row_idx: int, col_name: str) -> Tuple[int, int]:
        """
        Finds the coordinates of a state at a given row in the event table
        :param row_idx: What row we are considering
        :param col_name: s or s_prime
        :return: the x and y coordinates of said state
        """
        s = self._agent_events[col_name].iloc[row_idx]
        [x, y] = np.argwhere(self._maze == s)[0]
        return x, y

    def __event_to_img__(self, values: pd.core.frame.DataFrame) -> np.ndarray:
        """
        Takes a row from a pandas dataframe, each column of ot containing a value corresponding a state. This row is
        then converted into a numpy array where these values are projected onto the actual maze.
        :param values:
        :return:
        """
        values = values.to_numpy()
        image = np.zeros(self._maze.shape)
        image[self._maze >= 0] = values
        return image

    def __status_to_image__(self, it: int) -> np.ndarray:
        """
        It will produce an array reflecting the status of the maze in iteration it. The array will follow the following
        conventions: wall = 0, path = 1, reward = 2 (irrelevant of value), agent = 3. If the agent is in a rewarded
        state, the state will have a value of 3 (agent)
        :param it: the iteration we are in
        :return:
        """
        # wall = 0, path = 1, reward = 2, agent = 3
        image = np.zeros(self._maze.shape)
        image[self._maze >= 0] = 1
        reward_num = int((self._events.shape[1] - 3) / 4)
        for rew_idx in range(reward_num):
            if self._events[f'rew{rew_idx}_pos_x'].iloc[it] >= 0:
                image[int(self._events[f'rew{rew_idx}_pos_x'].iloc[it]),
                int(self._events[f'rew{rew_idx}_pos_y'].iloc[it])] = 2
        image[int(self._events['agent_pos_x'].iloc[it]), int(self._events['agent_pos_y'].iloc[it])] = 3
        return image

    def __replay_to_image__(self, curr_image: np.ndarray, row_idx: int) -> np.ndarray:
        """
        Takes the last array representing the replayed states (if no replay had taken lace earlier, we simply use an
        array of zeros) and based on the current row_idx (not iter, not step), we add the last replay to this maze
        :param curr_image: The array depicting the last replay step
        :param row_idx: the row idx in the agent event memory table the replay of which we want to depict
        :return:
        """
        max_val = curr_image.max().max()
        x, y = self.__find_state_coord__(row_idx, 's')
        curr_image[x, y] = max_val + 1
        [x, y] = self.__find_state_coord__(row_idx, 's_prime')
        curr_image[x, y] = max_val + 2
        return curr_image

    def __rate_plotter__(self, data: dict, batches: list, ax) -> None:
        """
        Plots the reward or replay rates on a given axis.
        :param data: self._rew_rate or self._replay_rate
        :param batches: the list of batches to plot
        :param ax: the axis on which we are working
        :return:
        """
        rr = data[batches[0]]
        for batch_idx in range(1, len(batches)):
            # Between the different conditions the length of the time axis might differ given that we might replay
            # differently. Thus, the longest time axis needs to be found and the rest needs to be padded
            rr_curr = data[batches[batch_idx]]
            if rr.shape[0] > rr_curr.shape[0]:
                extension = np.repeat(rr_curr[[-1], :], rr.shape[0] - rr_curr.shape[0], axis=0)
                rr_curr = np.append(rr_curr, extension, axis=0)
            elif rr.shape[0] < rr_curr.shape[0]:
                extension = np.repeat(rr[[-1], :], rr_curr.shape[0] - rr.shape[0], axis=0)
                rr = np.append(rr, extension, axis=0)
            rr = np.append(rr, rr_curr, axis=1)
        t = range(rr.shape[0])
        header = pd.MultiIndex.from_product([batches, [f'{idx}' for idx in range(data[batches[0]].shape[1])]],
                                            names=['batch', 'run'])
        df = pd.DataFrame(rr, index=t, columns=header)
        x_name = 'steps'
        if self._use_epochs:
            x_name = 'epochs'
        df[x_name] = t

        # Plotting
        df = pd.melt(df, id_vars=x_name)
        sns.lineplot(df, x=x_name, y='value', hue='batch', ax=ax)

    # Function to handle the data
    def load_events(self, file_name: str, batch: str, **kwargs):
        """
        Loads the steps of an agent
        :param file_name: the name of the agent file [csv]
        :param batch: what batch the current file belongs to [str]
        :param kwargs:
            path: path to the file. If nothing is specified we'll be looking in the working folder
        :return:
        """
        path = kwargs.get('path', None)
        if path is not None:
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                raise FileNotFoundError(f'No directory named {path}')
        else:
            path = './'

        if os.path.isfile(f'{path}{file_name}'):
            self._agent_events = pd.read_csv(f'{path}{file_name}')
        else:
            raise FileNotFoundError(f'No file named {file_name}')

        # And then we immediately perform some basic data aggregation
        self.__count_replay__(batch)
        if self._use_epochs is not None:
            self.__compute_reward_rate__(batch)
            self.__compute_cumul_rew__(batch)
        return

    # Plotters
    def plot_events(self, **kwargs):
        """
        Plots the events of the experiment in an animated fashion. It uses 2 distinct plots: one for the maze, the
        replay and the Q values, the other one for the Ur and the Ut values.
        :param kwargs:
            weights: [wQ, wUr, wUt], the weights of the different components of the C-value
            gamma: the discounting factor (needed to compute Qmax for the normalization)
        :return:
        """
        # 0) Preparing the dataframes -- we need the max Q value for each state, and (as of now) the mean H value
        Q_vals = pd.DataFrame()
        Ur_vals, Ut_vals = pd.DataFrame(), pd.DataFrame()
        empty_arr = np.empty(self._agent_events.shape[0])
        empty_arr[:] = np.nan
        for s_idx in range(self.state_num()):
            if f'Q_{s_idx}_0' in self._agent_events.columns:
                cols = [f'Q_{s_idx}_{a_idx}' for a_idx in range(self.act_num())]
                Q_vals[f'Q_{s_idx}'] = self._agent_events[cols].max(axis=1)
            else:
                Q_vals[f'Q_{s_idx}'] = pd.DataFrame(empty_arr,
                                                    columns=[f'Q_{s_idx}'])
            # As for the H values, some actions (namely the forbidden ones) will never be explored by the agent. Thus
            # instead of storing all H values from the table, we only store those that the agent had a chance to learn
            if f'Ur_{s_idx}_0' in self._agent_events.columns:
                cols = [f'Ur_{s_idx}_{a_idx}' for a_idx in self.possible_moves(s_idx)]
                Ur_vals[f'Ur_{s_idx}'] = self._agent_events[cols].max(axis=1)
            else:
                Ur_vals[f'Ur_{s_idx}'] = pd.DataFrame(empty_arr,
                                                      columns=[f'Ur_{s_idx}'])

            if f'Ut_{s_idx}_0' in self._agent_events.columns:
                cols = [f'Ut_{s_idx}_{a_idx}' for a_idx in self.possible_moves(s_idx)]
                Ut_vals[f'Ut_{s_idx}'] = self._agent_events[cols].max(axis=1)
            else:
                Ut_vals[f'Ut_{s_idx}'] = pd.DataFrame(empty_arr,
                                                      columns=[f'Ut_{s_idx}'])

        weights = kwargs.get('weights', np.array([1, 0, 0]))
        if np.sum(weights) < 1 - np.finfo(np.float32).eps or 1 + np.finfo(np.float32).eps < np.sum(weights):
            raise ValueError('The weights of the different quality values must sum up to 1.')
        max_vals = np.array([np.nanmax(Q_vals.to_numpy()),
                             np.nanmax(Ur_vals.to_numpy()),
                             np.nanmax(Ut_vals.to_numpy())])
        gamma = kwargs.get('gamma', None)
        if gamma is not None:
            # max_vals = np.array([max(self.max_rew(iter=0) / (1 - gamma),
            #                          Q_vals.iloc[0].to_numpy().max()),  # Qmax
            #                      max(RLA.entropy(np.ones(2) / 2) / (1 - gamma),
            #                          Ur_vals.iloc[0].to_numpy().max()),  # Ur max
            #                      max(RLA.entropy(np.ones(self.state_num()) / self.state_num()) / (1 - gamma),
            #                          Ut_vals.iloc[0].to_numpy().max())])  # Ut max
            max_vals = np.array([np.nanmax(Q_vals.iloc[0].to_numpy()),  # Qmax
                                 np.nanmax(Ur_vals.iloc[0].to_numpy()),  # Ur max
                                 np.nanmax(Ut_vals.iloc[0].to_numpy())])  # Ut max
            max_vals[max_vals == 0] = 1

        # 1) Preparing the Q plots and the H plots
        plt.ion()
        fig_env, ax_env = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        ax_env[0].set_title("Map")
        ax_env[1].set_title("Replay")
        ax_env[2].set_title("C values")
        curr_maze = self.__status_to_image__(0)
        curr_replay = np.zeros(self._maze.shape)
        curr_C_df = pd.DataFrame(np.reshape(Q_vals.iloc[0].to_numpy() / max_vals[0] * weights[0] +
                                            Ur_vals.iloc[0].to_numpy() / max_vals[1] * weights[1] +
                                            Ut_vals.iloc[0].to_numpy() / max_vals[2] * weights[2],
                                            (1, self.state_num())))
        curr_C = self.__event_to_img__(curr_C_df.iloc[0])
        axim_env = np.array([ax_env[0].imshow(curr_maze),
                             ax_env[1].imshow(curr_replay),
                             ax_env[2].imshow(curr_C, vmin=0, vmax=1)])
        axim_env[0].autoscale()  # Since here the extremes already appear
        axim_env[1].autoscale()  # This will have to be done in every step if we want the old replay steps to fade away

        fig_rla, ax_rla = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        ax_rla[0].set_title("Q values")
        ax_rla[1].set_title("Ur values")
        ax_rla[2].set_title("Ut values")
        curr_Ur = self.__event_to_img__(Ur_vals.iloc[0])
        curr_Ut = self.__event_to_img__(Ut_vals.iloc[0])
        curr_Q = self.__event_to_img__(Q_vals.iloc[0])
        axim_rla = np.array([ax_rla[0].imshow(curr_Q, vmin=0, vmax=max_vals[0]),
                             ax_rla[1].imshow(curr_Ur, vmin=0, vmax=max_vals[1]),
                             ax_rla[2].imshow(curr_Ut, vmin=0, vmax=max_vals[2])])

        txt = np.empty((*self._maze.shape, 4), dtype=matplotlib.text.Text)  # txt will appear for the Q and the H values
        for idx_x in range(txt.shape[0]):
            for idx_y in range(txt.shape[1]):
                txt[idx_x, idx_y, 0] = ax_env[2].text(idx_y, idx_x, f"{curr_C[idx_x, idx_y]: .2f}",
                                                      ha="center", va="center", color="w")
                txt[idx_x, idx_y, 1] = ax_rla[0].text(idx_y, idx_x, f"{curr_Q[idx_x, idx_y]: .2f}",
                                                      ha="center", va="center", color="w")
                txt[idx_x, idx_y, 2] = ax_rla[1].text(idx_y, idx_x, f"{curr_Ur[idx_x, idx_y]: .1f}",
                                                      ha="center", va="center", color="w")
                txt[idx_x, idx_y, 3] = ax_rla[2].text(idx_y, idx_x, f"{curr_Ut[idx_x, idx_y]: .1f}",
                                                      ha="center", va="center", color="w")

        # plt.pause(.001)

        # 2) Looping through the memories
        for row_idx in range(1, self._agent_events.shape[0]):
            it = int(self._agent_events['iter'].iloc[row_idx])
            step = int(self._agent_events['step'].iloc[row_idx])

            # 2.a) If the agent's memory does not correspond to that of the environment, we quit
            # It is important to note here that during replay there's always a mismatch (hence if self > 0 we ignore)
            # and that if a reward is given, the agent is moved, so there's also a mismatch
            if step == 0 and self._agent_events['r'].iloc[row_idx] == 0 \
                    and self._agent_events['s_prime'].iloc[row_idx] != \
                    self._maze[int(self._events['agent_pos_x'].iloc[it]), int(self._events['agent_pos_y'].iloc[it])]:
                raise ValueError("mismatch between agent and environment memory")

            # 2.b) Else we have to see if we perform replay or not
            if step > 0:
                curr_replay = self.__replay_to_image__(curr_replay, row_idx)
            else:
                curr_replay = np.zeros(self._maze.shape)
            curr_maze = self.__status_to_image__(it)
            curr_Q = self.__event_to_img__(Q_vals.iloc[row_idx])
            curr_Ur = self.__event_to_img__(Ur_vals.iloc[row_idx])
            curr_Ut = self.__event_to_img__(Ut_vals.iloc[row_idx])
            if gamma is not None:
                # max_vals = np.array([max(self.max_rew(iter=it) / (1 - gamma),
                #                          Q_vals.iloc[row_idx].to_numpy().max()),  # Qmax
                #                      max(RLA.entropy(np.ones(2) / 2) / (1 - gamma),
                #                          Ur_vals.iloc[row_idx].to_numpy().max()),  # Ur max
                #                      max(RLA.entropy(np.ones(self.state_num()) / self.state_num()) / (1 - gamma),
                #                          Ut_vals.iloc[row_idx].to_numpy().max())])  # Ut max
                max_vals = np.array([np.nanmax(Q_vals.iloc[row_idx].to_numpy()),  # Qmax
                                     np.nanmax(Ur_vals.iloc[row_idx].to_numpy()),  # Ur max
                                     np.nanmax(Ut_vals.iloc[row_idx].to_numpy())])  # Ut max
                max_vals[max_vals == 0] = 1
            curr_C_df = pd.DataFrame(
                np.reshape(Q_vals.iloc[row_idx].to_numpy() / max_vals[0] * weights[0] +
                           Ur_vals.iloc[row_idx].to_numpy() / max_vals[1] * weights[1] +
                           Ut_vals.iloc[row_idx].to_numpy() / max_vals[2] * weights[2],
                           (1, self.state_num())))
            curr_C = self.__event_to_img__(curr_C_df.iloc[0])

            # 2.c) Refresh txt
            for idx_x in range(txt.shape[0]):
                for idx_y in range(txt.shape[1]):
                    txt[idx_x, idx_y, 0].set_text(f"{curr_C[idx_x, idx_y]: .2f}")
                    txt[idx_x, idx_y, 1].set_text(f"{curr_Q[idx_x, idx_y]: .2f}")
                    txt[idx_x, idx_y, 2].set_text(f"{curr_Ur[idx_x, idx_y]: .1f}")
                    txt[idx_x, idx_y, 3].set_text(f"{curr_Ut[idx_x, idx_y]: .1f}")

            # 2.d) Refresh plots
            axim_env[0].set_data(curr_maze)
            axim_env[1].set_data(curr_replay)
            axim_env[2].set_data(curr_C)
            axim_env[1].autoscale()

            axim_rla[0].set_data(curr_Q)
            axim_rla[1].set_data(curr_Ur)
            axim_rla[2].set_data(curr_Ut)
            axim_rla[0].set_clim(vmax=max_vals[0])
            axim_rla[1].set_clim(vmax=max_vals[1])
            axim_rla[2].set_clim(vmax=max_vals[2])

            # 2.e) Stop
            fig_env.canvas.flush_events()
            fig_rla.canvas.flush_events()
            # plt.pause(.001)

    def plot_reward_rates(self, batches: list, **kwargs) -> None:
        """
        Plots the reward rates over the steps or over the epochs. If replay steps are not considered in the x axis, a
        replay-rate subplot is also produced.
        :param batches: a list of the names of the batches we want to compare
        :param kwargs:
            save_img: do I want to save the output fig
            path: if save_img, where do I want to save (default: ./)
            label: if save_img, what tag should I attach to the figure name
        :return:
        """
        if self._use_epochs is None:
            raise ValueError(
                'Cannot compute reward rates without knowing if the experiment is epoch or step based.')
        if self._with_replay:
            fig, axes = plt.subplots(1, 1, figsize=(15, 4))
            axes = np.array([axes])
        else:
            fig, axes = plt.subplots(2, 1, figsize=(15, 9))
            axes[1].set_title('Replay rates')
        axes[0].set_title('Reward rates')
        if self._use_epochs:
            axes[0].set_xlabel('Epochs')
            axes[0].set_ylabel(f'Reward / steps in epoch')
            if not self._with_replay:
                axes[1].set_xlabel('Epochs')
                axes[1].set_ylabel(f'Replay steps / real steps in epoch')
        else:
            axes[0].set_xlabel('Steps')
            axes[0].set_ylabel(f'Reward / {self._win_size} steps')
            if not self._with_replay:
                axes[1].set_xlabel('Steps')
                axes[1].set_ylabel(f'Replay steps / {self._win_size} real steps')

        # Creating a dataframe for seaborn to work with
        self.__rate_plotter__(self._rew_rate, batches, axes[0])
        if not self._with_replay:
            self.__rate_plotter__(self._replay_rate, batches, axes[1])

        plt.show(block=False)

        # Saving the fig if we need to
        if kwargs.get('save_img', False):
            path = kwargs.get('path', './')
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                os.mkdir(path)
            label = kwargs.get('label', '')
            plt.savefig(f'{path}rew_rate_{label}.pdf', format="pdf", bbox_inches="tight")

    def plot_replay(self, to_plot: str, batches: list, shape: list, **kwargs) -> None:
        """
        Plots the maze with the average replay or stopping frequencies in each state.
        :param to_plot: 'loc' for where the agent stopped to replay, 'content' for what the agent reoplayed
        :param batches: a list of the names of the batches we consider (each in a separate img, shared color scale)
        :param shape: the shape of the subplots
        :param kwargs:
            save_img: do I want to save the output fig
            path: if save_img, where do I want to save (default: ./)
            label: if save_img, what tag should I attach to the figure name
        :return:
        """
        if to_plot not in ['loc', 'content']:
            raise ValueError('to_plot has to be either "loc" or "content"')
        if to_plot == 'loc':
            title = 'Stopped to replay at'
        else:
            title = 'Replayed'
        fig, axes = plt.subplots(shape[0], shape[1], figsize=(8 * shape[1], 4 * shape[0]))
        if shape[0] * shape[1] != len(batches):
            raise ValueError('The number of batches has to be the same as the number of subplots')
        elif shape[0] == 1 or shape[1] == 1:
            axes = np.array([axes])  # So that later on I can index it

        # Collecting the final dataframes
        df = [None] * len(batches)  # All te dataframes
        vmin, vmax = None, None
        for batch_idx in range(len(batches)):
            if to_plot == 'loc':
                df[batch_idx] = pd.DataFrame(np.mean(self._stopped_at[batches[batch_idx]], axis=2))
                lab = 'expected stops'
                if self._norm_rep:
                    lab = 'likelihood of stopping'
            elif to_plot == 'content':
                df[batch_idx] = pd.DataFrame(np.mean(self._replayed[batches[batch_idx]], axis=2))
                lab = 'expected replay'
                if self._norm_rep:
                    lab = 'likelihood of replay'

            # For normalization purposes
            if batch_idx == 0:
                vmin = df[batch_idx].to_numpy().min()
                vmax = df[batch_idx].to_numpy().max()
            else:
                if vmin > df[batch_idx].to_numpy().min():
                    vmin = df[batch_idx].to_numpy().min()
                if vmax < df[batch_idx].to_numpy().max():
                    vmax = df[batch_idx].to_numpy().max()

        # The actual plotting
        for batch_idx in range(len(batches)):
            idx_x = math.floor(batch_idx / axes.shape[1])
            idx_y = batch_idx % axes.shape[1]
            axes[idx_x, idx_y].set_title(f'{title} ({batches[batch_idx]})')
            sns.heatmap(df[batch_idx], ax=axes[idx_x, idx_y], vmin=vmin, vmax=vmax, cbar_kws={'label': lab})

        plt.show(block=False)

        if kwargs.get('save_img', False):
            path = kwargs.get('path', './')
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                os.mkdir(path)
            label = kwargs.get('label', '')
            plt.savefig(f'{path}rep_{to_plot}_{label}.pdf', format="pdf", bbox_inches="tight")

    def plot_cumul_rew_matrix(self, params: list, **kwargs):
        """
        Plots (and potentially stores) a matrix representation of the max (or mean) reward rates over a series of runs
        :param params: The parameters constituting the x and y axes of the produced matrix [list of 2 strings]
        :param kwargs:
            save_img: do I want to save the output fig
            path: if save_img, where do I want to save (default: ./)
            label: if save_img, what tag should I attach to the figure name
            method: if "mean" then mean cumulative reward rates will be computed instead of max (default). In this case
                the matrix annotation will change too: instead of the parameters of the best model, we'll use the avg
                reward rate as label
        :return:
        """
        if self._use_epochs is None:
            raise ValueError(
                'Cannot compute cumulative rewards without knowing if the experiment is epoch or step based.')
        # 1) Preparing the figure
        fig, ax = plt.subplots(1, 1, figsize=(1.8 * len(self._cumul_rew[params[0]].unique()),
                                              1.5 * len(self._cumul_rew[params[1]].unique())))
        titl = 'step'
        if self._use_epochs:
            titl = 'epoch'
        beg = self._win_begin
        if self._win_begin < 0:
            beg = f'(end - {-self._win_begin})'
        end = self._win_end
        if self._win_end is None:
            end = 'the end'
        elif self._win_end < 0:
            end = f'(end - {-self._win_end})'
            titl = titl + 's'

        # 2) Preparing the data -- we take the maximum for each group
        method = kwargs.get('method', 'max')
        if method == 'max':
            ax.set_title(f'Max cumulative rewards between {titl} {beg} and {end}')
            # 2.a) We can find the max of the cum reward rates
            cumul_mat = self._cumul_rew.loc[
                self._cumul_rew.groupby(params)['cumul_rew'].transform(max) == self._cumul_rew['cumul_rew']]
            # Since this returns *all* maxima, we might want to drop the duplicates, otherwise it'll be impossible to plot
            cumul_mat = cumul_mat.drop_duplicates(params, keep='first')
            # Now we need to do pandas magic to make it understand non-numeric axis values
            cumul_mat_heatmap = cumul_mat.pivot(columns=params[0], index=params[1], values='cumul_rew')
            # And we need to make an identically shaped annotation matrix
            cumul_mat_annot = None
            for col_name in cumul_mat.columns:
                if col_name in params or col_name == 'cumul_rew':
                    continue
                if cumul_mat_annot is None:
                    cumul_mat_annot = f'{col_name}=' + cumul_mat.pivot(columns=params[0], index=params[1],
                                                                       values=col_name)
                else:
                    cumul_mat_annot += f'\n {col_name}=' + cumul_mat.pivot(columns=params[0], index=params[1],
                                                                           values=col_name)
            lab = 'max cumulative reward'
            annot_format = ''
        elif method == 'mean':
            ax.set_title(f'Mean cumulative rewards between {titl} {beg} and {end}')
            # 2.b) Or we can just plot the means
            cumul_mat = self._cumul_rew.groupby(params)['cumul_rew'].mean()
            # Then we basically need to "un-groupby" it
            cumul_mat_heatmap = cumul_mat.reset_index(params)
            # And then make sure that pandas understands non-numeric axis values
            cumul_mat_heatmap = cumul_mat_heatmap.pivot(columns=params[0], index=params[1], values='cumul_rew')
            # Finally the annotation matrix, here it will only contain the reward values
            cumul_mat_annot = cumul_mat_heatmap
            lab = 'mean cumulative reward'
            annot_format = '.2f'
        else:
            raise ValueError('Method has to be "max" or "mean".')
        sns.heatmap(cumul_mat_heatmap, annot=cumul_mat_annot, cbar_kws={'label': lab}, fmt=annot_format)
        ax.invert_yaxis()
        plt.show(block=False)

        # 3) Saving if necessary
        if kwargs.get('save_img', False):
            path = kwargs.get('path', './')
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                os.mkdir(path)
            label = kwargs.get('label', '')
            plt.savefig(f'{path}{method}_cumul_rew_{params[0]}_{params[1]}_{label}.pdf', format="pdf",
                        bbox_inches="tight")
