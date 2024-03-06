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

        return

    # Plotters
    def plot_events(self):
        """
        Plots the events of the experiment in an animated fashion. It uses 2 distinct plots: one for the maze, the
        replay and the Q values, the other one for the Ur and the Ut values.
        :return:
        """
        # 0) Preparing the dataframes -- we need the max Q value for each state, and (as of now) the mean H value
        Q_vals = pd.DataFrame()
        Ur_vals, Ut_vals, C_vals = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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

            if f'C_{s_idx}_0' in self._agent_events.columns:
                cols = [f'C_{s_idx}_{a_idx}' for a_idx in self.possible_moves(s_idx)]
                C_vals[f'C_{s_idx}'] = self._agent_events[cols].max(axis=1)
            else:
                C_vals[f'C_{s_idx}'] = pd.DataFrame(empty_arr,
                                                    columns=[f'C_{s_idx}'])

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
        curr_C = self.__event_to_img__(C_vals.iloc[0])
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
            max_vals = np.array([np.nanmax(Q_vals.iloc[row_idx].to_numpy()),  # Qmax
                                 np.nanmax(Ur_vals.iloc[row_idx].to_numpy()),  # Ur max
                                 np.nanmax(Ut_vals.iloc[row_idx].to_numpy())])  # Ut max
            max_vals[max_vals == 0] = 1
            curr_C = self.__event_to_img__(C_vals.iloc[row_idx])

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
