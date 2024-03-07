import csv
import random
import numpy as np
import pickle as pkl 

'''
    Parent class implementing the Sequential Episodic Control (SEC) algorithm
'''

class SECagent(object):

    def __init__(self, action_space=4, emb=20, stm=50, ltm=500, 
                sequential_bias=True, value_function="default", forget="NONE", 
                coll_threshold_act=0.98, coll_threshold_proportion=0.995,
                alpha_trigger=0.05, tau_decay=0.9, load_ltm=False,
                exploration_mode='default', exploration_steps=2500, epsilon=0.05,
                selection_mode='default', memory_threshold=1):

        self.stm_length = stm       # STM sequence length
        self.ltm_length = ltm       # LTM buffer capacity: Total n of sequences stored in LTM
        self.emb_length = emb        # EMB = embedding length: Size of the state vector
        
        self.sequential_bias = sequential_bias   # sequential bias
        self.value_function = value_function  # TODO check it, it used to be ==
        self.forget = forget    # can be "FIFO", "SING" or "PROP"

        #print("STM length: ", self.stm_length)
        #print("LTM length: ", self.ltm_length)
        #print("Forgetting: ", self.forget)
        #print("Sequential Bias: ", self.sequential_bias)

        self.coll_thres_act = coll_threshold_act            # default 0.9
        self.coll_thres_prop = coll_threshold_proportion    # default 0.95
        self.alpha_tr = alpha_trigger                       # default 0.005
        self.tau_decay = tau_decay                          # default 0.9

        self.action_space = action_space                                     # can be a list "[3, 3]" or a integer "6"
        #print("CL action_space: ", self.action_space)
        #print("action_space type: ", type(self.action_space))

        self.action = 0
        #print("action: ", self.action)

        #self.STM = [[np.zeros(self.emb_length), np.zeros(1)] for _ in range(self.stm_length)] # pl = prototype length (i.e. dimension of the state vector)
        self.STM = [[[0] * self.emb_length , 0] for i in range(self.stm_length)]
        self.LTM = [[],[],[]]
        self.memory_full = False 
        self.memory_threshold =  memory_threshold 
        self.forget_ratio = 0.01 # 1% of memories will be erased when using Forgetting PROP

        self.tr = []
        self.last_actions_indx = []
        self.selected_actions_indx = []

        self.entropy = 0.
        self.selection_mode = selection_mode

        self.steps = 0
        self.exploration_mode = exploration_mode
        self.exploration_steps = exploration_steps # full random exploration time
        self.epsilon = epsilon

        if load_ltm: self.load_LTM()

    def choose_action(self, state):

        if self.exploration_mode == 'default':
            action, q = self.default_step(state)
        if self.exploration_mode == 'fixed':
            action, q = self.fixed_step(state)
        if self.exploration_mode == 'greedy':
            action, q = self.action_selection(state)
        if self.exploration_mode == 'epsilon':
            action, q = self.epsilon_step(state)
            #print('state is: ', state)
        if self.exploration_mode == 'epsilon_decay':
            action, q = self.epsilon_step(state)
            self.update_epsilon()

        # MEMORY UPDATE PHASE 1
        #self.update_STM(sa_couplet = [state, action])
        #self.update_sequential_bias()

        return action, q

    def update_epsilon(self):
        if self.epsilon > 0.05: #R
            self.epsilon -= (0.9/self.exploration_steps)

    def default_step(self, state):
        # Standard: Explore until achieve X number of memories
        if len(self.LTM[2]) > self.memory_threshold:
            action, q = self.action_selection(state)
        else:
            action = np.random.choice(a=self.action_space)
            q = 0
        return action, q

    def fixed_step(self, state):
        self.steps += 1
        # For Atari games: Chose CL action after a minimum number of exploration steps have been taken
        action, q = self.action_selection(state)
        if self.steps < self.exploration_steps:
            action = np.random.choice(a=self.action_space)
        return action, q

    def epsilon_step(self, state):
        # For Atari games: Follow an epsilon-greedy policy
        action, q = self.action_selection(state)
        if (np.random.random() < self.epsilon):
            action = np.random.choice(a=self.action_space)
        return action, q

    def action_selection(self, state):
        # get updated policy for a given state
        q = self.estimate_return(state)
        #print('Q: ', q)

        if self.selection_mode == 'default':
            # SEC DEFAULT: SAMPLE FROM WEIGHTED PROBABILITY
            self.action = np.random.choice(np.arange(q.shape[0]), p=q)
            #ac_indx = np.random.choice(np.arange(int(self.action_space[0]*self.action_space[1])), p=q)
            #self.action = [int(ac_indx/self.action_space[0]), int(ac_indx%self.action_space[1])]

        if self.selection_mode == 'argmax':
            # RL STANDARD: ARGMAX
            self.action = np.argmax(q)

        q_action = q[self.action]

        if isinstance(self.action_space, list):
            print("NOTE: action space is a list")
            self.action = [int(self.action/self.action_space[1]), self.action % self.action_space[1]]

        return self.action, q_action

    def estimate_return(self, state):
        # get the state-action value based on the memories stored in the LTM
        q = np.ones(self.action_space) / self.action_space

        if len(self.LTM[0]) > 0:

            bias = 1
            if self.sequential_bias:
                bias = np.array(self.tr)
                #print("bias length: ", len(bias[0])) # proportional to sequence's length, n = LTM sequences

            collectors = (1 - (np.sum(np.abs(state - self.LTM[0]), axis=2)) / len([state])) * bias
            #print ("collectors ", collectors) # proportional to sequence's length, n = LTM sequences

            # Collector values must be above both thresholds (absolute and relative) to contribute to action.
            self.selected_actions_indx = (collectors > self.coll_thres_act) & ((collectors/collectors.max()) > self.coll_thres_prop) # proportional to sequence's length, n = LTM sequences
            #print ("selected_actions_indx ", self.selected_actions_indx)

            if np.any(self.selected_actions_indx):

                actions = np.array(self.LTM[1])[self.selected_actions_indx]
                # choose (normalized, or relative) rewards of sequences with actions selected
                rewards = np.array(self.LTM[2])[(np.nonzero(self.selected_actions_indx)[0])]
                rewards = rewards/rewards.max()
                # choose (normalized) distances of each action selected within its sequence
                distances = (self.stm_length - np.nonzero(self.selected_actions_indx)[1])/self.stm_length
                # choose collector info about the actions selected (that take euclidean distance of current state and collector's selected states)
                collectors = collectors[self.selected_actions_indx]

                q = self.get_policy(actions, collectors, rewards, distances)

                # compute entropy over the policy
                self.compute_entropy(q)

            self.selected_actions_indx = self.selected_actions_indx.tolist()
            #print ("selected_actions_indx ", self.selected_actions_indx)

        return q

    def get_policy(self, actions, collectors, rewards, distances):
        # map each selected action-vector into a matrix of N dimensions where N are the dimensions of the action space
        m = np.zeros((len(actions), self.action_space))
        #m[np.arange(len(actions)), actions[:].astype(int)] = collectors*(rewards*np.exp(-distances/self.tau_decay))

        if self.value_function == 'default':
            #print('COMPUTING ACTIONS CLASSIC SEC...')
            m[np.arange(len(actions)), actions[:].astype(int)] = collectors*(rewards*np.exp(-distances/self.tau_decay))
        if self.value_function == 'noGi':
            #print('COMPUTING ACTIONS WITHOUT SIMILARITY...')
            m[np.arange(len(actions)), actions[:].astype(int)] = rewards*np.exp(-distances/self.tau_decay)
        if self.value_function == 'noDist':
            #print('COMPUTING ACTIONS WITHOUT DISTANCE...')
            m[np.arange(len(actions)), actions[:].astype(int)] = collectors*rewards
        if self.value_function == 'noRR':
            #print('COMPUTING ACTIONS WITHOUT REWARD...')
            m[np.arange(len(actions)), actions[:].astype(int)] = collectors*np.exp(-distances/self.tau_decay)
        if self.value_function == 'soloGi':
            #print('COMPUTING ACTIONS WITH ONLY SIMILARTY...')
            m[np.arange(len(actions)), actions[:].astype(int)] = collectors
        if self.value_function == 'soloDist':
            #print('COMPUTING ACTIONS WITH ONLY DISTANCE...')
            m[np.arange(len(actions)), actions[:].astype(int)] = np.exp(-distances/self.tau_decay)
        if self.value_function == 'soloRR':
            #print('COMPUTING ACTIONS WITH ONLY RELATIVE REWARD...')
            m[np.arange(len(actions)), actions[:].astype(int)] = rewards

        q = np.sum(m, axis=0)
        #q = q + np.abs(q.min())+1 # NEW
        q = q/q.sum()  #proportion of being selected based on the action's relative reward based on the stored experiences
        
        ### TO TEST CHANGE RELATIVE REWARD FOR SOFTMAX FOR ENVS WITH NEGATIVE REWARDS
        # q = np.softmax(q)
        # q = np.exp(q - np.max(q)) / np.exp(q - np.max(m)).sum()  -- sofmax function corrected for large numbers
        # q = np.exp(q) / np.exp(q).sum()  -- sofmax function unstable for large numbers

        q = q.flatten()

        return q

    def get_policy_from_list(self, actions, collectors, rewards, distances):
        # map each selected action-vector into a matrix of N dimensions where N are the dimensions of the action space
        m = np.zeros((len(actions), self.action_space[0], self.action_space[1]))
        #m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = ((collectors*rewards)/distances)
        #m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = collectors*(rewards*np.exp(-distances/self.tau_decay))
        
        if self.value_function == 'default':
            #print('COMPUTING ACTIONS CLASSIC SEC...')
            m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = collectors*(rewards*np.exp(-distances/self.tau_decay))
        if self.value_function == 'noGi':
            #print('COMPUTING ACTIONS WITHOUT SIMILARITY...')
            m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = rewards*np.exp(-distances/self.tau_decay)
        if self.value_function == 'noDist':
            #print('COMPUTING ACTIONS WITHOUT DISTANCE...')
            m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = collectors*rewards
        if self.value_function == 'noRR':
            #print('COMPUTING ACTIONS WITHOUT REWARD...')
            m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = collectors*np.exp(-distances/self.tau_decay)
        if self.value_function == 'soloGi':
            #print('COMPUTING ACTIONS WITH ONLY SIMILARTY...')
            m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = collectors
        if self.value_function == 'soloDist':
            #print('COMPUTING ACTIONS WITH ONLY DISTANCE...')
            m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = np.exp(-distances/self.tau_decay)
        if self.value_function == 'soloRR':
            #print('COMPUTING ACTIONS WITH ONLY RELATIVE REWARD...')
            m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = rewards

        m = np.sum(m, axis=0)
        m = m/m.sum()  #proportion of being selected based on the action's relative reward based on the stored experiences

        q = m.flatten()

        return q

    def compute_q_table(self):
        # Step 1: Retrieve all unique states experienced from LTM
        unique_states = np.unique(self.LTM[0])
        print("unique_states:", unique_states)

        # Step 2: Compute Q values for each state using the estimate_return() function
        q_values = {}
        for state in unique_states:
            q_values[state] = self.estimate_return(state)

        # Step 3: Save the maximum value of the resulting q values for each state in a new table
        max_q_values = {}
        for state, q_value in q_values.items():
            max_q_values[state] = np.max(q_value)

        # Step 4: Save the max_q table into a csv file
        self.save_max_q_values(max_q_values, 'max_q_values.csv')

    def save_max_q_values(self, max_q_values, filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['State', 'Max_Q_Value'])
            for state, max_q_value in max_q_values.items():
                writer.writerow([state, max_q_value])
        print("Max Q values saved to", filename)

    def compute_entropy(self, policy):
        # Entropy of the prob distr for policy stability. (The sum of the % distribution multiplied by the logarithm -in base 2- of p)
        #q = policy
        #qlog = np.log2(q)
        #infs = np.where(np.isinf(qlog))
        #qlog[infs] = 0.
        #qqlog = q*qlog
        #qsum = -np.sum(qqlog)
        #self.entropy = qsum
        self.entropy = np.sum(-policy * np.log2(policy + 1e-12))  # avoid log(0) by adding a small constant

    # Couplet expects a list with [state, action]; Goal is -1 or 1 indicating aversive or appetitive goal has been reached.
    def update_STM(self, couplet=[]):

        # Update STM buffer with the new couplet (FIFO).
        self.STM.append(couplet)
        self.STM = self.STM[1:] # renew the STM buffer by removing the first value of the STM
        #print ("STM: ", self.STM[-1])

    def update_sequential_bias(self):
        # NEW: Update the last actions index first!
        self.last_actions_indx = np.copy(self.selected_actions_indx).tolist()  # Updates the last action indexes with the current actions indexes.
        #print ("last_actions_indx ", self.last_actions_indx)

        # Update trigger values.
        if (len(self.tr) > 0) and self.sequential_bias:
            self.tr = (np.array(self.tr) * (1. - self.alpha_tr)) + self.alpha_tr  # trigger values decay by default
            self.tr[(self.tr < 1.)] = 1.       # all trigger values below 1 are reset to 1.
            tr_last_actions_indx = np.array(self.last_actions_indx)
            self.tr[tr_last_actions_indx] = 1.    # NEW: the trigger value of previously selected segments are reset to 1!!!
            last_actions_shifted = np.roll(self.last_actions_indx, 1, axis=1) # shift the matrix one step to the right
            last_actions_shifted[:, 0] = False  # set the first value of each sequence to False

            # NEW: increase ONLY the trigger value of the next element in sequence (after the ones selected before)!
            tr_change_indx = np.array(last_actions_shifted)
            self.tr[tr_change_indx] += 0.01    # NEW: increase by an arbitrary amount (this amount should be tuned or modified).
            self.tr = self.tr.tolist()

            ## TO-DO ADD FORGETTING OF SEQUENCES BASED ON TRIGGER VALUES.

    def reset_memory(self):
        # MEMORY RESET when finishing an episode
        self.reset_STM()
        self.reset_sequential_bias()

    def reset_STM(self):
        # Reset STM when beggining a new episode
        #self.STM = [[np.zeros(self.emb_length), np.zeros(2)] for _ in range(self.stm_length)] # pl = prototype length (i.e. dimension of the state vector)
        #self.STM = [[np.zeros(self.emb_length), np.zeros(1)] for _ in range(self.stm_length)] # pl = prototype length (i.e. dimension of the state vector)
        if type(self.action_space) != list:
            #self.STM = [[np.zeros(self.emb_length), np.zeros(1)] for _ in range(self.stm_length)] # pl = prototype length (i.e. dimension of the state vector)
            self.STM = [[[0] * self.emb_length , 0] for i in range(self.stm_length)]
            #print(self.STM)
        else:
            #self.STM = [[np.zeros(self.emb_length), np.zeros(2)] for _ in range(self.stm_length)] # pl = prototype length (i.e. dimension of the state vector)
            self.STM = [[[0] * self.emb_length , [0, 0]] for i in range(self.stm_length)]
            #print(self.STM)

    def reset_sequential_bias(self):
        # Reset trigger values when beggining a new episode
        if (len(self.tr) > 0):
            self.tr = np.array(self.tr)
            self.tr[:] = 1.0
            self.tr = self.tr.tolist()

    def update_LTM(self, reward=0):
        # Verify space of LTM
        self.check_LTM_space()

        #print('REWARD: ', reward)
        #print('REWARD type: ', type(reward))
        reward_float = round(float(reward),2)
        #print('REWARD type: ', type(reward_float))

        # Update LTM if reached goal state and still have free space in LTM.
        if (reward_float > 0) and (len(self.LTM[2]) < self.ltm_length):
            #print('REWARD: ', reward_float)
            #print ("GOAL STATE REACHED! REWARD: ", reward_float)
            self.LTM[0].append([s[0] for s in self.STM])  #append prototypes of STM couplets.
            self.LTM[1].append([a[1] for a in self.STM])  #append actions of STM couplets.
            self.LTM[2].append(reward_float)
            self.tr.append(np.ones(self.stm_length).tolist())
            self.selected_actions_indx.append(np.zeros(self.stm_length, dtype='bool').tolist())
            self.last_actions_indx.append(np.zeros(self.stm_length, dtype='bool').tolist())
            #print("Sequences in LTM", len(self.LTM[2]), ", Sequence length:", len(self.STM))
            self.compute_q_table()

    def check_LTM_space(self):
        # Remove sequences when LTM is full
        if (len(self.LTM[2]) >= self.ltm_length):
            if self.memory_full == False:
                print ("LTM IS FULL!")
                self.memory_full = True
            if self.forget != "NONE":
                #print("FORGETTING ACTIVATED...")
                #print ("CURRENT LTM rewards: ", self.LTM[2])
                self.forget_LTM()

    def forget_LTM(self):
        if self.forget == "FIFO":
            self.LTM[0] = np.delete(np.array(self.LTM[0]),0,0).tolist()
            self.LTM[1] = np.delete(np.array(self.LTM[1]),0,0).tolist()
            self.LTM[2] = np.delete(np.array(self.LTM[2]),0,0).tolist()
            self.tr = np.delete(np.array(self.tr),0,0).tolist()
            self.selected_actions_indx = np.delete(np.array(self.selected_actions_indx),0,0).tolist()
            self.last_actions_indx = np.delete(np.array(self.last_actions_indx),0,0).tolist()
            #print ("FIRST MEMORY SEQUENCE FORGOTTEN")
            #print ("UPDATED LTM rewards: ", self.LTM[2])
        elif self.forget == "SING":
            idx = np.argsort(self.LTM[2])
            self.LTM[0] = np.delete(np.array(self.LTM[0]),idx[0],0).tolist()
            self.LTM[1] = np.delete(np.array(self.LTM[1]),idx[0],0).tolist()
            self.LTM[2] = np.delete(np.array(self.LTM[2]),idx[0],0).tolist()
            self.tr = np.delete(np.array(self.tr),idx[0],0).tolist()
            self.selected_actions_indx = np.delete(np.array(self.selected_actions_indx),idx[0],0).tolist()
            self.last_actions_indx = np.delete(np.array(self.last_actions_indx),idx[0],0).tolist()
            #print ("LOWEST REWARD SEQUENCE FORGOTTEN")
            #print ("UPDATED LTM rewards: ", self.LTM[2])
        elif self.forget == "PROP":
            maxfgt = int(len(self.LTM[2]) * self.forget_ratio)
            idx = np.argsort(self.LTM[2])
            self.LTM[0] = np.delete(np.array(self.LTM[0]),idx[0:maxfgt],0).tolist()
            self.LTM[1] = np.delete(np.array(self.LTM[1]),idx[0:maxfgt],0).tolist()
            self.LTM[2] = np.delete(np.array(self.LTM[2]),idx[0:maxfgt],0).tolist()
            self.tr = np.delete(np.array(self.tr),idx[0:maxfgt],0).tolist()
            self.selected_actions_indx = np.delete(np.array(self.selected_actions_indx),idx[0:maxfgt],0).tolist()
            self.last_actions_indx = np.delete(np.array(self.last_actions_indx),idx[0:maxfgt],0).tolist()
            #print ("NUMBER OF FORGOTTEN SEQUENCES: ", maxfgt)
            #print ("UPDATED LTM rewards: ", self.LTM[2])

    def get_memory_length(self):
        # In single memory units
        ltm_len =  len(self.LTM[2]) * self.stm_length
        return ltm_len

    def save_LTM(self, savePath, ID, n=1):
        with open(savePath+ID+'ltm'+str(len(self.LTM[2]))+'_'+str(n)+'.pkl','wb') as f:
            pkl.dump(self.LTM, f)

    def load_LTM(self, filename):
        ID = '/LTMs/'+filename
        # open a file, where you stored the pickled data
        file = open(ID, 'rb')
        # load information from that file
        self.LTM = pkl.load(file)
        print("LTM loaded!! Memories retrieved: ", len(self.LTM[2]))
        # close the file
        file.close()
        # generate trigger values matrix accordingly
        for s in (self.LTM[2]):
            self.tr.append(np.ones(self.stm_length).tolist())
            self.selected_actions_indx.append(np.zeros(self.stm_length, dtype='bool').tolist())
            self.last_actions_indx.append(np.zeros(self.stm_length, dtype='bool').tolist())

    # All about saving
    def toggle_save(self, **kwargs) -> None:
        """
        Toggles save. If the agent was saving its status so far, it sops doing so. Otherwise, it begins to do so,
        by already storing a snapshot of the current state as well.
        Important to note that this function can also be called to extend the saved table in case a new state is
        encountered.
        :param kwargs:
            save_on: If instead of toggling, we want to make sure to turn it on [True] or off [False], we can
        :return:
        """
        save_on = kwargs.get('save_on', not self._save_agent)
        if save_on:
            try:
                try:
                    s = self._states[-1]  # We need to make it understandable for the environment
                except AttributeError:
                    s = self._nS - 1
                if f'Q_{s}_0' not in self._events.columns:  # We added a new state
                    for s_idx in range(self._nS):
                        for a_idx in range(self._nA):
                            try:
                                s = self._states[s_idx]  # We need to make it understandable for the environment
                            except AttributeError:
                                s = s_idx
                            if f'Q_{s}_{a_idx}' not in self._events.columns:
                                self._events[f'Q_{s}_{a_idx}'] = np.full((self._events.shape[0], 1), np.nan)
                                self._events[f'Ur_{s}_{a_idx}'] = np.full((self._events.shape[0], 1), np.nan)
                                self._events[f'Ut_{s}_{a_idx}'] = np.full((self._events.shape[0], 1), np.nan)
                                self._events[f'C_{s}_{a_idx}'] = np.full((self._events.shape[0], 1), np.nan)

            except AttributeError:  # There is no such thing as _events
                poss_states = range(self._nS)
                try:
                    poss_states = self._states  # Thus we won't need to translate
                except AttributeError:
                    pass
                Q_names = [f'Q_{s_idx}_{a_idx}' for s_idx in poss_states for a_idx in range(self._nA)]
                Ur_names = [f'Ur_{s_idx}_{a_idx}' for s_idx in poss_states for a_idx in range(self._nA)]
                Ut_names = [f'Ut_{s_idx}_{a_idx}' for s_idx in poss_states for a_idx in range(self._nA)]
                C_names = [f'C_{s_idx}_{a_idx}' for s_idx in poss_states for a_idx in range(self._nA)]
                self._events = pd.DataFrame(columns=['iter', 'step', 's', 'u', 's_prime', 'r', 'hr', 'ht', 'deltaC',
                                                     *Q_names, *Ur_names, *Ut_names, *C_names])
            if not self._save_agent:
                self._save_agent = True
                self.__save_step__(True)  # If we just turned it on, we take a snapshot of the agent's current state
        else:
            self._save_agent = False

    def dump_agent(self, **kwargs) -> None:
        """
        Saves everything that we have stored into 2 different files: one for the agent, and one for the events.
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

        # 1) Save the whole agent
        # file = open(f'{path}agent{label}.txt', 'wb')
        # pickle.dump(self.__dict__, file, 2)
        # file.close()

        # 2) Save the events
        try:
            self._events.to_csv(f'{path}SORB_agent{label}.csv', sep=',', index=False, encoding='utf-8')
        except AttributeError:
            print('Note: This agent does not store the transpired events, no .csv generated.')

    def load_agent(self, file_name: str, **kwargs):
        """
        Loads a previously saved agent
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