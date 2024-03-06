from env import *
from math import *

# If we don't know the state space in advance, adding to _events memory dataframe will produce a performance warning due
# to "fragmentation". We, however, cannot be bothered at the moment.
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def entropy(proba: np.ndarray) -> float:
    """
    Computes the entropy of a vector of probabilities
    :param proba: said vector of probabilities
    :return: the corresponding entropy
    """
    if proba.sum() < 1 - np.finfo(np.float32).eps or 1 + np.finfo(
            np.float32).eps < proba.sum():  # Considering precision
        raise ValueError("Probabilities of outcomes must sum to 1.")
    proba = proba[proba != 0]
    return sum([-1 * p * log2(p) for p in proba])


class RLagent:
    """
    The agent class, capable of storing the metadata describing its model and the learned Q values. Furthermore, it will
    be capable of choosing an action based on different criteria, updating its Q values, and remembering and replaying
    its memory.
    """

    def __init__(self, mdp: Env, model_type: str, gamma: float, kappa: float, decision_rule: str, **kwargs):
        """
        Constructor for the basic instance of a Reinforcement Learning Agent.
        Exceptions: ValueError, if the str parameters are invalid
        :param mdp: the environment
        :param model_type: temporal difference ['TD'], value iteration ['VI'] or policy iteration ['PI']
        :param gamma: discounting factor [float]
        :param kappa: decay factor of EWMA computation of the reward function. Theoretically this is only needed for the
            MB agent, however, the computation of the entropy will necessitate its use in all cases
        :param kwargs: parameters regarding the decision rule, the replay and the epistemic rewards
            Regarding the model:
                alpha: learning rat eif model_type is 'TD' [float]
                beta: for softmax decisions (non-optional) [float]
                epsilon: for epsilon-greedy decisions (non-optional) [float]
                known_env: is the number of states previously known [True] or not [False, default]
            Regarding the replay:
                replay_type: "forward", "backward", "priority", "trsam", "bidir" (optional) [str]
                    event_handle: what should we compare a new event to when trying to estimate if we need to
                        overwrite an old memory or not: states ['s'], state-action ['sa'] or
                        state-action-new state ['sas']. Only needed if replay_type is "priority" or "bidir"
                    event_content: what should we replay, states ['s'], state-action ['sa'],
                        state-action-new state ['sas'], or state-action-new state-reward ['sasr', default].
                    replay_thresh: the replay threshold (minimum NORMALIZED error, optional) [float]
                    max_replay: the number of replay steps we'll take (optional) [int]
                    add_predecessors: if replay type is "priority" or "bidir", should I perform a predecessor search
                        when I add an element to the memory buffer ["act": after a real action, "rep": after a replay
                        step, "both": after both] (optional)
                    forbidden_walls: if replay_type is trsam or bidir; or if event_content is 's' (the action needs to
                        be selected) - is bumping into a wall forbidden during simulation [True, default] or not [False]
            Regarding the epistemic values:
                dec_weights: an array of float weights (summing up to 1) precising the contribution of the different
                    quality values to the C-value based on which decisions are made [w_Q, w_Ur, w_Ut]
                rep_weights: an array of float weights (summing up to 1) precising the contribution of the different
                    quality values to the C-value based on which replayed states are prioritized [w_Q, w_Ur, w_Ut]. This
                    is only relevant in case of prioritized sweeping and trajectory sampling. If not specified it will
                    default to dec_weights.
        """

        if model_type not in ["TD", "VI", "PI"]:
            raise ValueError("Model type '" + model_type + "' is not supported by the agent.")
        if decision_rule not in ["softmax", "max", "epsilon"]:
            raise ValueError("Decision rule '" + decision_rule + "' does not exist")

        # The parameters of the environment
        known_env = kwargs.get('known_env', False)
        if known_env:
            self._nS = mdp.state_num()  # number of possible states
        else:
            self._nS = 1  # We might only know a single state of the environment (the one we're in)
            self._states = np.array([mdp.curr_state()])  # an array of sates. The idx of each state is my state label
        self._nA = mdp.act_num()  # max number of possible actions per state
        self._maxrew = mdp.max_rew()  # max value of the achievable reward

        # The parameters of the agent
        self._gamma = gamma
        self._kappa = kappa
        self._model_type = model_type
        self._decision_rule = decision_rule
        if self._decision_rule == "softmax":
            self._beta = kwargs.get("beta", None)
            if self._beta is None:
                raise ValueError("Decision rule 'softmax' requires temperature parameter 'beta'")
        elif self._decision_rule == "epsilon":
            self._epsilon = kwargs.get("epsilon", None)
            if self._epsilon is None:
                raise ValueError("Decision rule 'epsilon' requires exploration parameter 'epsilon'")
        if self._model_type == 'TD':
            self._alpha = kwargs.get('alpha', None)
            if self._alpha is None:
                raise ValueError('The temporal difference error algorithm necessitates a learning parameter alpha.')

        # The agent's understanding of the environment
        # self._maxval = np.array([self._maxrew / (1 - self._gamma),  # Qmax
        #                          entropy(np.ones(2) / 2) / (1 - self._gamma),  # Ur max
        #                          entropy(np.ones(self._nS) / self._nS) / (1 - self._gamma)])  # Ut max
        self._maxval = np.array([0,  # Qmax
                                 entropy(np.ones(2) / 2) / (1 - self._gamma),  # Ur max
                                 entropy(np.ones(self._nS) / self._nS) / (1 - self._gamma)])  # Ut max
        self._N = np.zeros((self._nS, self._nA, self._nS))  # Number of visits
        self._T = np.ones((self._nS, self._nA, self._nS)) / self._nS  # Transition probs
        self._Pr = np.ones((self._nS, self._nA)) / 2  # Reward probs (not the expected reward but the proba of r > 0)
        # TODO R initialization
        self._R = np.zeros((self._nS, self._nA, 3))  # Reward function [R, deltaHr, deltaHt]
        # self._R[:, :, 0] = self._maxrew  # R is initialized to max for optimism
        self._R[:, :, 1] = entropy(np.ones(2) / 2)  # Delta Hr is initialized to max for optimism
        self._R[:, :, 2] = entropy(np.ones(self._nS) / self._nS)  # Delta Ht is initialized to max too
        self._C = np.zeros((self._nS, self._nA, 3))  # Quality values [Q, Ur, Ut]
        # TODO C initialization
        # self._C[:, :, 0] = self._maxval[0]  # Q is initialized to max for optimism
        self._C[:, :, 1] = self._maxval[1]  # Ur is initialized to max for optimism
        self._C[:, :, 2] = self._maxval[2]  # Ut is initialized to max too
        self._pi = np.random.randint(self._nA, size=self._nS)  # policy (what action to choose in each state)
        self._maxval[self._maxval == 0] = 1  # Since we'll only use _maxval for normalization, it cannot contain 0s

        # On the replay
        self._replay_type = kwargs.get("replay_type", None)
        if self._replay_type not in [None, "forward", "backward", "priority", "trsam", "bidir"]:
            raise ValueError(f"Replay type {self._replay_type} is not a valid value.")
        if self._replay_type is not None:
            # Should I consider an event to be described by a state-action couple (True), or only by the state (False)
            self._event_content = kwargs.get("event_content", 'sasr')
            self._replay_thresh = kwargs.get("replay_thresh", 0)
            if self._replay_thresh < 0:
                raise ValueError('Replay threshold needs to be non-negative.')
            self._max_replay = kwargs.get("max_replay", None)
            if self._replay_thresh == 0 and self._max_replay is None:
                raise ValueError("Either the replay threshold or the maximum number of replay steps "
                                 "needs to be specified.")
            if self._replay_type in ["forward", "backward"] and self._max_replay is None:
                # TODO is this really how we want it? Just forbid fd/bd an infinite memory?
                raise ValueError(f"Replay type '{self._replay_type}' is incompatible with an infinite memory.")
            self._add_predecessors = None  # Only updated in MB agent
            # Memory: state (s), action (u), new state (s_prime), reward (r), surprise (delta)
            # formatted like a table, the first element is the one replayed
            # we need it except for trsam which will generate its own states to replay
            if self._replay_type != "trsam":
                # We need a replay buffer in the shape of [s, a, s', r, hr, ht, deltaC]
                if self._max_replay is not None:
                    if self._replay_type != 'bidir':
                        self._memory_buff = np.zeros((self._max_replay, 7), dtype=float)
                    else:
                        self._memory_buff = np.zeros((floor(self._max_replay / 2), 7), dtype=float)
                    # Don't store more than what we can truly replay
                else:
                    self._memory_buff = np.zeros((0, 7), dtype=float)
                    # This one can be extended infinitely
            if self._replay_type in ["priority", "bidir"]:
                self._event_handle = kwargs.get("event_handle", None)
                if self._event_handle is None:
                    raise ValueError("If we use prioritized replay, event_handle need sto be defined.")
                if self._event_handle not in ['s', 'sa', 'sas']:
                    raise ValueError("Event handle needs to be 's', 'sa' or 'sas")
                self._add_predecessors = kwargs.get("add_predecessors", None)
                if self._add_predecessors not in [None, "act", "rep", "both"]:
                    raise ValueError("Predecessors can be added after action 'act', replay 'rep', or both 'both'.")
            self._forbidden_walls = kwargs.get('forbidden_walls', True)

        # For the epistemic rewards
        self._dec_weights = kwargs.get('dec_weights', np.array([1, 0, 0]))  # Q-value, Ur-value and Ut-value weights
        if np.sum(self._dec_weights) < 1 - np.finfo(np.float32).eps or 1 + np.finfo(
                np.float32).eps < np.sum(self._dec_weights):
            raise ValueError('The weights of the different quality values must sum up to 1.')
        self._rep_weights = kwargs.get('rep_weights', self._dec_weights)  # Q-value, Ur-value and Ut-value weights
        if np.sum(self._rep_weights) < 1 - np.finfo(np.float32).eps or 1 + np.finfo(
                np.float32).eps < np.sum(self._rep_weights):
            raise ValueError('The weights of the different quality values must sum up to 1.')

        # Regarding the dataframe of all events that happened during the lifetime of the agent
        self._save_agent = False  # Nothing is saved unless saving is specifically toggled
        self._events = None  # This will be the event memory if needed
        return

    # Methods related to the inner workings of the agent
    def __translate_s__(self, s: int) -> int | None:
        """
        Translates s into labels recognized by the agent (in case of automatically increasing state space)
        :param s: the state as detected from the environment
        :return: the state label recognized by the agent
        """
        if s is None:
            return s
        try:
            return np.argwhere(self._states == s)[0, 0]
        except AttributeError:
            return s

    def __entropy__(self, s: int, a: int, mode: str) -> float | None:
        """
        Computes the entropy under the specified circumstances
        :param s: The state
        :param a: The action
        :param mode: Do we compute reward-uncertainty ['rew'], or transition-uncertainty ['trans'] related entropy?
        :return: the entropy. If we requested a type of entropy the agent does not keep track of (e.g. 'rew' while the
            agent does not even consider rew entropy, the return value is None)
        """
        if mode not in ['rew', 'trans']:
            raise ValueError('Only transition and reward entropy can be computed this way.')
        if mode == 'rew':
            return entropy(np.array([self._Pr[s, a], 1 - self._Pr[s, a]]))
        return entropy(self._T[s, a, :])

    def __combine_C__(self, **kwargs) -> np.ndarray | float:
        """
        Combines the normalized Q and U values based on the pre-defined ratio for state. This will only be used
        to assess the priority of certain updates, that is Q values will be stored and updated in the usual fashion.
        This function can be used with 2 different sets of input variables (s, a; or Q, Ur, Ut)
        If the (s, a) couple is defined, we will compute the exact C value (norm in the C-space). If it is the
        (Q, Ur, Ut) vector, we consider it a delta C vector, and in that case we take the ABSOLUTE VALUE of each
        dimension to compute the ABSOLUTE CHANGE.
        :return: the combined value normalized between 0 and 1; or the combined delta value normalized between -1 and 1
        :param kwargs:
            s, a: the state-action couple whose C-value we want to compute [int] OR
            deltaC: a vector contianing the Q, Ur and Ut values [np.ndarray of float]
            replay: a boolean deciding whether we're combining quality values for replay purposes or not. If [True],
                rep_weights will be used (to use during every inference step). If [False], dec_weights will be used (to
                use during action selection, if it takes place in the real world). Defaults to [False].
        """
        weights = self._dec_weights
        if kwargs.get('replay', False):
            weights = self._rep_weights
        C_vector = kwargs.get('deltaC', None)
        if C_vector is None:
            s = kwargs.get('s', None)
            a = kwargs.get('a', None)
            if s is None or a is None:
                raise ValueError('To combine the C value, either the quality values or a state-action couple needs to '
                                 'be specified.')
            C_vector = self._C[s, a, :]
        else:
            C_vector = abs(C_vector)
        return np.sum(C_vector / self._maxval * weights)

    def __amax_C__(self) -> np.ndarray:
        """
        Returns an array of (nS x 3), where for each state we store the highest C(s, a) value where a is a legal action
        :return: the array mentioned above
        """
        max_C = np.zeros((self._C.shape[0], self._C.shape[2]))
        for s_idx in range(self._nS):
            a_poss = self.__find_good_actions__(s_idx)
            if len(a_poss) == 0:
                # TODO if we have no possible action, should we return the max over all actions, or maybe 0?
                a_poss = np.array(range(self._nA))
            C_temp = self._C[s_idx, a_poss, :]
            max_C[s_idx, :] = np.amax(C_temp, axis=0)
        return max_C

    def __pop_memory__(self) -> np.ndarray:
        """
        A function that pops (and consequently deletes) the top enry in the memory buffer.
        :return: The stored event
        """
        if self._replay_type not in ['priority', 'bidir']:
            raise RuntimeWarning('Popping from the memory is only recommended when the memory is prioritized.')

        # First we take the stored event
        event = np.copy(self._memory_buff[0, :])

        # Then we delete the row it was stored in
        if self._max_replay is not None:
            # If we have finite capacity, we will have to zero this row out and shift it down
            self._memory_buff[0, :] = np.zeros((1, 7))  # delete the pre-existing copy
            self._memory_buff[0:, :] = np.roll(self._memory_buff[0:, :], -1, axis=0)
        else:
            # If we have infinite capacity, we have to actually delete it
            self._memory_buff = np.delete(self._memory_buff, obj=0, axis=0)

        return event

    def __store_in_memory__(self, s: int, a: int, s_prime: int, rew: np.ndarray, deltaC: float) -> None:
        """
        Takes an action and stores it in the memory buffer. The memory buffer is a numpy array, where each row is a new
        memory. The rows contain [state, action, new_state, rew, r-epist_rew, t-epist_rew, C_value_difference]
        During replay, it's always the first element in the array that gets replayed first. This means, that for forward
        replay, the top row will contain the oldest element, for backwards replay it will contain the newest memory, and
        for prioritized, it will contain the most surprising memory.
        The new memory can be added to the buffer, or if a copy of it already exists in the buffer, then it can be
        overwritten by a more significant entry [prioritized] or duplicated [forward/backwards].
        :param s: current state idx
        :param a: taken action
        :param s_prime: arrival state
        :param rew: gained reward [r, hr, ht]
        :param deltaC: change in the (combined) normalized Q value
        :return: -
        """
        if self._replay_type == "trsam":
            return  # No need to store anything

        # 1) Sub-threshold elements won't be stored for pr/bidir
        if abs(deltaC) <= self._replay_thresh and self._replay_type in ['priority', 'bidir']:
            # TODO we might want to change it so that only the significant elements are stored for fd/bd too
            return
        to_store = np.array([[s, a, s_prime, rew[0], rew[1], rew[2], deltaC]])  # this is our new row
        empty_idx = np.where(np.all(self._memory_buff == 0, axis=1))[0]  # IDX of all empty rows

        # 2) Now we need to see if the new element will overwrite an old copy or be added to the buffer
        memory_idx = None
        if self._replay_type in ['priority', 'bidir']:
            if self._event_handle == 'sas' and np.any(
                    np.logical_and(np.logical_and(self._memory_buff[:, 0] == s, self._memory_buff[:, 1] == a),
                                   self._memory_buff[:, 2] == s_prime)):
                memory_idx = \
                    np.where(np.logical_and(np.logical_and(self._memory_buff[:, 0] == s, self._memory_buff[:, 1] == a),
                                            self._memory_buff[:, 2] == s_prime))[0]
            elif self._event_handle == 'sa' and np.any(
                    np.logical_and(self._memory_buff[:, 0] == s, self._memory_buff[:, 1] == a)):
                memory_idx = np.where(np.logical_and(self._memory_buff[:, 0] == s, self._memory_buff[:, 1] == a))[0]
            elif self._event_handle == 's' and np.any(self._memory_buff[:, 0] == s):
                memory_idx = np.where(self._memory_buff[:, 0] == s)[0]

        # TODO I allow for fd/bd to store an element twice, and only forbid it for pr/bidir in the next line. Is that OK
        if self._replay_type in ['priority', 'bidir'] and memory_idx is not None and memory_idx.size != 0:
            # 2.a) It is pr/bidir, and we found a copy, so we'll have to REPLACE it. This will mean that we'll first
            # remove the old copy, and then add the new one just like normal

            # 2.a.1) First question: is the original copy better, or do I want to overwrite?
            if abs(self._memory_buff[memory_idx[0], -1] > abs(deltaC)):
                # If the original copy is better, we keep it
                # TODO maybe we want to overwrite the pre-existing copy for pr/bidir, even if the new is worse? I think
                #  not, because that could mean we might want to even delete the entry, which is a problem; plus we
                #  might end up decreasing the importance of a state cuz we took an insignificant action
                return
            # Otherwise just scrape the old row and shift the elements so that the empty row is at the
            # bottom of the array
            self._memory_buff[memory_idx[0], :] = np.zeros((1, 7))  # delete the pre-existing copy
            self._memory_buff[memory_idx[0]:, :] = np.roll(self._memory_buff[memory_idx[0]:, :], -1, axis=0)
            # Now the bottom row is empty for sure

            # 2.a.2) If we're here that means that we certainly created an empty row, so empty_idx needs to be adjusted
            if empty_idx.size != 0:
                # If we deleted a row from a non-full buff, the IDX of the first empty row decreased
                empty_idx -= 1
            else:
                # If it was full but not infinite, now the last row is empty for sure, whether the capacity is infinite
                # or not.
                empty_idx = np.array([self._memory_buff.shape[0] - 1])

        else:
            # 2.b) It's either pr/bidir and no pre-existing copy was found in the buffer, or it is not prioritized --
            # this means we have to ADD the new element to the buffer no matter what
            if empty_idx.size == 0 and self._max_replay is None:
                # 2.b.1) If full but infinite, we expand. Otherwise, nothing will happen
                self._memory_buff = np.append(self._memory_buff, np.zeros((1, 7)), axis=0)
                empty_idx = np.array([self._memory_buff.shape[0] - 1])

        # 3) Now all we have to do is insert the new element at its proper place
        if self._replay_type in ["backward", "priority", "bidir"]:
            # 3.a) These algorithms prefer to insert more significant elements towards the top of the buffer
            insertion_idx = 0  # backwards -- put it on the front
            if self._replay_type in ["priority", "bidir"]:
                # Pr and bidir want to find the location based on priority
                insertion_idx = np.where(abs(self._memory_buff[:, -1]) < abs(deltaC))[0]
                if insertion_idx.size == 0:  # if full and all the stored elements are more important
                    return
                else:
                    insertion_idx = insertion_idx[0]

            # Shift everything down below the insertion idx, then overwrite
            self._memory_buff[insertion_idx:, :] = np.roll(self._memory_buff[insertion_idx:, :], 1, axis=0)
            self._memory_buff[insertion_idx, :] = to_store
            return
        elif self._replay_type == "forward":
            # 3.b) These algorithms prefer to insert more significant elements towards the end of the buffer
            insertion_idx = empty_idx  # IDX of first empty row
            if insertion_idx.size == 0:
                # If full, we'll have to insert to the bottom of the table. Remember that if we have infinite
                # capacity, we already inserted an empty row at the bottom, so this branch does not execute
                # Thus: since we are full and finite, shift up and overwrite the last element
                self._memory_buff = np.roll(self._memory_buff, -1, axis=0)
                insertion_idx = np.array([-1])
            # If not full, or already shifted up, just insert the element to the end
            self._memory_buff[insertion_idx[0], :] = to_store
            return

    def __extend_state_space__(self, s_prime) -> None:
        """
        Upon encountering a never-before seen state, this function extends the state-space
        :param s_prime: the label of the new state
        :return:
        """
        # Since we got a new state, the transition model needs to be rescaled completely; as for the reward model, the
        # new state is maximally uncertain
        maxval_old = self._maxval[2]
        self._maxval[2] = entropy(np.ones(self._nS + 1) / (self._nS + 1)) / (1 - self._gamma)
        self._maxval[1] = entropy(np.ones(2) / 2) / (1 - self._gamma)

        # First the C values
        self._states = np.append(self._states, np.array([s_prime]), axis=0)
        self._C[:, :, 2] *= self._maxval[2] / maxval_old  # The Ut values have to be rescaled
        # TODO decide on the initialization of C values
        self._C = np.append(self._C, np.array([[[0, self._maxval[1], self._maxval[2]]
                                                for _ in range(self._nA)]]), axis=0)

        # Now the model
        self._N = np.append(self._N, np.zeros((self._nS, self._nA, 1)), axis=2)
        self._N = np.append(self._N, np.zeros((1, self._nA, self._nS + 1)), axis=0)
        # self._T *= self._nS / (self._nS + 1)  # Rescaling the probabilities
        # self._T = np.append(self._T, 1 - np.sum(self._T, axis=2, keepdims=True), axis=2)  # Transition to new state
        # self._T = np.append(self._T, np.ones((1, self._nA, self._nS + 1)) / (self._nS + 1),
        #                     axis=0)  # Tr from new state
        # TODO if we intialize T to random, during replay we'll have impossible state couples
        self._T = np.append(self._T, np.zeros((self._nS, self._nA, 1)), axis=2)  # Transition to new state
        self._T = np.append(self._T, np.ones((1, self._nA, self._nS + 1)) / (self._nS + 1),
                            axis=0)  # Tr from new state
        self._Pr = np.append(self._Pr, np.ones((1, self._nA)) / 2, axis=0)
        # TODO decide on the initialization for R
        # R will be initialized to [0, max, max]
        self._R = np.append(self._R,
                            np.array([[[0, entropy(np.ones(2) / 2), entropy(np.ones(self._nS) / self._nS)]
                                       for _ in range(self._nA)]]), axis=0)

        self._nS += 1
        self.toggle_save(save_on=self._save_agent)  # To extend the saved dataframe too
        return

    # Hidden methods for MF learning
    def __temporal_difference_error__(self, s: int, a: int, s_prime: int, rew: np.ndarray) -> float:
        """
        The TDE algorithm. It simply updates the Q  values in a model-free fashion, and returnns the prediciton
        error
        :param s: current state label
        :param a: chosen action
        :param s_prime: arrival state label
        :param rew: received reward [r, hr, ht]
        :return: prediction error
        """
        # TD_error = rew + self._gamma * np.max(self._C[s_prime, :, :], axis=0) - self._C[s, a, :]
        C_max = self.__amax_C__()  # This way impossible actions (with U-value initialized to max) won't affect it
        TD_error = rew + self._gamma * C_max[s_prime] - self._C[s, a, :]
        self._C[s, a, :] += self._alpha * TD_error
        # Now we want to return the norm of the TD error, combined in a way to make it comparable to the replay buffer
        return self.__combine_C__(deltaC=TD_error, replay=True)

    # Hidden methods for MB learning
    def __MB_update__(self, s: int, a: int, val_func: np.ndarray) -> np.ndarray:
        """
        The heart of the model based update, computing the new C value. Returns said value in a three-by-one array???
        :param s: current state label
        :param a: action taken
        :param val_func: value function by which we update (Q or V, depending the self._model_type)
        :return: the new C value in a 3-by-1 array???
        """
        return self._R[s, a, :] + self._gamma * np.dot(np.reshape(self._T[s, a, :], (1, self._nS)), val_func)

    def __value_iteration__(self, s: int, a: int) -> float:
        """
        The value iteration algorithm
        :param s: current state
        :param a: action chosen
        :return: the difference in the Q value after update
        """
        # C_max = np.amax(self._C, axis=1)  # [nS x 3] array instead of [nS x 1 x 3] (otherwise use keepdims=True)
        C_max = self.__amax_C__()  # [nS x 3] array, impossible actions (with Uvalue initialized to max) won't affect it
        C_old = np.copy(self._C[s, a, :])
        self._C[s, a, :] = self.__MB_update__(s=s, a=a, val_func=C_max)
        # Now we want to return the norm of the TD error, combined in a way to make it comparable to the replay buffer
        return self.__combine_C__(deltaC=self._C[s, a, :] - C_old, replay=True)

    def __policy_iteration__(self, s: int, u: int) -> Tuple[float, np.ndarray]:
        """
        The policy iteration algorithm.
        :param s: current state
        :param u: chosen action
        :return: the difference in the Q value after update
        """
        pass
        # TODO algorithm does not converge -- we need to add an update to the policy
        # R = np.reshape(np.array([self._R[idx_s, self._pi[idx_s]] for idx_s in range(self._nS)]), (self._nS, 1))
        # T = np.array([[self._T[idx_s, self._pi[idx_s], idx_s_prime] for idx_s_prime in range(self._nS)]
        #               for idx_s in range(self._nS)])
        # V = np.linalg.lstsq(np.eye(self._nS) - np.dot(self._gamma, T), R, rcond=None)
        # Q_old = self._Q[s, u]
        # self._Q[s, u] = self.__MB_update__(s=s, u=u, val_func=V[0])
        # return self._Q[s, u] - Q_old, V[0]

    # Hidden methods for MB replay
    def __isrewarded__(self, s: int) -> bool:
        """
        Returns whether a given state is rewarded or not (assuming that every real state transition is more
        probable than 1/nS)
        :param s: The state in question
        :return: Rewarded [True] or not [False]
        """
        for a_pred in range(self._nA):
            s_pred = self._T[:, a_pred, s]
            s_pred = np.nonzero(np.logical_and(s_pred > 0, self._N[:, a_pred, s] > 0))
            if s_pred[0].size != 0 and np.any(self._R[s_pred, a_pred, 0] > 0):  # if we can get here AND rewarded
                return True
        return False

    def __find_predecessors__(self, s: int, p: float) -> None:
        """
        It can speed up the prioritized replay-based learning, to add predecessor states to the memory,
        so that is exactly what this function does. For a given state, we look at all predecessors, that are seemingly
        more likely than chance levels, and add them to the buffer with a priority discounted from that of state s.
        This necessitates the existence of a world model, so technically it's cheating for MF methods, but this will not
        be of our main concern for the time being.
        :param s: current state
        :param p: the priority of the current state/state-action pair
        :return: -
        """

        if abs(p) <= self._replay_thresh:
            return

        # We will go through all actions that might lead to this state to find the corresponding predecessors
        for a_pred in range(self._nA):
            # 1) For every possible "predecessor" step, leading to s we find all predecessor states
            s_all = np.array(range(self._nS))
            s_mask_reachable = np.logical_and(self._N[:, a_pred, s] > 0, self._T[:, a_pred, s] > 0)
            s_all = s_all[s_mask_reachable]
            for s_pred in s_all:
                # 2) For all predecessor states we back propagate the priority level. store_in_memory will ofc not store
                # sub-threshold events
                p_pred = p * self._gamma * self._T[s_pred, a_pred, s]
                self.__store_in_memory__(s_pred, a_pred, s, self._R[s_pred, a_pred, :], p_pred)

    def __find_good_actions__(self, s: int, **kwargs) -> np.ndarray:
        """
        Finds the desired (or at least possible) actions from a given state based on the model.
        :param s: The current state
        :param kwargs:
            prev_s: state we don't want to visit if possible
        :return: an array of the possible actions
        """
        prev_s = kwargs.get('prev_s', None)
        # 1) We choose an action based on the model -- this should not be the action to go back, unless necessary
        # What is really important here is that we want to take a step that doesn't just take us back to our last
        # state, unless that is absolutely inevitable. Furthermore, we should be able to notice if we cannot predict
        # where the agent might end up next -- this means that this particular step has never been taken during
        # learning, and thus it might as well be illegal, so it should also be avoided, if possible.
        a_s_prime = self._T[s, :, :]  # The matrix telling us the transition probabilities
        poss_moves = np.array(range(self._nA))  # moves that are physically possible, as far as we know
        # 1.a) let's remove all illegal steps (according to our knowledge)
        for a_idx in range(self._nA):
            illegal = np.sum(self._N[s, a_idx, :]) == 0  # haven't taken this step yet
            if self._forbidden_walls:
                illegal = illegal or np.argmax(a_s_prime[a_idx, :]) == s
            if illegal:
                poss_moves = np.delete(poss_moves, poss_moves == a_idx)
        # 1.b) now remove the steps that'd take us back to the previous or current state (if we can)
        # TODO maybe this whole section is useless, we should only use this function to filter out impossible steps
        good_moves = np.copy(poss_moves)  # moves that are possible AND take us further, as far as we know
        stay_in_place = True
        for a_idx in poss_moves:
            if np.argmax(a_s_prime[a_idx, :]) == prev_s:  # If I were to go back, delete
                good_moves = np.delete(good_moves, good_moves == a_idx)
            elif np.argmax(a_s_prime[a_idx, :]) != s:  # If this move allows for actual motion, remember that
                # We're not deleting moves that keep us in place, cuz whether or not walls (actions that result in no
                # motion) are forbidden is decided by the forbidden_walls variable
                stay_in_place = False
        # 1.c) if nothing is left, let's backtrack
        if len(good_moves) > 0 and not stay_in_place:  # if we have good moves that don't keep us in place
            actions_to_choose = good_moves
        elif len(poss_moves) > 0:  # if we have possible moves (even if bad)
            actions_to_choose = poss_moves
        else:
            # actions_to_choose = np.array(range(self._nA))
            actions_to_choose = np.array([])
        return actions_to_choose

    def __trajectory_sampling__(self, s: int, **kwargs) -> None:
        """
        This function performs basic trajectory sampling
        :param s: state we start the simulation from
        :param kwargs:
            stop_loc: state(s) we terminate in [np.ndarray]
            steps: how many steps we are going to simulate [int]
        :return:
        """
        stop_loc = kwargs.get('stop_loc', np.array([]))
        steps = kwargs.get('steps', self._max_replay)

        # 1) We start from the agent's position
        curr_s = s  # current state
        prev_s = s  # previous state
        rew = np.zeros((1, 3))
        max_delta = 0
        it = 0
        while steps is None or it < steps:
            # 2.a) finding the possible actions
            actions_to_choose = self.__find_good_actions__(
                curr_s)  # , prev_s=prev_s) # TODO decide if we are allowed to backtrack

            if len(actions_to_choose) > 0:
                # 2.b) committing to a choice
                a = self.choose_action(curr_s, actions_to_choose, virtual=True)
                # If we need to combine delta, we choose action based on epist values, otherwise not (done in
                # choose_action)
                s_prime = np.random.choice(list(range(self._nS)), p=self._T[curr_s, a, :])

                # 3) And we get a reward
                rew = self._R[curr_s, a, :]

                # 4) We learn
                delta_C = self.inference(curr_s, a, s_prime, rew, virtual=True, update_buffer=False)
                # We don't update the buffer, as the path we're taking is completely imaginary

                # 5) And we consider one step to be done
                it += 1

                # 6) Handle delta and if this state is rewarded (and delta is significant), we go back to start
                if abs(delta_C) > abs(max_delta):
                    max_delta = delta_C
                if curr_s != s_prime:  # If we stay in place, let's not update anything
                    prev_s = curr_s
                    curr_s = s_prime
            if len(actions_to_choose) == 0 or curr_s in stop_loc or rew[0] > 0 or self.__isrewarded__(curr_s):
                if abs(max_delta) > self._replay_thresh:
                    curr_s = s
                    max_delta = 0
                else:
                    break

    # Hidden method concerning the saving
    def __save_step__(self, virtual: bool, **kwargs) -> None:
        """
        Saves the current state of the maze by adding a row to the _events memory.
        :param virtual: is this a virtual step [True] or a real one [False]
        :param kwargs:
            s: last state
            a: last action
            s_prime: current state
            rew: last rewards [r, hr, ht]
            deltaC: (combined) delta Q of the learning step
        :return:
        """
        if not self._save_agent:
            return

        # 1) Which step are we at
        it, step = 0, 0
        if self._events.shape[0] > 0:
            if not virtual:  # If real, it's a new iteration
                it = self._events['iter'].iloc[-1] + 1
            else:  # else it's the same iteration but a new step
                it = self._events['iter'].iloc[-1]
                step = self._events['step'].iloc[-1] + 1

        # 2) Format the event to store
        # iter, step, s, a, s', r, hr, ht, deltaC, Q values, (Ur values, Ut values)
        # for the latter three it's ordered Q(s1, a1), Q(s1, a2), ..., Q(s1, ak), Q(s2, a1), ..., Q(sn, ak)
        event = {'iter': [it], 'step': [step]}
        s = kwargs.get('s', None)
        if s is not None:
            try:
                event['s'] = [self._states[s]]
            except AttributeError:
                event['s'] = [s]
        else:
            event['s'] = [None]
        event['a'] = [kwargs.get('a', None)]
        s_prime = kwargs.get('s_prime', None)
        if s_prime is not None:
            try:
                event['s_prime'] = [self._states[s_prime]]
            except AttributeError:
                event['s_prime'] = [s_prime]
        else:
            event['s_prime'] = [None]
        rew = kwargs.get('rew', [None] * 3)
        event['r'] = [rew[0]]
        event['hr'] = [rew[1]]
        event['ht'] = [rew[2]]
        event['deltaC'] = [kwargs.get('deltaC', None)]

        # Now the Q and U values
        for s_idx in range(self._nS):
            for a_idx in range(self._nA):
                try:
                    s = self._states[s_idx]  # We need to make it understandable for the environment
                except AttributeError:
                    s = s_idx
                event[f'Q_{s}_{a_idx}'] = [self._C[s_idx, a_idx, 0]]
                event[f'Ur_{s}_{a_idx}'] = [self._C[s_idx, a_idx, 1]]
                event[f'Ut_{s}_{a_idx}'] = [self._C[s_idx, a_idx, 2]]
                event[f'C_{s}_{a_idx}'] = [self.__combine_C__(s=s, a=a_idx)]

        # 3) Add it to the table
        events_temp = pd.DataFrame.from_dict(event).fillna(value=np.nan)
        self._events = self._events.copy() if events_temp.empty \
            else events_temp.copy() if self._events.empty \
            else pd.concat([self._events, events_temp], ignore_index=True)
        # If we want to evade a FutureWarning about a possible concatenation between an empty and a non-empty table we
        # need to check for all possibilities
        return

    # Methods used to instruct the agent
    def choose_action(self, s: int, a_poss: np.ndarray, **kwargs) -> int:
        """
        Chooses actions from the available ones from a predefined state and the set of available actions observed from
        env. The action choice will depend on the decision rule. We might use a softmax function, a greedy choice by Q,
        or an epsilon greedy one.
        :param s: state from which we want to take a step
        :param a_poss: array of possible actions, given by the env
        :param kwargs:
            virtual: is this a virtual step (True) or a real one (False, default)
        :return: chosen action
        """
        # Let's see what epistemic values we will have to combine (is this a real decision, or a virtual replay?)
        virtual = kwargs.get('virtual', False)
        if not virtual:
            s = self.__translate_s__(s)

        # Let's make the decision:
        # 1) if epsilon greedy, and we explore
        if self._decision_rule == "epsilon":
            # Simplest case, we choose randomly
            if np.random.uniform(0, 1, 1) <= self._epsilon:
                a = np.random.choice(a_poss)
                return int(a)

        # 2) For the other methods combine all the potential constituents
        C_poss = np.array([self.__combine_C__(s=s, a=idx_a, replay=virtual) for idx_a in a_poss])
        # C_poss is between 0 and 1

        # 3) If we choose to put these combined values through a softmax
        if self._decision_rule == "softmax":
            p_poss = np.exp(self._beta * C_poss) / np.sum(np.exp(self._beta * C_poss))
            a = np.random.choice(a_poss, p=p_poss)
            return int(a)

        # 4) If we choose the maximum (either due to greedy or epsilon greedy policies)
        a_poss = a_poss[C_poss == max(C_poss)]
        a = np.random.choice(a_poss)
        return int(a)

    def model_learning(self, s: int, a: int, s_prime: int, r: float) -> Tuple[float, float]:
        """
        Tuning of the agent's model parameters (MF agent will only use it for the computation of the epistemic rewards)
        :param s: current state label
        :param a: chosen action
        :param s_prime: arrival state label
        :param r: the received reward
        :return: the epistemic rewards (hr, ht)
        """
        # First we might need to extend the model if s_prime is never before seen and if the environment is unknown
        try:
            if s_prime not in self._states:
                self.__extend_state_space__(s_prime)
        except AttributeError:
            pass

        # Since s comes from the environment
        s, s_prime = self.__translate_s__(s), self.__translate_s__(s_prime)

        # Let's store the current entropy values for epistemic rewards
        Hr = self.__entropy__(s, a, 'rew')
        Ht = self.__entropy__(s, a, 'trans')

        # Then we just update
        # a) the Transition function
        self._N[s, a, s_prime] += 1
        self._T[s, a, :] = (1 - 1 / np.sum(self._N[s, a, :])) * self._T[s, a, :] \
                           + np.reshape(np.array(range(self._nS)) == s_prime, (1, 1, self._nS)) \
                           / np.sum(self._N[s, a, :])

        # b) the reward function
        # TODO for both the Pr and the R function, here is a version using a stochastic averaging and an exponentially
        #  weighted moving average method. The problem with the former is that in case of a change in the reward
        #  positions after N steps, it will take another N steps to unlearn the original pattern (SAME GOES FOR THE T
        #  FUNCTION ABOVE!!!). As for the latter solution, it necessitates yet another parameter (kappa) that we might
        #  want to get rid of, furthermore, the epistemic reward is now double-discounted by kappa (learning Pr and THEN
        #  Ht/Hr as part of the reward function). Is there a Bayesian solution to this?
        # self._Pr[s, a] = (1 - 1 / np.sum(self._N[s, a, :])) * self._Pr[s, a] + float(r > 0) / np.sum(self._N[s, a, :])
        self._Pr[s, a] = (1 - self._kappa) * self._Pr[s, a] + self._kappa * float(r > 0)  # EWMA

        hr = Hr - self.__entropy__(s, a, 'rew')  # negative change!
        ht = Ht - self.__entropy__(s, a, 'trans')
        rew = np.array([r, hr, ht])

        # self._R[s, a, :] = (1 - 1 / np.sum(self._N[s, a, :])) * self._R[s, a, :] + rew / np.sum(self._N[s, a, :])
        self._R[s, a, :] = (1 - self._kappa) * self._R[s, a, :] + self._kappa * rew  # EWMA

        # Taking care of the overestimation of uncertainty # TODO is this the right way of doing so
        if r > 0:
            self._C[s_prime, :, 1:] = 0
        return hr, ht

    def inference(self, s: int, a: int, s_prime: int, rew: np.ndarray, **kwargs) -> float:
        """
        Overwrites the parent class's method. Uses TDE to actualy update the Q values
        :param s: current state label
        :param a: taken action label
        :param s_prime: arriving state label
        :param rew: reward values [r, hr, ht]
        :param kwargs:
            update_buffer: should I store this item in the memory buffer [True, default] or not [False] -- during
                forward and backward replay or trsam we don't want to update the memory buffer
            virtual: is this a virtual step [True] or a real one [False, default] -- will decide if we'll learn epist
                values from it, and *in case we update the buffer*, do we add predecessors to it
        :return: the (combined) delta Q
        """
        # Let's see what situation we're in (real step or virtual, do we update the buffer or not)
        update_buffer = kwargs.get('update_buffer', True)
        virtual = kwargs.get('virtual', False)

        if not virtual:  # If s comes from the environment
            try:
                if s_prime not in self._states:
                    self.__extend_state_space__(s_prime)
            except AttributeError:
                pass
            s, s_prime = self.__translate_s__(s), self.__translate_s__(s_prime)
            # Extend the state space if needed

        # 1) learn Q val (each model will do its own thing)
        delta_C = 0
        if self._model_type == 'TD':
            delta_C = self.__temporal_difference_error__(s, a, s_prime, rew)
        elif self._model_type == 'VI':
            delta_C = self.__value_iteration__(s, a)
        elif self._model_type == "PI":
            delta_C = self.__policy_iteration__(s, a)
        # TODO: 2 questions. 1, I update the maxvals here, after having computed the delta C. Is that a problem?
        #  Should I do it before? Delta C consists of the prior and posterior estimates of the C values, so should I
        #  normalize them the same or differently? 2, if I keep the current maximum of the U values as maxval, then
        #  small uncertainties will slowly become more and more important as they scale up. However, if I keep the
        #  historical maximum, then, for the normal Q values, what happens if the reward decreases?
        self._maxval = np.maximum(self._maxval, np.amax(self.__amax_C__(), axis=0))
        self._maxval[self._maxval == 0] = 1  # It's only used for normalization

        # 4) Store if needed
        if update_buffer:
            self.__store_in_memory__(s, a, s_prime, rew, delta_C)
            if self._replay_type in ['priority', 'bidir'] and \
                    ((not virtual and self._add_predecessors in ['act', 'both']) or
                     (virtual and self._add_predecessors in ['rep', 'both'])):
                self.__find_predecessors__(s, abs(delta_C))
        # Saving the step in a table
        self.__save_step__(virtual, s=s, a=a, s_prime=s_prime, rew=rew, deltaC=delta_C)

        return delta_C

    def memory_replay(self, **kwargs) -> None:
        """
        Performs the memory replay. It either goes through the memory buffer, and performs a learning step on the stored
        events, in an order predefined by self._replay_type; or calls trajectory_sampling to generate virtual experience
        :param kwargs:
            s: the label of the starting state, in case the replay is trajectory sampling [int]
        :return:
        """

        # 0) This function uses stored memories to replay. For simulating experience, we need to call trsam
        update_buffer = False  # We will most likely not update the memory buffer during replay
        max_replay = self._max_replay
        if self._replay_type == 'trsam':
            s = kwargs.get('s', None)
            if s is None:
                raise ValueError('Trajectory sampling needs a starting state specified.')
            s = self.__translate_s__(s)  # As s comes from the outside
            self.__trajectory_sampling__(s)
            return
        elif self._replay_type in ['priority', 'bidir']:
            # TODO I think for bidir we should launch it not when we see a significant change, but precisely when we
            #  don't. When we do see a significant change, the prioritized sweeping will take care of it, leaving
            #  nothing for the trsam.
            # For bidir and priority it is essential to always update the buffer, as priorities change during replay
            update_buffer = True
            if self._replay_type == 'bidir' and max_replay is not None:
                # TODO should we keep this "half the replay is pr, the other half is trsam", or should we go pr until
                #  the buffer is empty?
                max_replay = floor(self._max_replay / 2)

        delta = 0  # the biggest change encountered over the course of the last swipe (fd, bd)
        it = 0  # how many iterations of memory replay have e performed so far
        buffer_idx = 0  # which memory are we replaying (for priority and bidir it's always 0)
        stop_loc = np.array([])  # for bidirectional, we might want to collect the states in which we need to stop

        # 1.1) Iterate while we can
        while self._memory_buff.size > 0 and (self._max_replay is None or it < max_replay):
            event = np.copy(self._memory_buff[buffer_idx])  # buffer_idx == 0 for priority and bidir
            # 1.2) We break out of the loop if nothing significant is produced. The criterion differs between the
            # different replay methods.
            if self._replay_type in ["priority", "bidir"]:
                # 1.2.a) If priority/bidir, we'll check significance: if we're still significant (delta > threshold) we
                # take the first element, and perform the update on it. If it's an empty event, that means the buffer
                # is empty, and the replay is over
                if np.all(event == 0):
                    break
                self.__pop_memory__()  # Remove the event that we'll replay
            elif self._replay_type in ["forward", "backward"]:
                # 1.2.b) If forward or backward, we need to loop through the entire memory buffer (without changing it)
                # before we can tell whether the updates are significant or not. If we never encountered a single
                # above-threshold delta over the whole array, we stop. Otherwise, we just restart.
                if np.all(event == 0) or (it > 0 and buffer_idx == 0):
                    if abs(delta) < self._replay_thresh:
                        break
                    delta = 0
                    if buffer_idx != 0:
                        # In case the memory buffer is not full, so we iterated onto an empty element, start over
                        buffer_idx = 0
                        event = self._memory_buff[buffer_idx]

            # 1.3) We perform the replay, and we learn from it. The learning takes place on virtual experience,
            # but whether we update the buffer or not depends on whether we use priorities or not
            # Based on whether we consider events as recollections of s, or (s, a), the rest will be filled by the
            # model
            s = int(event[0])
            if self._event_content in ['sa', 'sas', 'sasr']:
                a = int(event[1])
            else:
                a_poss = self.__find_good_actions__(s)
                # If we are replaying a state without having explored the actions leading out of it, then we'll find no
                # action to choose. In this case we need to move to the next element in the memory
                # This scenario is impossible if we allow bumping into walls
                if len(a_poss) == 0:
                    if self._replay_type in ['forward', 'backward']:
                        it += 1
                    continue
                a = self.choose_action(s, a_poss, virtual=True)
            if self._event_content in ['sas', 'sasr']:
                s_prime = int(event[2])
            else:
                s_prime = np.random.choice(list(range(self._nS)), p=self._T[s, a, :])
            if self._event_content == 'sasr':
                rew = event[3:6]
            else:
                rew = self._R[int(event[0]), int(event[1]), :].squeeze()
            delta_curr = self.inference(s=s, a=a, s_prime=s_prime, rew=rew,
                                        virtual=True, update_buffer=update_buffer)

            # 1.4) Conclude by some final step
            if self._replay_type in ["priority", "bidir"]:
                if self._replay_type == 'bidir':
                    # We store the replayed state as a bidir stopping criterion
                    # (we only store s, not s_prime, as technically a decision from s_prime has not yet been replayed)
                    stop_loc = np.append(stop_loc, np.array([int(event[0])]), axis=0)
            elif self._replay_type in ["forward", "backward"]:
                if abs(delta_curr) > abs(delta):
                    delta = delta_curr
                buffer_idx += 1
                if buffer_idx >= self._memory_buff.shape[0]:
                    # No need to take care of arriving at an empty row, that's handled in 1.2.b)
                    buffer_idx = 0
            it += 1

        # 2) If bidir, let's run some simulations using the remaining steps
        if self._replay_type == 'bidir':
            s = kwargs.get('s', None)
            if s is None:
                raise ValueError('Bidirectional search needs a starting state specified.')
            s = self.__translate_s__(s)  # As s comes from the outside
            steps = self._max_replay
            if steps is not None:
                steps -= it
            self.__trajectory_sampling__(s, stop_loc=np.unique(stop_loc), steps=steps)
        return

    # def update_agent(self, mdp: Env) -> None:
    #     """
    #     Updates the agent's expectations about the reward, should a change take place in the environment.
    #     :param mdp: the changed environment
    #     :return:
    #     """
    #     self._maxrew = mdp.max_rew()  # max value of the achievable reward
    #     self._maxval[0] = self._maxrew / (1 - self._gamma)

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
            self._events.to_csv(f'{path}agent{label}.csv', sep=',', index=False, encoding='utf-8')
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
