import numpy as np
from typing import Tuple
import time

from SORB_agent import *
from SECS_agent import *


class metaAgent():
    """
    This agent will be capable of making decisions based on the combined outputs of the MB and the sequential MF agents
    """

    def __init__(self, SEC_params: dict, SORB_params: dict) -> None:
        self._replay_thresh = SORB_params['replay_thresh']
        self._plan = SORB_params['replay_type'] in ['trsam', 'bidir']
        self._SEC = SECagent(**SEC_params)
        self._SORB = RLagent(**SORB_params)

        # For the internal representation
        self._states = np.array(
            [SORB_params['curr_state']])  # an array of sates. The idx of each state is my state label
        state_idx = self.__translate_s__(SORB_params['curr_state'])
        self._beta = SORB_params['beta']

        # Low-pass filetered action probabilities and computation time values for both models
        self._LPF_action_prob_SEC = {state_idx: 0.}
        self._LPF_action_prob_SORB = {state_idx: 0.}
        self._LPF_comp_time_SEC = {state_idx: 0.}
        self._LPF_comp_time_SORB = {state_idx: 0.}

        # Initialize histories
        self._action_prob_history_SEC = {state_idx: []}
        self._action_prob_history_SORB = {state_idx: []}
        self._computation_time_history_SEC = {state_idx: []}
        self._computation_time_history_SORB = {state_idx: []}

        self.kappa = 7  # Parameter of equation from Dromnelle et al. 2023, set at 7 based on paper
        self.tau = 0.67  # Time constant for the low-pass filter

        return

    def __translate_s__(self, s: np.ndarray) -> Union[int, None]:
        """
        Translates s into labels recognized by the meta agent. This is important for the arrays
        :param s: the state as detected from the environment
        :return: the state label recognized by the agent
        """
        if s is None:
            return s
        return np.where((self._states == s).all(axis=1))[0][0]

    def __extend_state_space__(self, s_prime: np.ndarray) -> None:
        """
        Upon encountering a never-before seen state, this function extends the state-space
        :param s_prime: the label of the new state
        :return:
        """
        # First the states
        self._states = np.append(self._states, np.array([s_prime]), axis=0)

        # Then the rest
        s_prime_idx = self.__translate_s__(s_prime)
        # Low-pass filetered action probabilities and computation time values for both models
        self._LPF_action_prob_SEC[s_prime_idx] = 0.
        self._LPF_action_prob_SORB[s_prime_idx] = 0.
        self._LPF_comp_time_SEC[s_prime_idx] = 0.
        self._LPF_comp_time_SORB[s_prime_idx] = 0.

        # Initialize histories
        self._action_prob_history_SEC[s_prime_idx] = []
        self._action_prob_history_SORB[s_prime_idx] = []
        self._computation_time_history_SEC[s_prime_idx] = []
        self._computation_time_history_SORB[s_prime_idx] = []

    def __update_and_filter__(self, history: list, new_value: float) -> float:
        """Update history with new value and return low-pass filtered value."""
        history.append(new_value)
        filtered_value = np.mean([v * np.exp(-i / self.tau) for i, v in enumerate(reversed(history))])
        return filtered_value / np.sum(np.exp(-np.arange(len(history)) / self.tau))

    def __update_action_probs_and_comp_time__(self, expert, state_idx, action_probs, comp_time):
        """Update action probabilities and computation time history for an expert."""
        if expert == 'SEC':
            filtered_prob_SEC = self.__update_and_filter__(self._action_prob_history_SEC[state_idx], action_probs)
            filtered_time_SEC = self.__update_and_filter__(self._computation_time_history_SEC[state_idx], comp_time)
            return filtered_prob_SEC, filtered_time_SEC
        elif expert == 'SORB':
            filtered_prob_SORB = self.__update_and_filter__(self._action_prob_history_SORB[state_idx], action_probs)
            filtered_time_SORB = self.__update_and_filter__(self._computation_time_history_SORB[state_idx], comp_time)
            return filtered_prob_SORB, filtered_time_SORB
        else:
            raise ValueError("Invalid expert name. Use 'SEC' or 'SORB'.")

    def __compute_entropy__(self, action_probs):
        """
        Equation 5 from Dromnelle et al. 2023.
        Compute the entropy of the action probability distribution.
        """
        return -np.sum(action_probs * np.log2(action_probs + 1e-9))  # Adding epsilon to avoid log(0)

    def __compute_expert_value__(self, H_MF, H_MB, CT_MF, CT_MB):
        """
        Equation 6 from Dromnelle et al. 2023.
        Compute the expert-value for the MF and MB strategies."""
        Q_MF = -(H_MF + np.exp(-self.kappa * H_MF) * CT_MF)
        Q_MB = -(H_MB + np.exp(-self.kappa * H_MF) * CT_MB)
        return Q_MF, Q_MB

    def action_selection(self, state: np.ndarray, poss_moves: np.ndarray) -> Tuple[int, str]:
        """
        Performs the action selection comparing the output of the 2 agents. The one that shows a higher Q value wins.
        Args:
            state: The state I am currently in (coordinates)
            poss_moves: What are the currently available moves (ints)

        Returns:
            'straight', 'left', or 'right' as the chosen action; and
            True if we used the MF agent, False if we used the MB agent
        """
        state_idx = self.__translate_s__(state)
        entropy_SEC = self.__compute_entropy__(self._LPF_action_prob_SEC[state_idx])
        entropy_SORB = self.__compute_entropy__(self._LPF_action_prob_SORB[state_idx])

        Q_SEC, Q_SORB = self.__compute_expert_value__(entropy_SEC, entropy_SORB, self._LPF_comp_time_SEC[state_idx],
                                                      self._LPF_comp_time_SORB[state_idx])

        # Selecting action based on the higher expert-value
        p_agent = np.exp(self._beta * np.array([Q_SEC, Q_SORB])) / np.sum(np.exp(self._beta * np.array([Q_SEC, Q_SORB])))
        agent_idx = np.random.choice(np.array([0, 1]), p=p_agent)
        if agent_idx == 0:
            start_SEC = time.time()
            action_SEC, action_prob_SEC = self._SEC.choose_action(state)
            end_SEC = time.time()
            self._LPF_action_prob_SEC[state_idx], self._LPF_comp_time_SEC[state_idx] = \
                self.__update_action_probs_and_comp_time__('SEC', state_idx, action_prob_SEC, end_SEC - start_SEC)
            return action_SEC, 'SEC'
        else:
            start_SORB = time.time()
            action_SORB, action_prob_SORB = self._SORB.choose_action(state, poss_moves)
            end_SORB = time.time()
            self._LPF_action_prob_SORB[state_idx], self._LPF_comp_time_SORB[state_idx] = \
                self.__update_action_probs_and_comp_time__('SORB', state_idx, action_prob_SORB, end_SORB - start_SORB)
            return action_SORB, 'SORB'

    def learning(self, state: np.ndarray, action: int, new_state: np.ndarray, reward: float):
        """
        The learning performed by both sub-agents
        Args:
            state: the state that the agent started from
            action: the action the agent took
            new_state: the state the agent arrived in
            reward: the reward the agent gained

        Returns:
            True if the agent performed replay, False if not
        """
        # 1) Teach the MF agent
        # Update SEC's STM based on previous (state,action) couplet
        self._SEC.update_STM(couplet=[state, action])  # TODO changed it to couplet from sa_couplet
        self._SEC.update_sequential_bias()
        self._SEC.update_LTM(reward)

        # 2) Teach the MB agents
        hr, ht = self._SORB.model_learning(state, action, new_state, reward)
        self._SORB.inference(state, action, new_state, np.array([reward, hr, ht]))

        # 3) Store and return
        if not any(np.equal(self._states, new_state).all(1)):
            self.__extend_state_space__(new_state)
        return

    def reset(self, state) -> None:
        """
        Reset the short-term memory of the MF agent
        """
        self._SEC.reset_memory()
        if self._plan:
            self._SORB.memory_replay(s=state)

    def toggle_save(self):
        """
        Toggles saving for future visualization
        Args:

        Returns:

        """
        # self._SEC.toggle_save()
        self._SORB.toggle_save()
        return

    def dump_agent(self, **kwargs):
        """
        Saving the MB agent
        Args:
            **kwargs:
                path: Path to save. If undefined we save to the working folder
                tag: the tag to add to the file [optional str]
        Returns:

        """
        self._SEC.dump_agent(path=kwargs.get('path', None), label=kwargs.get('label', None))
        self._SORB.dump_agent(path=kwargs.get('path', None), label=kwargs.get('label', None))
        return
