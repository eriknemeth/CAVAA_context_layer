import numpy as np
from typing import Tuple

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

        # Low-pass filetered action probabilities and computation time values for both models
        self.LPF_action_prob_SEC = 0.
        self.LPF_action_prob_SORB = 0.
        self.LPF_comp_time_SEC = 0.
        self.LPF_comp_time_SORB = 0.

        # Initialize histories
        self.action_prob_history_SEC = []
        self.action_prob_history_SORB = []
        self.computation_time_history_SEC = []
        self.computation_time_history_SORB = []

        self.kappa = 7  # Parameter of equation from Dromnelle et al. 2023, set at 7 based on paper
        self.tau = 0.67  # Time constant for the low-pass filter

        return

    def update_and_filter(self, history: list, new_value: float) -> float:
        """Update history with new value and return low-pass filtered value."""
        history.append(new_value)
        filtered_value = np.mean([v * np.exp(-i / self.tau) for i, v in enumerate(reversed(history))])
        return filtered_value / np.sum(np.exp(-np.arange(len(history)) / self.tau))

    def update_action_probs_and_comp_time(self, expert, action_probs, comp_time):
        """Update action probabilities and computation time history for an expert."""
        if expert == 'SEC':
            filtered_prob_SEC = self.update_and_filter(self.action_prob_history_SEC, action_probs)
            filtered_time_SEC = self.update_and_filter(self.computation_time_history_SEC, comp_time)
            return filtered_prob_SEC, filtered_time_SEC
        elif expert == 'SORB':
            filtered_prob_SORB = self.update_and_filter(self.action_prob_history_SORB, action_probs)
            filtered_time_SORB = self.update_and_filter(self.computation_time_history_SORB, comp_time)
            return filtered_prob_SORB, filtered_time_SORB
        else:
            raise ValueError("Invalid expert name. Use 'SEC' or 'SORB'.")


    def compute_entropy(self, action_probs):
        """
        Equation 5 from Dromnelle et al. 2023.
        Compute the entropy of the action probability distribution.
        """
        return -np.sum(action_probs * np.log2(action_probs + 1e-9))  # Adding epsilon to avoid log(0)


    def compute_expert_value(self, H_MF, H_MB, CT_MF, CT_MB):
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

        entropy_SEC = self.compute_entropy(self.LPF_action_prob_SEC)
        entropy_SORB = self.compute_entropy(self.LPF_action_prob_SORB)

        Q_SEC, Q_SORB = self.compute_expert_value(entropy_SEC, entropy_SORB, self.LPF_comp_time_SEC, self.LPF_comp_time_SORB)

        # Selecting action based on the higher expert-value
        if Q_SEC >= Q_SORB:
            action_SEC, action_prob_SEC, comp_time_SEC, _ = self._SEC.choose_action(state)
            self.LPF_action_prob_SEC, self.LPF_comp_time_SEC = self.update_action_probs_and_comp_time('SEC', action_prob_SEC, comp_time_SEC)
            return action_SEC, 'SEC'
        else:
            action_SORB, action_prob_SORB, comp_time_SORB, _ = self._SORB.choose_action(state, poss_moves)
            self.LPF_action_prob_SORB, self.LPF_comp_time_SORB = self.update_action_probs_and_comp_time('SORB', action_prob_SORB, comp_time_SORB)
            return action_SORB, 'SORB'

    def learning(self, state: np.ndarray, action: int, new_state: np.ndarray, reward: float) -> bool:
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
        replayed = self._SORB.inference(state, action, new_state, np.array([reward, hr, ht]))

        # 3) Return
        return replayed

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
