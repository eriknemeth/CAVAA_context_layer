import numpy as np

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
        return

    # Public methods
    def action_selection(self, state: np.ndarray, poss_moves: np.ndarray) -> Tuple[int, bool]:
        """
        Performs the action selection comparing the output of the 2 agents. The one that shows a higher Q value wins.
        Args:
            state: The state I am currently in (coordinates)
            poss_moves: What are the currently available moves (ints)

        Returns:
            'straight', 'left', or 'right' as the chosen action; and
            True if we used the MF agent, False if we used the MB agent
        """
        # 1) MF action selection
        action_SEC, Q_SEC = self._SEC.choose_action(state)

        # 2) MB action selection
        action_SORB, Q_SORB = self._SORB.choose_action(state, poss_moves)

        # 3) Compare results
        if Q_SEC >= Q_SORB:
            return action_SEC, True
        else:
            return action_SORB, False

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
        self._SORB.dump_agent(path=kwargs.get('path', None), label=kwargs.get('label', None))
        return
