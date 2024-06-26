from SORB_agent import *
from tqdm import tqdm
# from meta_agent import *
from meta_agent_new import *
import re


def experiment_plotter(path: str, env_file: str, SORB_file: str, SECS_file: str):
    """
    Plots the data gathered from a specific experiment
    :param path: path to the experimental data
    :param env_file: name of the environment file [.txt]
    :param SORB_file: name of the agent events file [.csv]
    :param SECS_file: name of the SECS event file [.csv]
    :return:
    """
    dTm = PlotterEnv(env_file, path=path)
    dTm.load_events(SORB_file, 'SORB', path=path)
    dTm.load_events(SECS_file, 'SEC', path=path,
                    memories_name='activated_memories.csv', states_name='retrieved_states.csv')
    dTm.plot_events()


def spatial_navigation() -> None:
    """
    This function realizes the simple navigation by creating a meta agent and passing the messages from the environment
    to it
    Returns:
    """
    ####################################################################################################################
    # Parameters of the experiment
    # TODO rewise these parameters, potentially adaptively, and replace the test environment
    env_params = dict()
    steps = 1000
    env_params['actions'] = [0, 1, 2, 3]

    # About saving
    env_params['save_data'] = True  # -------------------- Should I even save at all
    if env_params['save_data']:
        env_params['save_path'] = './savedata'  # -------- Where should I save
        env_params['save_tag'] = None  # ----------------- What tag should I put on saved data

    # About the maze
    env_params['teleport'] = True  # ---------------------- The agent teleports after getting the reward [True] or not
    env_params['use_epochs'] = False  # ------------------- If [True] we use epochs, if [False] we use steps
    env_params['num_runs'] = steps  # --------------------- How many steps do we model
    env_params['rew_change'] = None  # -------------------- When do we change the reward location (if we do)
    env_params['rew_loc'] = np.array([[4, 0], [1, 8]])  # - What is (are) the rewarded state(s)
    env_params['rew_mean'] = np.array([[5], [20]])  # ------- What is (are) the value(s) of the reward(s)
    env_params['rew_std'] = np.array([[1], [1]])  # ------ What is (area) the probability/ies of the reward(s)
    env_params['new_rew_loc'] = None  # ------------------- What is (are) the rewarded state(s)
    env_params['new_rew_val'] = None  # ------------------- What is (are) the value(s) of the reward(s)
    env_params['new_rew_prob'] = None  # ------------------ What is (area) the probability/ies of the reward(s)
    env_params['wall_loc'] = None  # ---------------------- The wall is between what states (before the change)
    env_params['wall_change'] = None  # steps * 0.2  # ------------ When do we add a wall (if we do)
    env_params['new_wall_loc'] = None  # np.array([[[0, 6], [0, 7]],
                                           # [[5, 6], [5, 7]]])  # The wall is between what states (after the change)
    env_params['start_pos'] = np.array([4, 3])  # --------- What state do we start from
    env_params['forbidden_walls'] = False  # -------------- Is it forbidden to bump into walls?
    env_params['restricted_dT'] = False  # ---------------- Is the movement restricted to unidirectional?
    env_params['slip_prob'] = 0  # ------------------------ The probability of slipping after a step
    ####################################################################################################################

    ####################################################################################################################
    # Let's start by defining the SEC agent's parameters
    SEC_params = dict()
    SEC_params['action_space'] = len(env_params['actions'])
    #
    # # About the agent
    SEC_params['emb'] = 2
    SEC_params['stm'] = 15  # default: 50, test: 10
    SEC_params['ltm'] = 50  # default: 50K, test: 10K

    SEC_params['sequential_bias'] = True
    SEC_params[
        'value_function'] = 'noGi'  # value_functions = ['default', 'noGi', 'noDist', 'noRR', 'soloGi', 'soloDist', 'soloRR']
    SEC_params[
        'forget'] = 'FIFO'  # types = ['FIFO-SING', 'FIFO-PROP', 'RWD-SING', 'RWD-PROP', 'LRU-SING', 'LRU-PROP', 'LRU-PROB'] - default: FIFO-SING

    SEC_params['coll_threshold_act'] = 0.98  # default: 0.98
    SEC_params['coll_threshold_proportion'] = 0.995  # default animalai: 0.995

    SEC_params['alpha_trigger'] = 0.05  # default = 0.05
    SEC_params['tau_decay'] = 0.9  # default animalai: 0.9
    SEC_params['load_ltm'] = False

    SEC_params['exploration_mode'] = 'default'  # exploration_mode = ['default', 'fixed', 'epsilon', 'epsilon_decay']
    SEC_params[
        'exploration_steps'] = 150  # THE UNITS ARE NUMBER OF AGENT STEPS! - NatureDQN: 50k STEPS / 50 EPISODES ANIMALAI
    SEC_params['epsilon'] = 0.05  # DEFAULT FOR MFEC IN ATARI: 0.1

    SEC_params['selection_mode'] = 'default'  # selection_mode = ['default', 'argmax']
    SEC_params['memory_threshold'] = 2  # originally = 4
    ####################################################################################################################

    ####################################################################################################################
    # Then let's define the MB agent's parameters:
    SORB_params = dict()
    SORB_params['actions'] = env_params['actions']  # ------- What are the possible actions
    SORB_params['curr_state'] = env_params['start_pos']  # -- Where the agent is

    # About saving
    SORB_params['save_data'] = True  # ---------------------- Should save the steps taken into a csv?
    if SORB_params['save_data']:
        SORB_params['save_path'] = './savedata'  # ---------- Where should I save
        SORB_params['save_tag'] = None  # ------------------- What tag should I put on saved data

    # About the agent
    SORB_params['teleport'] = env_params['teleport']  # ----- Do we teleport after each step
    SORB_params['act_num'] = len(SORB_params['actions'])  # - Size of action space # TODO make it adaptive
    SORB_params['model_type'] = 'VI'  # --------------------- 'VI' value iteration or 'TD' temporal difference
    if SORB_params['model_type'] == 'TD':
        SORB_params['alpha'] = 0.8  # ----------------------- from Massi et al. (2022) MF-priority
    SORB_params['nV'] = 4  # -------------------------------- The model is updated based on the last nV visits
    SORB_params['gamma'] = 0.9  # --------------------------- Discounting factor
    SORB_params['decision_rule'] = 'softmax'  # ------------- Could be 'max', 'softmax', 'epsilon'
    if SORB_params['decision_rule'] == 'epsilon':
        SORB_params['epsilon'] = 0.1  # --------------------- Epsilon of the epsilon-greedy
    SORB_params['beta'] = 100  # ----------------------------- Beta for softmax
    SORB_params['replay_type'] = 'priority'  # -------------- 'priority', 'trsam', 'bidir', 'backwards', 'forward'
    SORB_params['replay_every_step'] = True  # -------------- Replay after each step
    if SORB_params['replay_type'] in ['trsam', 'bidir']:
        SORB_params['replay_every_step'] = False  # --------- Replay after each epoch
    if SORB_params['replay_type'] in ['priority', 'bidir']:
        SORB_params['event_handle'] = 'sa'  # --------------- What is each new memory compared to [s, sa, sas]
    SORB_params['event_content'] = 'sas'  # ----------------- What is not estimated from model [s, sa, sas, sasr]
    SORB_params['replay_thresh'] = 0.1  # ------------------ Smallest surprise necessary to initiate replay
    SORB_params['max_replay'] = 50  # ----------------------- Max replay steps per replay event
    SORB_params['add_predecessors'] = 'both'  # ------------- When should I add predecessors (None, act, rep or both)
    SORB_params['forbidden_walls'] = False  # --------------- If we replay (simulate), is bumping into a wall forbidden?
    SORB_params['dec_weights'] = np.array([0.4, 0.4, 0.2])  # The weights used for decision-making [Q, Ur, Ut] float
    SORB_params['rep_weigths'] = np.array([0.4, 0.4, 0.2])  # The weights used for replay [Q, Ur, Ut] float
    ####################################################################################################################

    ####################################################################################################################
    # Initializing the environment and the agent
    env = DTMaze(**env_params)

    # 1) Get the first state
    env.place_reward(env_params['rew_loc'],
                     env_params['rew_mean'],
                     env_params['rew_std'])
    env.place_wall(env_params['wall_loc'])

    META = metaAgent(SEC_params=SEC_params, SORB_params=SORB_params)
    ####################################################################################################################

    ####################################################################################################################
    # Initiating saving for visual purposes
    if env_params['save_data']:
        env.toggle_save()
    if SORB_params['save_data']:
        META.toggle_save()

    # Running the experiment
    state = env.place_agent(env_params['start_pos'])
    step = 0
    pbar = tqdm(total=env_params['num_runs'])
    while step < env_params['num_runs']:
        # 2) Choose an action
        poss_moves = env.possible_moves(state)
        action, SEC_winner = META.action_selection(state, poss_moves)

        # 3) Commit to action
        new_state, reward, done = env.step(state, action, SEC_winner)

        # 4) Learn
        replayed = META.learning(state, action, new_state, reward)

        # 5) If the agent reached a reward, send it back to the starting position
        if done:
            if env_params['use_epochs']:
                step += 1
                pbar.update(1)
            state = env.place_agent(env_params['start_pos'])
            META.reset(state)
        else:
            state = new_state

        if not env_params['use_epochs']:
            step += 1
            pbar.update(1)

        # 6) Change reward location if must
        if step == env_params['rew_change']:
            env.reset_reward()
            env.place_reward(env_params['new_rew_loc'],
                             env_params['new_rew_mean'],
                             env_params['new_rew_std'])

        # 7) Change wall location if must
        if step == env_params['wall_change']:
            env.reset_wall()
            env.place_wall(env_params['new_wall_loc'])

    # 7) Save for visualization
    if env_params['save_data']:
        env.dump_env(path=env_params['save_path'], label=env_params['save_tag'])
    if SORB_params['save_data']:
        META.dump_agent(path=SORB_params['save_path'], label=SORB_params['save_tag'])
    ####################################################################################################################
