from SORB_agent import *
from tqdm import tqdm
import re


def cumulative_plotter(path: str, batches: list, env_file: str, label: str, fig_shape: list, norm_rep: bool) -> None:
    pass
    # TODO plotterEnv
    # dTm = PlotterEnv(env_file, False, with_replay=False, win_size=100, path=f'{path}{batches[0]}/', norm_rep=norm_rep)
    # regex = re.compile('agent.*csv')
    # for batch in batches:
    #     curr_path = f'{path}{batch}/'
    #     print(f'Batch {batch}...')
    #     for root, subdirs, files in os.walk(curr_path):
    #         for file in tqdm(files):
    #             if regex.match(file):
    #                 dTm.load_events(file, batch, path=curr_path)
    # print('Plotting...')
    # dTm.plot_reward_rates(batches, save_img=True, path='./img', label=label)
    # plt.pause(5)
    # plt.close()
    # dTm.plot_replay('loc', batches, fig_shape, save_img=True, path='./img', label=label)
    # plt.pause(5)
    # plt.close()
    # dTm.plot_replay('content', batches, fig_shape, save_img=True, path='./img', label=label)
    # plt.pause(5)
    # plt.close()


def matrix_plotter(path: str, axes: list, axes_to_plot: list, batches: list, env_file: str, **kwargs) -> None:
    pass
    # TODO plotterEnv
    # win_begin = kwargs.get('win_begin', 0)
    # win_end = kwargs.get('win_end', None)
    # dTm = PlotterEnv(env_file, False, with_replay=False, win_size=100, path=f'{path}{batches[0]}/', params=axes,
    #                  win_begin=win_begin, win_end=win_end)
    # regex = re.compile('agent.*csv')
    # for batch in batches:
    #     curr_path = f'{path}{batch}/'
    #     print(f'Batch {batch}...')
    #     for root, subdirs, files in os.walk(curr_path):
    #         for file in tqdm(files):
    #             if regex.match(file):
    #                 dTm.load_events(file, batch, path=curr_path)
    # print('Plotting...')
    # label = kwargs.get('label', '')
    # methods = kwargs.get('methods', 'max')
    # for ax in axes_to_plot:
    #     for meth in methods:
    #         dTm.plot_cumul_rew_matrix(ax, save_img=True, path='./img', label=label, method=meth)


def experiment_plotter(path: str, env_file: str, agent_file: str, **kwargs):
    """
    Plots the data gathered from a specific experiment
    :param path: path to the experimental data
    :param env_file: name of the environment file [.txt]
    :param agent_file: name of the agent events file [.csv]
    :param kwargs:
        weights: weights to combine the Q, Ur and Ut values into C-values [np.ndarray]
    :return:
    """
    weights = kwargs.get('weights', np.array([1, 0, 0]))
    gamma = 0.9
    dTm = PlotterEnv(env_file, path=path)
    dTm.load_events(agent_file, 'MB', path=path)
    dTm.plot_events(weights=weights, gamma=gamma)


def complex_experiment(save_path: str, tag: str, dec_weights, **kwargs) -> None:
    """
    Based on Massi et al.
    double reward maze, free motion
    :return:
    """

    # The parameters I have tested
    rep_weights = kwargs.get('rep_weights', dec_weights)
    replay_threshold = kwargs.get('replay_threshold', 0.02)
    decision_rule = kwargs.get('decision_rule', 'softmax')
    add_predecessors = kwargs.get('add_predecessors', 'both')
    rew_change = kwargs.get('rew_change', 1 / 3)
    new_rew_loc = kwargs.get('new_rew_loc', np.array([9]))
    new_rew_val = kwargs.get('new_rew_val', np.array([5]))

    model_type = kwargs.get('model_type', 'MB')
    max_replay = kwargs.get('max_replay', 50)
    known_env = kwargs.get('known_env', True)
    replay_type = kwargs.get('replay_type', 'priority')

    # Creating a model
    params = dict()

    # About saving
    params['save_data'] = True  # --------------------------- Should save the steps taken into a csv?
    if params['save_data']:
        params['save_path'] = save_path  # --------------- Where should I save
        params['save_tag'] = tag  # ------------------------- What tag should I put on saved data

    # About the maze
    params['use_epochs'] = False  # ------------------------- If [True] we use epochs, if [False] we use steps
    params['num_runs'] = 1000  # ---------------------------- How many steps/epochs do we model
    params['rew_change'] = None  # params['num_runs'] * rew_change  # - When do we change the reward location (if we do)
    params['rew_loc'] = np.array([11, 20])  # --------------- What is (are) the rewarded state(s)
    params['rew_val'] = np.array([5, 1])  # ----------------- What is (are) the value(s) of the reward(s)
    params['rew_prob'] = np.array([1, 1])  # ---------------- What is (area) the probability/ies of the reward(s)
    params['new_rew_loc'] = new_rew_loc  # ------------------ What is (are) the rewarded state(s)
    params['new_rew_val'] = new_rew_val  # ------------------ What is (are) the value(s) of the reward(s)
    params['new_rew_prob'] = np.ones(new_rew_loc.shape)  # -- What is (area) the probability/ies of the reward(s)
    params['start_pos'] = 21  # ----------------------------- What state do we start from
    params['env_forbidden_walls'] = True  # ----------------- Is it forbidden to bump into walls?
    params['restricted_dT'] = False  # ---------------------- Is the movement restricted to unidirectional?
    params['slip_prob'] = 0  # ------------------------------ The probability of slipping after a step

    # About the agent
    params[
        'known_env'] = known_env  # --------------------------- Is the state-space known in advance [True] or not [False]
    params['model'] = model_type  # ------------------------------- Model free or model based

    if params['model'] == 'MF':
        params['model_type'] = 'TD'  # ---------------------- TD (for MF) or VI/PI (for MB)
        params['alpha'] = 0.8  # ---------------------------- from Massi et al. (2022) MF-priority
    elif params['model'] == 'MB':
        params['model_type'] = 'VI'  # ---------------------- TD (for MF) or VI/PI (for MB)
        params['pre_training'] = None  # -------------------- how many steps do we pre-train
    params['gamma'] = 0.9  # -------------------------------- Discounting factor
    params['kappa'] = 1  # -------------------------------- Learning rate for the model (needed for the epist rewards)
    params['decision_rule'] = decision_rule  # -------------- Greedy decisions (could be 'max', 'softmax', 'epsilon')
    if params['decision_rule'] == 'epsilon':
        params['epsilon'] = 0.1  # -------------------------- Epsilon of the epsilon-greedy
    elif params['decision_rule'] == 'softmax':
        params['beta'] = 10  # ------------------------------ Beta for softmax from Massi et al. (2022) MF-priority
    params['replay_type'] = replay_type  # ------------------ 'priority', 'trsam', 'bidir', 'backwards', 'forward'
    params['replay_every_step'] = True
    if params['replay_type'] in ['trsam', 'bidir']:
        params['replay_every_step'] = False
    if params['replay_type'] in ['priority', 'bidir']:
        params['event_handle'] = 'sa'  # -------------------------- What is each new memory compared to [s, sa, sas]
    params['event_content'] = 'sas'  # ------------------------ What is not estimated from model [s, sa, sas, sasr]
    params['replay_thresh'] = replay_threshold  # ----------- Smallest surprise necessary to initiate replay
    params['max_replay'] = max_replay  # ---------------------------- Max replay steps per replay event
    params['add_predecessors'] = add_predecessors  # -------- When do I add state predecessors (None, act, rep or both)
    params[
        'replay_forbidden_walls'] = True  # ----------------- If we replay (simulate), is bumping into a wall forbidden?
    params['dec_weights'] = dec_weights  # ------------------ The weights used for decision-making [Q, Ur, Ut] float
    params['rep_weigths'] = rep_weights  # ------------------ The weights used for replay [Q, Ur, Ut] float

    run_dT(**params)


def run_dT(rew_loc: np.ndarray, start_pos: int, num_runs: int,
           model: str, model_type: str, gamma: float, kappa: float, decision_rule: str,
           **kwargs):
    """
    Runs the double-T-maze experiment
    :param rew_loc: where the OG reward will be placed
    :param start_pos: where the agent starts from
    :param num_runs: how many steps/epochs are we modelling
    :param model: 'MF' or 'MB'
    :param model_type: 'TD', 'VI' or 'PI'
    :param gamma: discount factor
    :param kappa: weighing parameter for the internal model (also used for the epist rew in both agnets!!!) [float]
    :param decision_rule: 'max', 'epsilon' or 'softmax'
    :param kwargs:
        Environment-related variables:
            use_epochs: if [True] we use epochs instead of steps (an epoch ends when a reward is received)
            env_forbidden_walls: can the agent choose to bump into a wall [bool]
            restricted_dT: is the movement unidirectional or not [bool]
            slip_prob: probability of slipping while moving [float]
            rew_val: value of reward [float array]
            rew_prob: proba of reward [float array]
            rew_change: what step will we change the reward location (if we do) [int]
                new_rew_loc: where the reward will be placed after the location change [int array]
                new_rew_val: value of reward [float array]
                new_rew_prob: proba of reward [float array]
        Agent-related variables:
            known_env: is the state-space previously known [True] or not [False, default]
            based on 'model':
                alpha: learning parameter for the MF agent [float]

                pre_training: number of unrewarded pre-training steps (for tuning the model of MB agent) [int]
            based on 'decision_rule':
                epsilon: exploration constant of epsilon greedy agent [float]
                beta: exploitation constant of softmax agent [float]
            replay_type: 'forward', 'backward', 'priority', 'trsam', 'bidir' or None:
                replay_every_step: do I replay after every step [True, default] or only after receiving a reward [False]
                event_handle: what should we compare a new event to when trying to estimate if we need to
                    overwrite an old memory or not: states ['s'], state-action ['sa'] or
                    state-action-new state ['sas']. Only needed if replay_type is "priority" or "bidir"
                event_content: what should we replay, states ['s'], state-action ['sa'],
                    state-action-new state ['sas'], or state-action-new state-reward ['sasr', default].
                replay_thresh: replay threshold [float]
                max_replay: max number of replay steps [int]
                add_predecessors: for priority and bidir, when do I add predecessors to the buffer ['act', 'rep',
                    'both', None]
                replay_forbidden_walls: is choosing a wall forbidden for replay [True] or not [False]
            dec_weight: weight of the different quality functions contributing to decisions [Q, Ur, Ut], float array
            rep_weight: weight of the different quality functions contributing to replay [Q, Ur, Ut], float array
        Storing-related variables:
            save_data: Should we save the data generated [True] or not [False, default]
            save_path: Where should we save [str] (default: current folder)
            save_tag: What tag should I add to the end of the filename [str, default: None]
    :return:
    """
    # Arguments for the environment
    use_epochs = kwargs.get('use_epochs', False)
    env_forbidden_walls = kwargs.get('env_forbidden_walls', False)
    restricted_dT = kwargs.get('restricted_dT', False)
    slip_prob = kwargs.get('slip_prob', 0)
    rew_val = kwargs.get('rew_val', np.ones(rew_loc.shape))
    rew_prob = kwargs.get('rew_prob', np.ones(rew_loc.shape))
    rew_change = kwargs.get('rew_change', None)
    new_rew_loc = kwargs.get('new_rew_loc', rew_loc)
    new_rew_val = kwargs.get('new_rew_val', rew_val)
    new_rew_prob = kwargs.get('new_rew_prob', rew_prob)

    # Arguments for the model
    known_env = kwargs.get('known_env', False)
    alpha, epsilon, beta, pre_training = None, None, None, None
    if model == 'MF':
        alpha = kwargs.get('alpha', None)
    elif model == 'MB':
        pre_training = kwargs.get('pre_training', None)
    if decision_rule == 'epsilon':
        epsilon = kwargs.get('epsilon', None)
    elif decision_rule == 'softmax':
        beta = kwargs.get('beta', None)
    replay_type = kwargs.get('replay_type', None)
    event_handle = kwargs.get('event_handle', None)
    event_content = kwargs.get('event_content', 'sasr')
    replay_every_step = kwargs.get('replay_every_step', True)
    replay_thresh = kwargs.get('replay_thresh', None)
    max_replay = kwargs.get('max_replay', None)
    add_predecessors = kwargs.get('add_predecessors', None)
    replay_forbidden_walls = kwargs.get('replay_forbidden_walls', True)
    dec_weights = kwargs.get('dec_weights', np.array([0.75, 0.2, 0.05]))
    rep_weights = kwargs.get('rep_weights', dec_weights)

    # Arguments about saving
    save_data = kwargs.get('save_data', False)
    save_path = kwargs.get('save_path', None)
    save_tag = kwargs.get('save_tag', None)

    # 0) Creating the environment and the agent within
    dTm = DTMaze(forbidden_walls=env_forbidden_walls, restricted_dT=restricted_dT, slip_prob=slip_prob)
    dTm.place_agent(start_pos)

    agent = None
    agent = RLagent(dTm, model_type, gamma, kappa, decision_rule, alpha=alpha,
                    beta=beta, epsilon=epsilon, known_env=known_env,
                    replay_type=replay_type, event_content=event_content, event_handle=event_handle,
                    replay_thresh=replay_thresh, max_replay=max_replay,
                    dec_weights=dec_weights, rep_weights=rep_weights,
                    add_predecessors=add_predecessors, forbidden_walls=replay_forbidden_walls)
    if model == 'MB':
        # 1) Pre-training if the agent is MB
        run_experiment(pre_training, start_pos, dTm, agent, use_epochs=use_epochs, pre_training=True)

    # 2) Preparing the experiment
    for r_idx in range(len(rew_loc)):
        dTm.place_reward(rew_loc[r_idx], rew_val[r_idx], rew_prob[r_idx])
    dTm.place_agent(start_pos)
    # agent.update_agent(dTm)
    if save_data:
        dTm.toggle_save()
        agent.toggle_save()

    # 3) Running the experiment (with and without reward change)
    if rew_change is None:
        run_experiment(num_runs, start_pos, dTm, agent, use_epochs=use_epochs, replay_thresh=replay_thresh,
                       replay_every_step=replay_every_step)
    else:
        run_experiment(rew_change, start_pos, dTm, agent, use_epochs=use_epochs, replay_thresh=replay_thresh,
                       replay_every_step=replay_every_step)
        dTm.reset_reward()
        for r_idx in range(len(new_rew_loc)):
            dTm.place_reward(new_rew_loc[r_idx], new_rew_val[r_idx], new_rew_prob[r_idx])
        # agent.update_agent(dTm)
        run_experiment(num_runs - rew_change, start_pos, dTm, agent, use_epochs=use_epochs, replay_thresh=replay_thresh,
                       replay_every_step=replay_every_step)

    # 4) Save everything
    if save_data:
        dTm.dump_env(path=save_path, label=save_tag)
        agent.dump_agent(path=save_path, label=save_tag)


def run_experiment(num_runs: int, start_pos: int, env: Env, agent: RLagent, **kwargs):
    """
    Runs an experiment of a pre-defined length
    :param num_runs: how many steps/epochs we are modelling
    :param start_pos: where the agent is starting from
    :param env: what is the environment
    :param agent: what is the agent
    :param kwargs:
        use_epochs: if [True] we use epochs instead of steps. An epoch ends when a reward is received
        replay_thresh: what is the threshold to trigger replay (if None, no replay)
        pre_training: is this a pre-training setting [True -- no need to learn Q values] or not [False]
        replay_every_step: do I replay after each step [True -- default] of only after getting a reward [False]
    :return:
    """
    if num_runs is None:
        return
    use_epochs = kwargs.get('use_epochs', False)
    replay_thresh = kwargs.get('replay_thresh', None)
    pre_training = kwargs.get('pre_training', False)
    if pre_training and agent.agent_type() == 'MF':
        raise ValueError('Cannot pre-train a model-free agent.')
    replay_every_step = kwargs.get('replay_every_step', True)

    step = 0
    pbar = tqdm(total=num_runs)
    while step < num_runs:
        # 1) Observe the environment
        s = env.curr_state()
        a_poss = env.possible_moves(s)

        # 2) Choose an action
        a = agent.choose_action(s, a_poss)

        # 3) Perform a step
        s_prime, r = env.step(s, a)

        # 4) Learn
        hr, ht = agent.model_learning(s, a, s_prime, r)  # The epistemic rewards
        if not pre_training:
            delta_C = agent.inference(s, a, s_prime, np.array([r, hr, ht]))
            if replay_thresh is not None and replay_every_step and abs(delta_C) > replay_thresh:
                agent.memory_replay(s=s)

        # 5) Return to start if I must
        if r > 0:
            env.place_agent(start_pos)
            if use_epochs:
                step += 1
                pbar.update(1)
            if not replay_every_step:
                agent.memory_replay(s=env.curr_state())
        if not use_epochs:
            step += 1
            pbar.update(1)
