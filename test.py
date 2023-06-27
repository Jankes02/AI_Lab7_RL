import os
from envs.frozen_lake import FrozenLake
from gui.main_pygame import main_pygame
from q_agent import QAgent

from glob import glob
import datetime

if __name__ == '__main__':
    render = True
    mode = 'test'
    env = FrozenLake()
    agent = QAgent(n_states=env.n_states, n_actions=env.n_actions)

    date = 'run-{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now()).replace(':', '-')
    save_path_dir = '/'.join(['saved_models', env.name, agent.name, date])

    def get_prev_run_model(base_dir):
        dirs = glob(os.path.dirname(base_dir) + '/*')
        dirs.sort(reverse=True)
        if len(dirs) == 0:
            raise AssertionError("Run code in 'train' mode first.")
        return dirs[0] + '/model.npy'

    state_path = get_prev_run_model(save_path_dir)
    print(f"Testing model from latest run.")
    print(f"\tLoading agent state from {state_path}")
    agent.load(state_path)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
        agent.eps_min = 0.0

    print(f"Env name: {env.__class__.__name__}")
    print(f"Mode: {mode}")

    main_pygame(env, agent, render=render,
                num_episodes=5000, test_mode=(mode == 'test'))
