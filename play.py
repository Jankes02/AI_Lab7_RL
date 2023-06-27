from envs.frozen_lake import FrozenLake
from gui.main_pygame import main_pygame
from gui.manual_pygame_agent import ManualPygameAgent


if __name__ == '__main__':
    render = True
    mode = 'test'
    env = FrozenLake()
    agent = ManualPygameAgent()

    print(f"Env name: {env.__class__.__name__}")
    print(f"Mode: {mode}")

    main_pygame(env, agent, render=render,
                num_episodes=5000, test_mode=(mode == 'test'))
