# Youtube reference for paper implementation - https://www.youtube.com/watch?v=GJJc1t0rtSU&ab_channel=freeCodeCamp.org
# repository - https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/blob/master/DDPG/main_ddpg.py

from ddpg_tf_org import Agent
import gym
import numpy as np

from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    agent = Agent(alpha=0.001, beta=0.001, input_dims=[3], tau=0.001, env=env, batch_size=64, layer1_size=400,
                  layer2_size=300, n_actions=1)

    score_history = []
    np.random.seed(0)

    for i in range(1000):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
        score_history.append(score)
        print('episode ', 1, 'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))

    filename = 'pendulum.png'
    figure_file = 'plots/' + filename + '.png'

    x = [i + 1 for i in range(1000)]
    plot_learning_curve(x, score_history, filename)
