from itertools import count

import matplotlib.pyplot as plt
import torch

from dqn_model import env, get_screen
from train import policy_net, target_net, select_action, plot_durations, optimize_model, memory, episode_durations
from utils import device, TARGET_UPDATE

# Hyperparameters
num_episodes = 50


# Main training loop
def main():
    # Initialize environment
    for i_episode in range(num_episodes):
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        for t in count():
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen = last_screen
            else:
                next_state = None

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
