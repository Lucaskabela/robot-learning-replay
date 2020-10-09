"""
train.py

AUTHOR: Lucas Kabela

PURPOSE: This file defines the code for training the neural networks in pytorch
"""
import gym
from models import SAC
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard as tb
from replay import ReplayBuffer
from utils import guard_q_actions


def batch_to_torch_device(batch, device):
    return [torch.from_numpy(b).float().to(device) for b in batch]


def update_SAC(sac, replay, step, writer, batch_size=256, log_interval=100):
    batch = replay.sample(batch_size)
    batch = batch_to_torch_device(batch, sac.device())
    states, action, reward, next_states, done = batch

    if not math.isnan(reward.std()):
        stable_denom = reward.std() + np.finfo(np.float32).eps
        reward = (reward - reward.mean()) / stable_denom

    q_loss = sac.calc_critic_loss(states, action, reward, next_states, done)
    sac.update_critics(q_loss[0], q_loss[1])

    actor_loss, log_action_probabilities = sac.calc_actor_loss(states)
    sac.update_actor(actor_loss)

    alpha_loss = sac.calc_entropy_tuning_loss(log_action_probabilities)
    sac.update_entropy(alpha_loss)
    sac.soft_copy()

    if step % log_interval == 0 and writer is not None:
        writer.add_scalar("loss/Q1", q_loss[0].detach().item(), step)
        writer.add_scalar("loss/Q2", q_loss[1].detach().item(), step)
        writer.add_scalar("loss/policy", actor_loss.detach().item(), step)
        writer.add_scalar("loss/alpha", alpha_loss.detach().item(), step)
        writer.add_scalar("stats/alpha", sac.alpha, step)
        writer.add_scalar(
            "stats/entropy",
            log_action_probabilities.detach().mean().item(),
            step,
        )
    # Save and intialize episode history counters
    sac.actor.loss_history.append(actor_loss.item())
    sac.actor.reset()
    del sac.actor.rewards[:]
    del sac.actor.saved_log_probs[:]


def plot_success(policy):
    # number of episodes for rolling average
    window = 50

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
    rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
    std = pd.Series(policy.reward_history).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(
        range(len(policy.reward_history)),
        rolling_mean - std,
        rolling_mean + std,
        color="blue",
        alpha=0.2,
    )
    ax1.set_title("Episode Length Moving Average")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Length")

    ax2.plot(policy.reward_history)
    ax2.set_title("Episode Length")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length")

    fig.tight_layout(pad=2)
    plt.show()


def train(args):
    print("Starting training!")
    env = gym.make("CartPole-v1")
    env.seed(1)
    torch.manual_seed(1)
    if args.log_dir is not None:
        writer = tb.SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    sac = SAC(env, disc=True).to(device)
    sac.init_opt(lr=args.learning_rate)
    scores = []
    reward_cum = 0
    replay = ReplayBuffer(10000, sac.actor.state_space, sac.actor.action_space)
    step = 0
    for episode in range(args.num_episodes):
        # Reset environment and record the starting state
        state = env.reset()

        for time in range(1000):
            state = torch.from_numpy(state).float().to(device)
            action = sac.get_action(state)

            # Uncomment to render the visual state in a window
            # env.render()

            # Step through environment using chosen action
            next_state, reward, done, _ = env.step(action)
            replay.store(state.cpu(), action, next_state, reward, done)
            state = next_state
            reward_cum += reward
            # Save reward
            sac.actor.rewards.append(reward)
            if done:
                break
            if len(replay) > args.batch_size:
                update_SAC(
                    sac,
                    replay,
                    step,
                    writer,
                    batch_size=args.batch_size,
                )
            step += 1
        # Calculate score to determine when the environment has been solved
        scores.append(time)
        mean_score = np.mean(scores[-100:])

        if episode % 50 == 0:
            print("Episode {} Avg length {:.2f}".format(episode, mean_score))

        if mean_score > env.spec.reward_threshold:
            print("Solved after {} episodes!".format(episode))
            break

    plot_success(sac.actor)
