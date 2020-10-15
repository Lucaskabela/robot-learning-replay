"""
train.py

AUTHOR: Lucas Kabela

PURPOSE: This file defines the code for training the neural networks in pytorch
"""
import gym
from models import SAC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard as tb
from replay import ReplayBuffer
from utils import get_one_hot_np, NormalizedActions


def batch_to_torch_device(batch, device):
    return [torch.from_numpy(b).float().to(device) for b in batch]


def update_SAC(sac, replay, step, writer, batch_size=256, log_interval=100):
    batch = replay.sample(batch_size)
    batch = batch_to_torch_device(batch, sac.device())
    states, actions, reward, next_states, done = batch
    q_loss = sac.calc_critic_loss(states, actions, reward, next_states, done)
    sac.update_critics(q_loss[0], q_loss[1])

    actor_loss, log_probs = sac.calc_actor_loss(states)
    sac.update_actor(actor_loss)

    alpha_loss = sac.calc_entropy_tuning_loss(log_probs)
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
            log_probs.detach().mean().item(),
            step,
        )


def plot_success(reward_history):
    # number of episodes for rolling average
    window = 50

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
    rolling_mean = pd.Series(reward_history).rolling(window).mean()
    std = pd.Series(reward_history).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(
        range(len(reward_history)),
        rolling_mean - std,
        rolling_mean + std,
        color="blue",
        alpha=0.2,
    )
    ax1.set_title("Episode Length Moving Average")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Length")

    ax2.plot(reward_history)
    ax2.set_title("Episode Length")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length")

    fig.tight_layout(pad=2)
    plt.savefig("sac.png")
    plt.show()


def init_environment(env_name):
    """
    Initialize the gym environment, normalize if continuous
    and return
    """
    env = gym.make(env_name)
    discrete = False
    if type(env.action_space) is gym.spaces.Discrete:
        discrete = True
    else:
        env = NormalizedActions(env)
    return env, discrete


def seed_random(env, rand_seed):
    env.seed(rand_seed)
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)


def init_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def init_logger(log_dir=None):
    if log_dir is not None:
        writer = tb.SummaryWriter(log_dir=log_dir)
    else:
        writer = None
    return writer


def train(args):
    env, discrete = init_environment(env_name=args.env_name)
    thresh = env.spec.reward_threshold
    print("Starting training!  Need {} to solve".format(thresh))
    print(env)
    # print(env.observation_space["observation"].shape[0])
    seed_random(env, args.rand_seed)
    device = init_device()
    writer = init_logger(log_dir=args.log_dir)

    sac = SAC(env, device, at=args.alph_tune, disc=discrete).to(device)
    sac.init_opt(lr=args.learning_rate)
    reward_history = []
    reward_cum = 0
    replay = ReplayBuffer(args.buff_size, sac.actor.state_space, sac.actor.action_space)
    step = 0
    episode = 0
    while step < args.steps and episode < args.num_episodes:
        # Reset environment and record the starting state
        state = env.reset()
        reward_cum = 0
        done = False
        time = 0
        while not done and time < args.time_limit:
            state = torch.from_numpy(state).float().to(device)
            action = sac.get_action(state)
            next_state, reward, done, _ = env.step(action)
            if sac.discrete:
                action = get_one_hot_np(action, sac.soft_q1.action_space)
            replay.store(state.cpu(), action, next_state, reward, done)
            state = next_state
            reward_cum += reward
            step += 1
            time += 1
            if len(replay) > args.batch_size:
                update_SAC(
                    sac,
                    replay,
                    step,
                    writer,
                    batch_size=args.batch_size,
                )

        # Calculate score to determine when the environment has been solved
        reward_history.append(reward_cum)
        mean_score = np.mean(reward_history[-100:])
        if writer is not None:
            writer.add_scalar("stats/reward", reward_cum, step)
            writer.add_scalar("stats/avg_reward", mean_score, step)

        print(
            "Episode {} Steps {} Reward {:.2f} Avg reward {:.2f}".format(
                episode, step, reward_history[-1], mean_score
            )
        )

        episode += 1
        if thresh is not None and mean_score > thresh:
            print("Solved after {} episodes!".format(episode))
            print("And {} environment steps".format(step))
            break

    fname = "results.out"
    data = np.array(reward_history)
    np.savetxt(fname, data)
    plot_success(reward_history)
