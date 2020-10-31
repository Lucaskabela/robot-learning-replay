"""
train.py

AUTHOR: Lucas Kabela

PURPOSE: This file defines the code for training the neural networks in pytorch
"""
import gym
import itertools
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


def update_SAC(sac, replay, step, writer, batch_size=256, log_interval=20):
    batch = replay.sample(batch_size)
    batch = batch_to_torch_device(batch, sac.device)
    states, actions, reward, next_states, done = batch
    reward = reward.unsqueeze(1)
    done = done.unsqueeze(1)
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
        if sac.adjust_alpha:
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
    ax1.set_title("Episode Return Moving Average")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Return")

    ax2.plot(reward_history)
    ax2.set_title("Return per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Return")

    fig.tight_layout(pad=2)
    plt.savefig("sac.png")
    plt.show()


def init_environment(env_name):
    """
    Initialize the gym environment
    """
    env = gym.make(env_name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    return env, discrete


def seed_random(env, rand_seed):
    env.seed(rand_seed)
    env.action_space.seed(rand_seed)
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


def evaluate_SAC(args, env, sac, writer, step):
    reward_cum = 0
    done = False
    state = env.reset()
    while not done:
        action = sac.get_action(state, eval=True)
        state, reward, done, _ = env.step(action)
        reward_cum += reward

    return reward_cum


def train(args):
    env, discrete = init_environment(env_name=args.env_name)
    thresh = env.spec.reward_threshold
    print("Starting training!  Need {} to solve".format(thresh))
    print(env)

    seed_random(env, args.rand_seed)
    device = init_device()
    writer = init_logger(log_dir=args.log_dir)

    sac = SAC(env, at=args.alph_tune, dis=discrete)
    sac.init_opt(lr=args.learning_rate)
    sac = sac.to(device)
    act_size = sac.actor.action_space
    replay = ReplayBuffer(args.buff_size, sac.actor.state_space, act_size)

    # Need to go another level for continuous because nested
    total_steps = 0
    updates = 0
    max_reward = -float("inf")

    reward_history = []
    eval_history = []
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        while not done:
            # Reset environment and record the starting state
            if total_steps < args.start_steps:
                action = env.action_space.sample()
                action = np.array(action)
            else:
                action = sac.get_action(state)
            if len(replay) > args.batch_size:
                for i in range(args.updates_per_step):
                    update_SAC(
                        sac,
                        replay,
                        updates,
                        writer,
                        batch_size=args.batch_size,
                    )
                    updates += 1

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            total_steps += 1
            episode_steps += 1

            if sac.discrete:
                action = get_one_hot_np(action, sac.soft_q1.action_space)
            # Ignore done signal if it was timeout related
            done = False if episode_steps==env._max_episode_steps else done
            replay.store(state, action, next_state, reward, done)

            state = next_state
                
        if i_episode % args.eval_freq == 0:
            print("Evaluating")
            sac.eval()
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                curr_reward = evaluate_SAC(args, env, sac, writer, total_steps)
                avg_reward += curr_reward
            avg_reward /= episodes
            eval_history.append((i_episode / args.eval_freq, avg_reward))
            if writer is not None:
                writer.add_scalar("stats/eval_reward", avg_reward, total_steps)
            if curr_reward > max_reward:
                print("Saving model...")
                max_reward = curr_reward
                sac.save()
            print("Avg Eval Reward {:.2f}".format(avg_reward))
            sac.train()

        # Calculate score to determine when the environment has been solved
        reward_history.append(episode_reward)
        mean_score = np.mean(reward_history[-100:])
        if writer is not None:
            writer.add_scalar("stats/reward", episode_reward, total_steps)
            writer.add_scalar("stats/avg_reward", mean_score, total_steps)

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_steps, episode_steps, round(episode_reward, 2)))


        if thresh is not None and mean_score > thresh:
            print("Solved after {} episodes!".format(i_episode))
            print("And {} environment steps".format(total_steps))
            break

        if total_steps > args.steps:
            break

    fname = "results.out"
    data = np.array(reward_history)
    np.savetxt(fname, data)
    plot_success(reward_history)
