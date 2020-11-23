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
import random
import torch
import torch.utils.tensorboard as tb
from replay import ReplayBuffer, PrioritizedReplay, HindsightReplay, PHEReplay
from utils import get_one_hot_np, her_sampler, GoalMountainCar, GoalMountainCarContinuous

def batch_to_torch_device(batch, device):
    return [torch.from_numpy(b).float().to(device) for b in batch]

def preproc_inputs(state, goal):
    return np.concatenate((state, goal), axis=-1)

def update_SAC(sac, replay, step, writer, batch_size=256, use_per=False, use_her=False, log_interval=20):
    if use_per:
        # Calculate beta depending on step
        batch = replay.sample(batch_size)
        batch = batch_to_torch_device(batch, sac.device)
        states, actions, reward, next_states, done, weights, idxes = batch
    else:
        batch = replay.sample(batch_size)
        batch = batch_to_torch_device(batch, sac.device)
        states, actions, reward, next_states, done = batch
        weights, batch_idxes = torch.ones_like(reward), None

    q_loss = sac.calc_critic_loss(states, actions, reward, next_states, done, weights)
    sac.update_critics(q_loss[0], q_loss[1])
    if use_per:
        new_priorities = q_loss[2] + 1e-6
        replay.update_priorities(idxes, new_priorities)

    actor_loss, log_probs = sac.calc_actor_loss(states, weights)
    sac.update_actor(actor_loss)

    alpha_loss = sac.calc_entropy_tuning_loss(log_probs, weights)
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


def plot_success(reward_history, time_steps):
    # number of episodes for rolling average
    window = 50

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
    rolling_mean = pd.Series(reward_history).rolling(window).mean()
    std = pd.Series(reward_history).rolling(window).std()
    ax1.plot(time_steps, rolling_mean)
    ax1.fill_between(
        time_steps,
        rolling_mean - std,
        rolling_mean + std,
        color="blue",
        alpha=0.2,
    )
    ax1.set_title("Episode Return Moving Average")
    ax1.set_xlabel("Learning Update Step")
    ax1.set_ylabel("Average Return")

    ax2.plot(time_steps, reward_history)
    ax2.set_title("Return per Episode")
    ax2.set_xlabel("Learning Update Step")
    ax2.set_ylabel("Return")

    fig.tight_layout(pad=2)
    plt.savefig("sac.png")
    plt.show()


def init_environment(env_name):
    """
    Initialize the gym environment
    """
    if env_name == "Door":
        env = DoorWrapper(
            suite.make(
                "Door",
                robots="Sawyer",                # use Sawyer robot
                use_camera_obs=False,           # do not use pixel observations
                has_offscreen_renderer=False,   # not needed since not using pixel obs
                has_renderer=False,              # make sure we can render to the screen
                reward_shaping=False,            # use dense rewards
                control_freq=20,                # control should happen fast enough so that simulation looks smooth
            )
        )
    elif env_name == "MountainCarContinuous-v0":
        env = GoalMountainCarContinuous(gym.make(env_name))
    elif env_name == "MountainCar-v0":
        env = GoalMountainCar(gym.make(env_name))
    else:
        env = gym.make(env_name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    return env, discrete


def seed_random(env, rand_seed):
    random.seed(rand_seed)
    env.seed(rand_seed)
    env.action_space.seed(rand_seed)
    torch.manual_seed(rand_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rand_seed)
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

def get_env_params(env):
    obs = env.reset()
    if isinstance(env, GoalMountainCar) or isinstance(env, GoalMountainCarContinuous):
        # Cut through the wrapper
        env = env.env
    # close the environment
    if type(env.action_space) is gym.spaces.Box:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': action_dim,
            }
    if type(env.action_space) is not gym.spaces.Discrete:
        params['action_max'] = env.action_space.high[0]
    params['max_timesteps'] = env._max_episode_steps
    return params

def evaluate_goal(args, env, sac, writer, step):
    total_success_rate = []
    for _ in range(10):
        per_success_rate = []
        observation = env.reset()
        obs = observation['observation']
        g = observation['desired_goal']
        done = False
        while not done:
            with torch.no_grad():
                input_tensor = preproc_inputs(obs, g)
                action = sac.get_action(input_tensor, eval=True)
                # convert the actions
            observation_new, _, done, info = env.step(action)
            obs = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info['is_success'])
        total_success_rate.append(per_success_rate)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])
    return local_success_rate

def evaluate_SAC(args, env, sac, writer, step):
    reward_cum = 0
    done = False
    state = env.reset()
    if args.goal_env:
        g = state['desired_goal']
        obs = state['observation']
        ag = state['achieved_goal']

    while not done:
        if args.goal_env:
            state = preproc_inputs(obs, g)
        action = sac.get_action(state, eval=True)
        state, reward, done, _ = env.step(action)
        if args.goal_env:
            g = state['desired_goal']
            obs = state['observation']        
        reward_cum += reward

    return reward_cum


def train(args):
    env, discrete = init_environment(env_name=args.env_name)
    params = None
    if args.goal_env:
        params = get_env_params(env)
        success_rate = []
    thresh = env.spec.reward_threshold
    print("Starting training!  Need {} to solve".format(thresh))
    print(env)

    seed_random(env, args.rand_seed)
    device = init_device()
    writer = init_logger(log_dir=args.log_dir)

    sac = SAC(env, alpha=args.alpha, at=args.alph_tune, dis=discrete, params=params)
    if args.continue_training:
        sac.load()
    sac.init_opt(lr=args.learning_rate)
    sac = sac.to(device)
    act_size = sac.actor.action_space
    if args.per:
        replay = PrioritizedReplay(args.buff_size, sac.actor.state_space, act_size)
    elif args.her:
        her_sample = her_sampler('future',  4, env.compute_reward)
        replay = HindsightReplay(params, args.buff_size, her_sample.sample_her_transitions)
    elif args.pher:
        # replay = PHEReplay()
        print("PHER not yet supported")
    else:
        replay = ReplayBuffer(args.buff_size, sac.actor.state_space, act_size)

    # Need to go another level for continuous because nested
    
    time_limit = params['max_timesteps'] if args.goal_env else env._max_episode_steps
    total_steps = 0
    updates = 0
    max_reward = -float("inf")
    reward_history = []
    time_steps = []
    updates_history = []
    eval_history = []
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        if args.goal_env:
            obs = state['observation']
            ag = state['achieved_goal']
            g = state['desired_goal']
        while not done and episode_steps < time_limit:
            # If her, concatenate state and goal
            if args.goal_env:
                state = preproc_inputs(obs, g)

            if total_steps < args.start_steps:
                action = env.action_space.sample()
                action = np.array(action)
            else:
                with torch.no_grad():
                    action = sac.get_action(state)

            next_state, reward, done, info = env.step(action)
            if sac.discrete:
                action = get_one_hot_np(action, sac.soft_q1.action_space)
            total_steps += 1
            episode_steps += 1
            episode_reward += reward
            # Ignore done signal if it was timeout related
            done = False if episode_steps==time_limit else done

            if args.goal_env:
                next_obs = next_state['observation']
                next_obs = preproc_inputs(next_obs, g)
                replay.store(state, action, next_obs, reward, done)
                obs = next_state['observation']
                ag = next_state['achieved_goal']
            else:
                replay.store(state, action, next_state, reward, done)
                state = next_state

            if len(replay) > args.batch_size:
                for i in range(args.updates_per_step):
                    update_SAC(
                        sac,
                        replay,
                        updates,
                        writer,
                        batch_size=args.batch_size,
                        use_per=args.per,
                    )
                    updates += 1

        if i_episode % args.eval_freq == 0 and not args.goal_env:
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
            if avg_reward > max_reward:
                print("Saving model...")
                max_reward = avg_reward
                sac.save()
            print("Avg Eval Reward {:.2f}".format(avg_reward))
            sac.train()
        elif i_episode % args.eval_freq == 0:
            print("Evaluating")
            sac.eval()
            avg_reward = evaluate_goal(args, env, sac, writer, total_steps)
            success_rate.append(avg_reward)
            updates_history.append(updates)
            if writer is not None:
                writer.add_scalar("stats/eval_reward", avg_reward, total_steps)
            if avg_reward > max_reward:
                print("Saving model...")
                max_reward = avg_reward
                sac.save()
            print("Avg Eval Reward {:.2f}".format(avg_reward))
            sac.train()
        # Calculate score to determine when the environment has been solved
        reward_history.append(episode_reward)
        time_steps.append(total_steps)
        mean_score = np.mean(reward_history[-100:])
        if writer is not None:
            writer.add_scalar("stats/reward", episode_reward, total_steps)
            writer.add_scalar("stats/avg_reward", mean_score, total_steps)

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, avg reward: {}".format(i_episode, total_steps, episode_steps, round(episode_reward, 2), round(mean_score, 2)))


        # if thresh is not None and mean_score > thresh:
        #     print("Solved after {} episodes!".format(i_episode))
        #     print("And {} environment steps".format(total_steps))
        #     break

        if total_steps > args.steps:
            break

    fname = "results.out"
    data = np.array(reward_history, dtype=np.float32)
    time = np.array(time_steps, dtype=np.float32)
    dat = np.stack((data, time), axis=0)
    np.savetxt(fname, dat)

    if args.goal_env:
        fname = "success.out"
        success = np.array(success_rate, dtype=np.float32)
        updates = np.array(updates_history, dtype=np.float32)
        dat = np.stack((success, updates), axis=0)
        np.savetxt(fname, dat)
    plot_success(reward_history, time_steps)
