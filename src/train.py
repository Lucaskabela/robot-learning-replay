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
from utils import get_one_hot_np, NormalizedActions, normalizer, her_sampler
from datetime import datetime

def batch_to_torch_device(batch, device):
    return [torch.from_numpy(b).float().to(device) for b in batch]


def update_SAC(sac, replay, step, writer, batch_size=256, use_per=False, log_interval=20):
    if use_per:
        # Calculate beta depending on step
        beta = .4
        batch = replay.sample(batch_size, beta)
        batch = batch_to_torch_device(batch, sac.device)
        states, actions, reward, next_states, done, weights, idxes = batch
    else:
        batch = replay.sample(batch_size)
        batch = batch_to_torch_device(batch, sac.device)
        states, actions, reward, next_states, done = batch
        weights, batch_idxes = torch.ones_like(reward), None

    # reward = reward.unsqueeze(1)
    # done = done.unsqueeze(1)
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
    env = gym.make(env_name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    return env, discrete


def seed_random(env, rand_seed):
    random.seed(rand_seed)
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

    sac = SAC(env, alpha=args.alpha, at=args.alph_tune, dis=discrete)
    if args.continue_training:
        sac.load()
    sac.init_opt(lr=args.learning_rate)
    sac = sac.to(device)
    act_size = sac.actor.action_space
    if args.per:
        replay = PrioritizedReplay(args.buff_size, sac.actor.state_space, act_size)
    elif args.her:
        # replay = HindisightReplay()
        print("HER not yet supported")
    elif args.pher:
        # replay = PHEReplay()
        print("PHER not yet supported")
    else:
        replay = ReplayBuffer(args.buff_size, sac.actor.state_space, act_size)

    # Need to go another level for continuous because nested
    total_steps = 0
    updates = 0
    max_reward = -float("inf")

    reward_history = []
    time_steps = []
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
                        use_per=args.per or args.pher,
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
    data = np.array(reward_history)
    time = np.array(time_steps)
    dat = np.stack((data, time), axis=0)
    np.savetxt(fname, dat)
    plot_success(reward_history, time_steps)

#-----------------------------------------------Goal Environments-----------------------------------------------#
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def preproc_inputs(obs, g, o_norm, g_norm):
    obs_norm = o_norm.normalize(obs)
    g_norm = g_norm.normalize(g)
    # concatenate the stuffs
    inputs = np.concatenate([obs_norm, g_norm])
    return inputs

def preproc_og(o, g):
    o = np.clip(o, -5, 5)
    g = np.clip(g, -5, 5)
    return o, g

def update_normalizer(episode_batch, sampler, o_norm, g_norm):
    mb_obs, mb_ag, mb_g, mb_actions = episode_batch
    mb_obs_next = mb_obs[:, 1:, :]
    mb_ag_next = mb_ag[:, 1:, :]
    # get the number of normalization transitions
    num_transitions = mb_actions.shape[1]
    # create the new buffer to store them
    buffer_temp = {'obs': mb_obs, 
                    'ag': mb_ag,
                    'g': mb_g, 
                    'actions': mb_actions, 
                    'obs_next': mb_obs_next,
                    'ag_next': mb_ag_next,
                    }
    transitions = sampler.sample_her_transitions(buffer_temp, num_transitions)
    obs, g = transitions['obs'], transitions['g']
    # pre process the obs and g
    transitions['obs'], transitions['g'] = preproc_og(obs, g)
    # update
    o_norm.update(transitions['obs'])
    g_norm.update(transitions['g'])
    # recompute the stats
    o_norm.recompute_stats()
    g_norm.recompute_stats()



def update_SAC_goal(sac, replay, step, writer, o_norm, g_norm, batch_size=256, use_per=False, log_interval=20):
    if use_per:
        # Calculate beta depending on step
        beta = .4
        batch = replay.sample(batch_size, beta)
        batch = batch_to_torch_device(batch, sac.device)
        states, actions, reward, next_states, done, weights, idxes = batch
    else:
        transitions = replay.sample(batch_size)
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = preproc_og(o_next, g)
        obs_norm = o_norm.normalize(transitions['obs'])
        goal_norm = g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, goal_norm], axis=1)
        obs_next_norm = o_norm.normalize(transitions['obs_next'])
        g_next_norm = g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        batch = batch_to_torch_device([inputs_norm, transitions["actions"], inputs_next_norm, transitions["r"]], sac.device)
        states, actions, next_states, reward = batch
        weights, batch_idxes, done = torch.ones_like(reward), None, torch.zeros_like(reward)

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

def eval_goal_agent(env, env_params, sac, o_norm, g_norm):
    total_success_rate = []
    for _ in range(10):
        per_success_rate = []
        observation = env.reset()
        obs = observation['observation']
        g = observation['desired_goal']
        for _ in range(env_params['max_timesteps']):
            with torch.no_grad():
                input_tensor = preproc_inputs(obs, g, o_norm, g_norm)
                action = sac.get_action(input_tensor)
            observation_new, _, _, info = env.step(action[0])
            obs = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info['is_success'])
        total_success_rate.append(per_success_rate)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])
    return local_success_rate

def train_goal(args):
    env, discrete = init_environment(env_name=args.env_name)
    seed_random(env, args.rand_seed)
    device = init_device()
    writer = init_logger(log_dir=args.log_dir)
    env_params = get_env_params(env)
    print(env)

    # Change this to the obs + goal
    sac = SAC(env, alpha=args.alpha, at=args.alph_tune, dis=discrete, env_params=env_params)
    if args.continue_training:
        sac.load()
    sac.init_opt(lr=args.learning_rate)
    sac = sac.to(device)
    
    act_size = sac.actor.action_space
    if args.per:
        replay = PrioritizedReplay(args.buff_size, sac.actor.state_space, act_size)
    elif args.her:
        sampler = her_sampler("future", 4, env.compute_reward)
        replay = HindsightReplay(env_params, args.buff_size, sampler.sample_her_transitions)
    elif args.pher:
        # replay = PHEReplay()
        print("PHER not yet supported")
    else:
        replay = ReplayBuffer(args.buff_size, sac.actor.state_space, act_size)

    o_norm = normalizer(size=env_params['obs'], default_clip_range=5)
    g_norm = normalizer(size=env_params['goal'], default_clip_range=5)

    # Need to go another level for continuous because nested
    total_steps = 0
    episode = 0
    updates = 0
    max_reward = -float("inf")

    reward_history = []
    time_steps = []
    eval_history = []

    for epoch in range(args.epochs):
        for _ in range(args.n_cycles):
            mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
            for _ in range(2):
                ep_reward = 0
                # reset the rollouts
                ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                # reset the environment
                observation = env.reset()
                obs = observation['observation']
                ag = observation['achieved_goal']
                g = observation['desired_goal']
                # start to collect samples
                for t in range(env_params['max_timesteps']):
                    if total_steps < args.start_steps:
                        action = env.action_space.sample()
                        action = np.array(action)
                    else:
                        with torch.no_grad():
                            input_tensor = preproc_inputs(obs, g, o_norm, g_norm)
                            action = sac.get_action(input_tensor)
                    # feed the actions into the environment
                    observation_new, reward, done, info = env.step(action)
                    ep_reward += reward
                    obs_new = observation_new['observation']
                    ag_new = observation_new['achieved_goal']
                    # append rollouts
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    ep_g.append(g.copy())
                    ep_actions.append(action.copy())
                    # re-assign the observation
                    obs = obs_new
                    ag = ag_new
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                mb_obs.append(ep_obs)
                mb_ag.append(ep_ag)
                mb_g.append(ep_g)
                mb_actions.append(ep_actions)
            # convert them into arrays
            mb_obs = np.array(mb_obs)
            mb_ag = np.array(mb_ag)
            mb_g = np.array(mb_g)
            mb_actions = np.array(mb_actions)
            # store the episodes
            replay.store([mb_obs, mb_ag, mb_g, mb_actions])
            update_normalizer([mb_obs, mb_ag, mb_g, mb_actions], sampler, o_norm, g_norm)
            for _ in range(args.updates_per_step):
                # train the network
                update_SAC_goal(sac, replay, updates, writer, o_norm, g_norm, batch_size=args.batch_size)
                updates += 1

        # start to do the evaluation
        success_rate = eval_goal_agent(env, env_params, sac, o_norm, g_norm)
        print('[{}] epoch is: {}, steps are {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, updates, success_rate))
        if success_rate > max_reward:
            print("New High Score!")
            max_reward = success_rate
            sac.save()


        # Calculate score to determine when the environment has been solved
        reward_history.append(success_rate)
        time_steps.append(updates)
        mean_score = np.mean(reward_history[-100:])
        if writer is not None:
            writer.add_scalar("stats/reward", success_rate, updates)
            writer.add_scalar("stats/avg_reward", mean_score, updates)


        # if thresh is not None and mean_score > thresh:
        #     print("Solved after {} episodes!".format(i_episode))
        #     print("And {} environment steps".format(total_steps))
        #     break

        if total_steps > args.steps:
            break

    fname = "results.out"
    data = np.array(reward_history)
    time = np.array(time_steps)
    dat = np.stack((data, time), axis=0)
    np.savetxt(fname, dat)
    plot_success(reward_history, time_steps)