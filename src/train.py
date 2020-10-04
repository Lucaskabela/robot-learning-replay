"""
train.py

AUTHOR: Lucas Kabela

PURPOSE: This file defines the code for training the neural networks in pytorch
"""
import gym
import models
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.tensorboard as tb


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
        color="orange",
        alpha=0.2,
    )
    ax1.set_title("Episode Length Moving Average ({}-episode window)".format(window))
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Length")

    ax2.plot(policy.reward_history)
    ax2.set_title("Episode Length")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length")

    fig.tight_layout(pad=2)
    plt.show()


def update_policy(replay, policy, value, optimizer, val_optimizer, gamma=0.99):
    device = policy.device()

    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns).float().to(device)
    states = torch.tensor(
        [state for state, _, _ in replay], dtype=torch.float, device=device
    )
    vals = value(states).squeeze(1)
    if not math.isnan(returns.std()):
        returns = (returns - returns.mean()) / (
            returns.std() + np.finfo(np.float32).eps
        )
    with torch.no_grad():
        advantage = returns - vals

    for log_prob, R in zip(policy.saved_log_probs, advantage):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum().to(device)
    policy_loss.backward()
    optimizer.step()

    val_optimizer.zero_grad()
    F.mse_loss(vals, returns).backward()
    val_optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(policy_loss.item())
    policy.reward_history.append(np.sum(policy.rewards))
    policy.reset()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def train(args):
    print("Starting training!")
    env = gym.make("CartPole-v1")
    env.seed(1)
    torch.manual_seed(1)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    policy = models.Policy(env).to(device)
    value = models.Value(env).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    val_optimizer = optim.Adam(value.parameters(), lr=args.learning_rate)
    scores = []
    replay = []
    for episode in range(args.num_episodes):
        # Reset environment and record the starting state
        state = env.reset()

        for time in range(1000):
            action = policy.predict(state)

            # Uncomment to render the visual state in a window
            # env.render()

            # Step through environment using chosen action
            next_state, reward, done, _ = env.step(action.item())
            replay.append((state, action, reward))
            state = next_state

            # Save reward
            policy.rewards.append(reward)
            if done:
                break

        update_policy(replay, policy, value, optimizer, val_optimizer)
        replay = []
        # Calculate score to determine when the environment has been solved
        scores.append(time)
        mean_score = np.mean(scores[-100:])

        if episode % 50 == 0:
            print(
                "Episode {}\tAverage length (last 100 episodes): {:.2f}".format(
                    episode, mean_score
                )
            )

        if mean_score > env.spec.reward_threshold:
            print(
                "Solved after {} episodes! Running average is now {}. Last episode ran to {} time steps.".format(
                    episode, mean_score, time
                )
            )
            break
    plot_success(policy)


# def train(args):
#     from os import path

#     model = None  # Planner()
#     train_logger, valid_logger = None, None
#     if args.log_dir is not None:
#         train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"))
#         valid_logger = tb.SummaryWriter(path.join(args.log_dir, "valid"))

#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     model = model.to(device)
#     if args.continue_training:
#         model = load_model(model)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

#     train_data = load_data("drive_data")
#     loss = torch.L1Loss()
#     global_step = 0
#     for epoch in range(args.num_epoch):

#         model.train()
#         losses = []
#         for img, label in train_data:
#             img, label = img.to(device), label.to(device)

#             pred = model(img)
#             loss_val = loss(pred, label)

#             if train_logger is not None:
#                 train_logger.add_scalar("loss", loss_val, global_step)
#                 if global_step % 100 == 0:
#                     fig, ax = plt.subplots(1, 1)
#                     ax.imshow(TF.to_pil_image(img[0].cpu()))
#                     ax.add_artist(plt.Circle(label[0], 2, ec="g", fill=False, lw=1.5))
#                     ax.add_artist(plt.Circle(pred[0], 2, ec="r", fill=False, lw=1.5))
#                     train_logger.add_figure("viz", fig, global_step)
#                     del ax, fig

#             optimizer.zero_grad()
#             loss_val.backward()
#             optimizer.step()
#             global_step += 1

#             losses.append(loss_val.detach().cpu().numpy())

#         avg_loss = np.mean(losses)
#         if train_logger is None:
#             print("epoch %-3d \t loss = %0.3f" % (epoch, avg_loss))
#         save_model(model)

#     save_model(model)
