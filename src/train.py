"""
train.py

AUTHOR: Lucas Kabela

PURPOSE: This file defines the code for training the neural networks in pytorch
"""
import gym
from .models import save_model, load_model
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard as tb
import torchvision.transforms.functional as TF

env = gym.make("CartPole-v1")
env.seed(1)
torch.manual_seed(1)


def predict(state):
    return torch.zeros(1)


def train(args):

    scores = []
    replay = []
    for episode in range(args.num_episodes):
        # Reset environment and record the starting state
        state = env.reset()

        for time in range(1000):
            action = predict(state)

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

        # update_policy(replay)
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


def train(args):
    from os import path

    model = None  # Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, "valid"))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    if args.continue_training:
        model = load_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_data = load_data("drive_data")
    loss = torch.L1Loss()
    global_step = 0
    for epoch in range(args.num_epoch):

        model.train()
        losses = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            pred = model(img)
            loss_val = loss(pred, label)

            if train_logger is not None:
                train_logger.add_scalar("loss", loss_val, global_step)
                if global_step % 100 == 0:
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(TF.to_pil_image(img[0].cpu()))
                    ax.add_artist(plt.Circle(label[0], 2, ec="g", fill=False, lw=1.5))
                    ax.add_artist(plt.Circle(pred[0], 2, ec="r", fill=False, lw=1.5))
                    train_logger.add_figure("viz", fig, global_step)
                    del ax, fig

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

            losses.append(loss_val.detach().cpu().numpy())

        avg_loss = np.mean(losses)
        if train_logger is None:
            print("epoch %-3d \t loss = %0.3f" % (epoch, avg_loss))
        save_model(model)

    save_model(model)
