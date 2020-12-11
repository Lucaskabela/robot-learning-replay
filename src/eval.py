# eval.py
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import pandas as pd

def _parse_args():
    """
    Parses the commandline arguments for running an expirement trail/series
    of trials

    Args:

    Returns:
        args: the parsed arguments in a new namespace
    """

    parser = argparse.ArgumentParser(
        description="Arguments for evaluating results from Experience Replay project"
    )
    parser.add_argument(
        "--files", 
        default=["../results/hyperparams/mountaincarcontinuous_v0/results_er_50k_64.out",
                "../results/hyperparams/mountaincarcontinuous_v0/results_er_50k_64.out", 
                 "../results/hyperparams/mountaincarcontinuous_v0/results_her_50k_512.csv",
                 "../results/hyperparams/mountaincarcontinuous_v0/results_her_50k_512.csv",
        ])
    parser.add_argument("--goal", action='store_true')

    args = parser.parse_args()
    return args


def load_csv(filename, solved_goal=.25):
    print(filename)
    with open(filename, 'r') as f:
        df = pd.read_csv(f)
        returns = df['evaluation/Returns Mean'].to_numpy()
        success = df['evaluation/env_infos/final/is_success Mean'].to_numpy()
        updates = df['trainer/num train calls'].to_numpy()
        returns = returns[~np.isnan(returns)]
        success = success[~np.isnan(success)]
        updates = updates[~np.isnan(updates)]
        print("Max return: ", max(returns))
        print("Max success rate: ", max(success))
        for idx, val in enumerate(success):
            # Hardcoded solve for half cheetah
            if val > solved_goal:
                print("Solved on step: ", updates[idx])
                break
        return returns, updates, success, updates


def load_out(filename, solved_threshold=-.1, solved_goal=.25, goal_env=False):
    print(filename)
    with open(filename, "r") as f:
        lines = f.readlines()
        returns = [float(n) for n in lines[0].strip().split()]
        update_step = [float(n) for n in lines[1].strip().split()]
        print("Max return: ", max(returns))
        for idx, val in enumerate(returns):
            # Hardcoded solve for half cheetah
            if val > solved_threshold:
                print("Solved on step: ", update_step[idx])
                break

    if goal_env:
        filename2 = filename[:11] + filename[11:].replace("results", "success")
        with open(filename2, "r") as f:
            lines = f.readlines()
            success_rate = [float(n) for n in lines[0].strip().split()]
            update_succ = [float(n) for n in lines[1].strip().split()]
            print("Max success rate: ", max(success_rate))
            for idx, val in enumerate(success_rate):
                # Hardcoded solve for half cheetah
                if val > solved_goal:
                    print("Solved on step: ", update_succ[idx])
                    break
        return np.array(returns), np.array(update_step), np.array(success_rate), np.array(update_succ)
    
    return np.array(returns), np.array(update_step)


def plot_returns(returns, time_steps):
    window = 100
    labels = ['ER', 'PER', 'HER', 'PHER']
    fig = plt.figure()
    jet = cm = plt.get_cmap('gist_rainbow') 
    cNorm  = colors.Normalize(vmin=0, vmax=3)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    for idx, cl in enumerate(returns):
        rolling_mean = pd.Series(returns[idx]).rolling(window).mean()
        std = pd.Series(returns[idx]).rolling(window).std()
        plt.plot(
            time_steps[idx], 
            rolling_mean,
            color=scalarMap.to_rgba(idx),
            label=labels[idx]
        )
        plt.fill_between(
            time_steps[idx],
            rolling_mean - std,
            rolling_mean + std,
            color=scalarMap.to_rgba(idx),
            alpha=0.1,            
        )
    plt.title("MountainCar-v0: Batch Size 512, Buffer Size 50,00")
    plt.xlabel("Learning Update Step")
    plt.ylabel('Average Return')
    plt.legend(loc='upper left')
    fig.savefig('returns.png')
    plt.show()


def plot_successes(successes, time_steps):
    window=100
    labels = ['ER', 'PER', 'HER', 'PHER']
    fig = plt.figure()
    jet = cm = plt.get_cmap('gist_rainbow') 
    cNorm  = colors.Normalize(vmin=0, vmax=3)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    for idx, cl in enumerate(successes):
        rolling_mean = pd.Series(successes[idx]).rolling(window).mean()
        std = pd.Series(successes[idx]).rolling(window).std()
        plt.plot(
            time_steps[idx], 
            rolling_mean,
            color=scalarMap.to_rgba(idx),
            label=labels[idx]
        )
        plt.fill_between(
            time_steps[idx],
            rolling_mean - std,
            rolling_mean + std,
            color=scalarMap.to_rgba(idx),
            alpha=0.1,            
        )
    plt.title("MountainCar-v0: Batch Size 512, Buffer Size 50,00")
    plt.xlabel("Learning Update Step")
    plt.ylabel('Success Rate')
    plt.legend(loc='upper left')
    fig.savefig('returns.png')
    plt.show()


def main():
    args = _parse_args()
    returns = []
    updates_returns = []
    successes = []
    updates_succ = []
    for file in args.files:
        if file.endswith('csv'):
            r, ur, s, us = load_csv(file)
        elif file.endswith('out'):
            if args.goal:
                r, ur, s, us = load_out(file, goal_env=True)
            else:
                r, ur = load_out(file, goal_env=False)
        else:
            raise Exception("File type not supported ")
        returns.append(r)
        updates_returns.append(ur)
        if args.goal:
            successes.append(s)
            updates_succ.append(us)
    plot_returns(returns, updates_returns)
    if args.goal:
        plot_successes(successes, updates_succ)


if __name__ == "__main__":
    main()
