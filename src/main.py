"""
main.py

AUTHOR: Lucas Kabela

PURPOSE: This file defines the driving functions for the expirements/code
        of the project and contains the argparser
"""
import argparse
from train import train


def _parse_args():
    """
    Parses the commandline arguments for running an expirement trail/series
    of trials

    Args:

    Returns:
        args: the parsed arguments in a new namespace
    """

    """
    Add arguments below.  Example format:
        parser.add_argument('-cp', '--continue_training_policy',
            action='store_true', help='A help message'
        )

        parser.add_argument('--q1_checkpoint_filename', type=str,
            default='./q1_checkpoint.pth', help="Name of file to save and load"
        )
    """
    parser = argparse.ArgumentParser(
        description="Arguments for Experience Replay project"
    )
    # Model related arguments here
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument(
        "--no_alpha_tune", dest="alph_tune", default=True, action="store_false"
    )
    parser.add_argument("--eval_freq", type=int, default=10)
    # Training related arguments here
    parser.add_argument("--rand_seed", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("-t", "--time_limit", type=int, default=10_000)
    parser.add_argument("-s", "--steps", type=int, default=1_000_000)
    parser.add_argument("-n", "--num_episodes", type=int, default=1000)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)
    parser.add_argument("-c", "--continue_training", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--start_steps", type=int, default=1000)
    parser.add_argument("--buff_size", type=int, default=100000)

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    train(args)


if __name__ == "__main__":
    main()
