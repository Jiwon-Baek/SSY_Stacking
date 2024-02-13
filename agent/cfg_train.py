import argparse


def get_cfg():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--load_model", type=int, default=0, help="load the trained model")
    parser.add_argument("--model_path", type=str, default=None, help="model file path")

    parser.add_argument("--num_episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--num_piles", type=int, default=10, help="number of target piles")
    parser.add_argument("--num_plates", type=int, default=20, help="average number of plates")
    parser.add_argument("--max_height", type=int, default=20, help="maximum number of plates to be stacked in each pile")

    parser.add_argument("--log_every", type=int, default=10, help="Record the training results every x episodes")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate every x episodes")
    parser.add_argument("--save_every", type=int, default=1000, help="Save a model every x episodes")
    parser.add_argument("--new_instance_every", type=int, default=10, help="Generate new scenarios every x episodes")

    return parser.parse_args()