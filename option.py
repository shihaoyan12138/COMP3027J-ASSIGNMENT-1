import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Basic options
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, -1 for CPU")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--bs", type=int, default=32, help="Batch size")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    parser.add_argument("--alpha", type=float, default=0.5, help="Mix-up alpha")

    # Model options
    parser.add_argument("--teacher_model", type=str, default="resnet34", choices=["resnet34", "efficientnet_b0"])

    parser.add_argument("--teacher_path", type=str, default="./results/resnet34_50.pth")

    parser.add_argument("--adjust_step", type=float, default=0.01, help="Coefficient adjustment step size (b)")

    parser.add_argument("--model", type=str, default="DEiT_tiny", choices=["DEiT_tiny", "DEiT_base"])

    parser.add_argument("--num_classes", type=int, default=12)

    # Dataset path
    parser.add_argument("--train_dir", type=str, default="./plants/train")

    parser.add_argument("--val_dir", type=str, default="./plants/val")

    parser.add_argument("--test_dir", type=str, default="./plants/val")

    # Image transformations
    parser.add_argument("--data_argu", type=bool, default=True)

    parser.add_argument("--norm_mean", type=list, default=[0.328, 0.289, 0.207])

    parser.add_argument("--norm_std", type=list, default=[0.094, 0.097, 0.107])

    # Save options
    parser.add_argument("--save_dir", type=str, default="./results")

    parser.add_argument("--test_model", type=str,  default="./results/model_50.pth")

    parser.add_argument("--save_freq", type=int, default=5, help="Checkpoint save frequency")

    return parser.parse_args()

