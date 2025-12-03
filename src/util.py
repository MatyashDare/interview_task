import argparse
import os
import random
import torch
import matplotlib.pyplot as plt

from typing import Optional, Tuple, List


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small Transformer model on a subset of the WMT17 de-en dataset.")

    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory where model outputs and checkpoints will be saved.")

    parser.add_argument("--model_dim", type=int, default=512,
                        help="Dimensionality of the model's hidden layers.")

    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads in the multi-head attention mechanism.")

    parser.add_argument("--enc_layers", type=int, default=6,
                        help="Number of layers in the encoder.")

    parser.add_argument("--dec_layers", type=int, default=6,
                        help="Number of layers in the decoder.")

    parser.add_argument("--mlp_ratio", type=int, default=4,
                        help="Ratio of the hidden layer size in the feed-forward network compared to model_dim.")

    parser.add_argument("--attn_dropout", type=float, default=0.1,
                        help="Dropout rate applied to the attention weights.")

    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate applied to the model's hidden layers.")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of samples per batch during training.")

    parser.add_argument("--epochs", type=int, default=3,
                        help="Total number of training epochs.")

    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.98),
                        help="Coefficients used for computing running averages of gradient and its square in the optimizer.")

    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Term added to the denominator.")
                    
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for the optimizer.")

    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of steps for learning rate warm-up.")

    parser.add_argument("--max_src_len", type=int, default=128,
                        help="Maximum length of source sequences.")

    parser.add_argument("--max_tgt_len", type=int, default=128,
                        help="Maximum length of target sequences.")

    parser.add_argument("--subset_train", type=int, default=50000,
                        help="Number of samples to use from the training dataset subset.")

    parser.add_argument("--subset_val", type=int, default=5000,
                        help="Number of samples to use from the validation dataset subset.")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    return parser.parse_args()


def plot_metrics(
    train_losses: List[float], val_losses: List[float], bleu_scores: List[float], output_dir: str
    ) -> None:
    """
    Plot and save training/validation losses and BLEU scores per epoch.

    Args:
        train_losses (List[float]): List of training losses per epoch.
        val_losses (List[float]): List of validation losses per epoch.
        bleu_scores (List[float]): List of BLEU scores per epoch.
        output_dir (str): Directory where plots will be saved (should be the 'plots' folder).
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss plot: {loss_plot_path}")

    # BLEU plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, bleu_scores, label="BLEU Score", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("BLEU Score")
    plt.title("Validation BLEU Score per Epoch")
    plt.legend()
    plt.grid(True)
    bleu_plot_path = os.path.join(output_dir, "bleu_plot.png")
    plt.savefig(bleu_plot_path)
    plt.close()
    print(f"Saved BLEU plot: {bleu_plot_path}")
