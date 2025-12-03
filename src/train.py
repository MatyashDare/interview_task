import argparse
import json
import os
import torch
import torch.optim as optim
import wandb

from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from typing import Optional, Tuple, List
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import TranslationCollator, load_wmt17_subset
from model import Transformer, TransformerConfig
from util import get_device, set_seed, parse_args, plot_metrics


IGNORE_INDEX = -100


def main():
    """
    Main training loop for the Transformer-based translation model.
    Handles:
    - Argument parsing
    - Dataset loading
    - Model creation
    - Training & validation
    - Logging (W&B)
    - Saving outputs & visualizations
    """
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    # Create structured output folders
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "model"), exist_ok=True)

    wandb.init(
        project="interview_task",
        name=f"run_{args.seed}",
        config=vars(args)
        )

    # Tokenizers
    src_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    tgt_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    # Datasets
    ds_train = load_wmt17_subset(split="train", size=args.subset_train)
    ds_val = load_wmt17_subset(split="validation", size=args.subset_val)
    
    collator = TranslationCollator(src_tokenizer, tgt_tokenizer,
                                   max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    # Model
    config = TransformerConfig(
        embedding_dimension=args.model_dim,
        num_attention_heads=args.num_heads,
        attention_dropout_p=args.attn_dropout,
        hidden_dropout_p=args.dropout,
        mlp_ratio=args.mlp_ratio,
        encoder_depth=args.enc_layers,
        decoder_depth=args.dec_layers,
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        learn_pos_embed=False
    )
    model = Transformer(config).to(device)
    wandb.watch(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=args.betas, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_loader)*args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    
    train_losses, val_losses, bleu_scores = [], [], []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            src_ids = batch["src_input_ids"].to(device)
            src_mask = batch["src_pad_mask"].to(device)
            tgt_ids = batch["tgt_input_ids"].to(device)
            tgt_mask = batch["tgt_pad_mask"].to(device)
            tgt_outputs = batch["tgt_outputs"].to(device)
    
            logits = model(
                src_ids, tgt_ids, src_attention_mask=src_mask, tgt_attention_mask=tgt_mask)
            B, Lt, V = logits.shape
            loss = loss_fn(logits.view(B * Lt, V), tgt_outputs.view(B * Lt))
    
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    
            running_loss += loss.item()
    
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
    
        # Validation
        val_loss, bleu_score = evaluate(
            model, val_loader, loss_fn, device, tgt_tokenizer)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)
    
        print(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={val_loss:.4f} | BLEU={bleu_score:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "bleu": bleu_score
            })
    
    # Save final model and plots
    model_dir = os.path.join(args.output_dir, "model")
    torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pt"))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)
    src_tokenizer.save_pretrained(model_dir)
    tgt_tokenizer.save_pretrained(model_dir)

    plot_dir = os.path.join(args.output_dir, "plots")
    plot_metrics(train_losses, val_losses, bleu_scores, plot_dir)

    print("Training complete. Model and plots saved.")


def evaluate(model, val_loader, loss_fn, device, tokenizer):
    """
    Run evaluation on the validation set.

    Args:
        model: Transformer model.
        val_loader: Validation dataloader.
        loss_fn: Loss function.
        device: CUDA/CPU device.
        tokenizer: Target language tokenizer.

    Returns:
        Tuple containing:
            avg_loss: Mean validation loss.
            bleu_score: Corpus-level BLEU score.
    """
    model.eval()
    total_loss = 0.0
    references, predictions = [], []

    with torch.no_grad():
        for batch in val_loader:
            src_ids = batch["src_input_ids"].to(device)
            src_mask = batch["src_pad_mask"].to(device)
            tgt_ids = batch["tgt_input_ids"].to(device)
            tgt_mask = batch["tgt_pad_mask"].to(device)
            tgt_outputs = batch["tgt_outputs"].to(device)
    
            logits = model(
                src_ids, tgt_ids, src_attention_mask=src_mask, tgt_attention_mask=tgt_mask)
            B, Lt, V = logits.shape
            loss = loss_fn(logits.view(B * Lt, V), tgt_outputs.view(B * Lt))
            total_loss += loss.item()
    
            # Greedy decode
            pred_ids = logits.argmax(-1).cpu().tolist()
            tgt_ids_list = tgt_outputs.cpu().tolist()
            for pred_seq, tgt_seq in zip(pred_ids, tgt_ids_list):
                pred_tokens = [tokenizer.decode([i], skip_special_tokens=True)
                               for i in pred_seq if i != tokenizer.pad_token_id and i != IGNORE_INDEX]
                tgt_tokens = [tokenizer.decode([i], skip_special_tokens=True)
                              for i in tgt_seq if i != tokenizer.pad_token_id and i != IGNORE_INDEX]
                if pred_tokens and tgt_tokens:
                    predictions.append(pred_tokens)
                    references.append([tgt_tokens])
    
    model.train()
    avg_loss = total_loss / max(1, len(val_loader))
    bleu_score = corpus_bleu(references, predictions)
    return avg_loss, bleu_score



if __name__ == "__main__":
    main()
