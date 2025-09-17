import argparse
import logging
from pathlib import Path
import torch
import json
import pandas as pd
from tqdm import tqdm
import math

# Use the correct model definition
from llama_3.model_text_only import Transformer, ModelArgs
from llama_3.tokenizer import Tokenizer
from sae import load_sae_model

# Suppress tokenizer logs
logging.getLogger("saint.llama_3.tokenizer").setLevel(logging.WARNING)

@torch.no_grad()
def calculate_perplexity(model: Transformer, tokenizer: Tokenizer, texts: list[str], device: torch.device) -> float:
    """
    Calculates the perplexity of a model on a given list of texts using a memory-efficient,
    token-by-token approach to avoid OOM errors.
    """
    model.eval()
    total_neg_log_likelihood = 0.0
    total_tokens = 0
    loss_fct = torch.nn.CrossEntropyLoss()

    for text in tqdm(texts, desc="Calculating Perplexity"):
        tokens = tokenizer.encode(text, bos=True, eos=True)
        if len(tokens) < 2:
            continue

        # Process token by token to save memory
        for i in range(1, len(tokens)):
            start_pos = i - 1
            cur_token_tensor = torch.tensor([[tokens[start_pos]]], device=device, dtype=torch.long)
            next_token_tensor = torch.tensor([tokens[i]], device=device, dtype=torch.long)

            # Get logits for the current token to predict the next one
            logits = model.forward(cur_token_tensor, start_pos=start_pos)
            # Logits shape: (1, 1, vocab_size), we need (1, vocab_size) for CrossEntropyLoss
            logits_for_loss = logits.squeeze(0)

            # Calculate loss for the single next token
            loss = loss_fct(logits_for_loss, next_token_tensor)
            total_neg_log_likelihood += loss.item()

        total_tokens += len(tokens) - 1

        # Reset KV cache for every layer for the next sentence
        # Reset KV cache for every layer for the next sentence.
        # In-place modification of inference tensors is not allowed, so we re-initialize the cache.
        for layer in model.layers:
            layer.attention.cache_k = torch.zeros_like(layer.attention.cache_k)
            layer.attention.cache_v = torch.zeros_like(layer.attention.cache_v)


    if total_tokens == 0:
        return float('inf')

    avg_neg_log_likelihood = total_neg_log_likelihood / total_tokens
    perplexity = math.exp(avg_neg_log_likelihood)
    return perplexity

def run_experiment(
    sae_model_path: Path,
    llama_model_dir: Path,
    sae_layer_idx: int,
    ablation_indices: list[int] | None,
    target_texts: list[str],
    control_texts: list[str],
    device: torch.device,
):
    """Loads models for a specific setting (ablation or not) and computes perplexity."""

    # --- Load SAE Model ---
    logging.info(f"Loading SAE model from {sae_model_path}...")
    sae_model = load_sae_model(
        model_path=sae_model_path,
        sae_top_k=64,
        sae_normalization_eps=1e-6,
        device=device,
        dtype=torch.float32,
    )

    # --- Set up Ablation ---
    if ablation_indices:
        logging.info(f"\n--- Running WITH Feature Ablation (Features: {ablation_indices}) ---")
        sae_model.set_ablation_feature_indices(ablation_indices)
        sae_layer_forward_fn = {sae_layer_idx: sae_model.forward}
    else:
        logging.info("\n--- Running WITHOUT Feature Ablation (Baseline) ---")
        # For baseline, we still need to pass the SAE to reconstruct, but without ablation
        sae_model.set_ablation_feature_indices(None)
        sae_layer_forward_fn = {sae_layer_idx: sae_model.forward}

    # --- Load Llama 3 Model with SAE Hook ---
    params_path = llama_model_dir / "params.json"
    tokenizer_path = llama_model_dir / "tokenizer.model"
    model_path = llama_model_dir / "consolidated.00.pth"

    tokenizer = Tokenizer(str(tokenizer_path))

    logging.info(f"Loading model parameters from {params_path}...")
    with params_path.open("r") as f:
        model_params = json.load(f)
    model_args = ModelArgs(**model_params)
    model_args.vocab_size = tokenizer.n_words

    # Set default dtype to bfloat16 for memory efficiency, mirroring llama_3_inference.py
    torch.set_default_dtype(torch.bfloat16)

    # Pass the SAE forward function during initialization
    model = Transformer(model_args, sae_layer_forward_fn=sae_layer_forward_fn)

    # It's good practice to set it back after model initialization
    torch.set_default_dtype(torch.float32)

    state_dict = torch.load(model_path, weights_only=True, map_location="cpu", mmap=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    logging.info(f"Llama 3 model loaded with SAE hooked at layer {sae_layer_idx}.")

    # --- Run Calculations ---
    logging.info(f"\n[Target Dataset Analysis]")
    target_ppl = calculate_perplexity(model, tokenizer, target_texts, device)
    logging.info(f"Perplexity on Target Dataset: {target_ppl:.4f}")

    logging.info(f"\n[Control Dataset Analysis]")
    control_ppl = calculate_perplexity(model, tokenizer, control_texts, device)
    logging.info(f"Perplexity on Control Dataset: {control_ppl:.4f}")

    # Cleanup to free memory
    del model
    del sae_model
    torch.cuda.empty_cache()

    return target_ppl, control_ppl

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run feature ablation experiments on a trained SAE.")
    parser.add_argument("--llama_model_dir", type=Path, required=True, help="Directory containing the Llama 3 model files.")
    parser.add_argument("--sae_model_path", type=Path, required=True, help="Path to the trained SAE model (.pt file).")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Directory containing the ablation datasets (JSONL files).")
    parser.add_argument("--sae_layer_idx", type=int, required=True, help="The layer index where the SAE was trained and should be hooked.")
    parser.add_argument("--ablation_feature_indices", type=int, nargs='+', required=True, help="List of feature indices to ablate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda', 'cpu').")

    return parser.parse_args()

def main():
    """Main function to orchestrate the experiment."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_arguments()

    device = torch.device(args.device)
    if not torch.cuda.is_available() and device.type == 'cuda':
        logging.warning("CUDA is not available, falling back to CPU.")
        device = torch.device('cpu')

    # Load datasets
    logging.info(f"Loading datasets from {args.dataset_dir}...")
    target_df = pd.read_json(args.dataset_dir / "target_dataset.jsonl", lines=True)
    control_df = pd.read_json(args.dataset_dir / "control_dataset.jsonl", lines=True)
    target_texts = target_df['text'].tolist()
    control_texts = control_df['text'].tolist()
    logging.info(f"Loaded {len(target_texts)} target samples and {len(control_texts)} control samples.")

    # --- Run Baseline Experiment (No Ablation) ---
    baseline_target_ppl, baseline_control_ppl = run_experiment(
        sae_model_path=args.sae_model_path,
        llama_model_dir=args.llama_model_dir,
        sae_layer_idx=args.sae_layer_idx,
        ablation_indices=None,
        target_texts=target_texts,
        control_texts=control_texts,
        device=device,
    )

    # --- Run Ablation Experiment ---
    ablated_target_ppl, ablated_control_ppl = run_experiment(
        sae_model_path=args.sae_model_path,
        llama_model_dir=args.llama_model_dir,
        sae_layer_idx=args.sae_layer_idx,
        ablation_indices=args.ablation_feature_indices,
        target_texts=target_texts,
        control_texts=control_texts,
        device=device,
    )

    # --- Print Final Summary ---
    target_change = ((ablated_target_ppl - baseline_target_ppl) / baseline_target_ppl) * 100 if baseline_target_ppl != 0 else float('inf')
    control_change = ((ablated_control_ppl - baseline_control_ppl) / baseline_control_ppl) * 100 if baseline_control_ppl != 0 else float('inf')

    logging.info("\n\n" + "="*50)
    logging.info("          Feature Ablation Experiment Summary")
    logging.info("="*50 + "\n")
    logging.info(f"Ablated Feature(s): {args.ablation_feature_indices}")
    logging.info(f"SAE Model: {args.sae_model_path.name}")
    logging.info(f"SAE Layer: {args.sae_layer_idx}")
    logging.info("-" * 50)
    logging.info(f"Target Dataset Perplexity:")
    logging.info(f"  - Baseline: {baseline_target_ppl:.4f}")
    logging.info(f"  - Ablated:  {ablated_target_ppl:.4f}")
    logging.info(f"  - Change:   {target_change:+.2f}%")
    logging.info("-" * 50)
    logging.info(f"Control Dataset Perplexity:")
    logging.info(f"  - Baseline: {baseline_control_ppl:.4f}")
    logging.info(f"  - Ablated:  {ablated_control_ppl:.4f}")
    logging.info(f"  - Change:   {control_change:+.2f}%")
    logging.info("="*50)

if __name__ == "__main__":
    main()