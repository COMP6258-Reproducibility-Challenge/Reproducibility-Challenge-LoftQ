# File: download_model.py
import argparse
import os
import transformers
import torch # Make sure torch is imported

def parse_args():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model and tokenizer.")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Hugging Face model ID to download (e.g., 'meta-llama/Llama-2-7b-hf').",
    )
    parser.add_argument(
        "--save_directory",
        type=str,
        required=True,
        help="Directory on the cluster filesystem (/home or /scratch) to save the model.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face Hub token for private models.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow trusting remote code for models that require it."
    )
    return parser.parse_args()

def download_model():
    args = parse_args()

    print(f"--- Model Downloader ---")
    print(f"Model ID:          {args.model_id}")
    print(f"Save Directory:    {args.save_directory}")
    print(f"Using Token:       {'Yes' if args.token else 'No'}")
    print(f"Trust Remote Code: {args.trust_remote_code}")
    print("-" * 25)

    # Ensure save directory exists
    print(f"Ensuring save directory exists: {args.save_directory}")
    os.makedirs(args.save_directory, exist_ok=True)

    # --- Download and save tokenizer ---
    print("\nDownloading tokenizer...")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_id,
            token=args.token,
            trust_remote_code=args.trust_remote_code
        )
        tokenizer.save_pretrained(args.save_directory)
        print(f"Tokenizer successfully saved to {args.save_directory}")
    except Exception as e:
        print(f"ERROR downloading/saving tokenizer: {e}")
        return # Stop if tokenizer fails

    # --- Download and save model ---
    print("\nDownloading model (this may take a while)...")
    try:
        # Use low_cpu_mem_usage=True to potentially avoid loading everything into RAM at once on the login node
        # Load with bfloat16 if possible to match quantization script's likely expectation
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_id,
            token=args.token,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=args.trust_remote_code
            # Note: No device_map here, just downloading files.
        )
        model.save_pretrained(args.save_directory)
        print(f"Model successfully saved to {args.save_directory}")
    except Exception as e:
        print(f"ERROR downloading/saving model: {e}")

    print("\n--- Download Complete ---")

if __name__ == "__main__":
    download_model()