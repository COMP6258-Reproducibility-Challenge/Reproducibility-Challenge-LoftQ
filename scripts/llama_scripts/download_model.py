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
        # Use the correct AutoModel class based on the model type/intended task
        model_id_lower = args.model_id.lower()
        model = None
        common_kwargs = {
            "token": args.token,
            "torch_dtype": torch.bfloat16, # Adjust dtype if needed, bfloat16 is often good
            "low_cpu_mem_usage": True,
            "trust_remote_code": args.trust_remote_code
        }

        # --- Determine the correct AutoModel class ---
        if "deberta" in model_id_lower or "roberta" in model_id_lower or "bert" in model_id_lower:
            # Models typically used for Sequence Classification
            print(f"Loading {args.model_id} using AutoModelForSequenceClassification...")
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                args.model_id, **common_kwargs
            )
        elif "llama" in model_id_lower or "mistral" in model_id_lower or "gpt" in model_id_lower or "falcon" in model_id_lower:
             # Models typically used for Causal Language Modeling
             print(f"Loading {args.model_id} using AutoModelForCausalLM...")
             model = transformers.AutoModelForCausalLM.from_pretrained(
                 args.model_id, **common_kwargs
             )
        elif "bart" in model_id_lower or "t5" in model_id_lower:
             # Models typically used for Sequence-to-Sequence tasks
             print(f"Loading {args.model_id} using AutoModelForSeq2SeqLM...")
             model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                 args.model_id, **common_kwargs
             )
        else:
             # As a general fallback for just downloading, AutoModel might work,
             # but it won't load a task-specific head.
             print(f"Warning: Unknown model type pattern for {args.model_id}.")
             print("Attempting generic AutoModel.from_pretrained download.")
             print("This will download config/weights but may not load the full architecture correctly for immediate use.")
             model = transformers.AutoModel.from_pretrained(
                 args.model_id, **common_kwargs
             )

        # --- Save the model ---
        if model:
            model.save_pretrained(args.save_directory)
            print(f"Model successfully saved to {args.save_directory}")
        else:
            # This case should ideally not be reached if AutoModel fallback works
            print(f"ERROR: Could not determine appropriate AutoModel class for {args.model_id}")

    except Exception as e:
        print(f"ERROR downloading/saving model: {e}")

    print("\n--- Download Complete ---")

if __name__ == "__main__":
    download_model()