import torch
import numpy as np
import argparse
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from transformers import AutoModelForMaskedLM
from src.models.rnaernie.tokenization_rnaernie import RNAErnieTokenizer

def load_model_and_tokenizer(model_name_or_path: str):
    """
    Loads the RNAErnie model and tokenizer from the Hugging Face Hub.
    
    Args:
        model_name_or_path (str): The name of the model on the Hugging Face Hub or path to a local directory.
        
    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    tokenizer = RNAErnieTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    return model, tokenizer

def get_embeddings(sequences: list[str], model, tokenizer):
    """
    Generates embeddings for a list of RNA sequences.
    
    Args:
        sequences (list[str]): A list of RNA sequences.
        model: The pre-trained model.
        tokenizer: The tokenizer.
        
    Returns:
        A tensor containing the mean embeddings for each sequence.
    """
    inputs = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Embeddings are taken from the last hidden state
    embeddings = outputs.hidden_states[-1]
    
    # Average the token embeddings to get a single embedding per sequence
    mean_embeddings = embeddings.mean(dim=1)
    
    return mean_embeddings

def save_embeddings(embeddings: np.ndarray, output_path: str):
    """
    Saves the embeddings to a NumPy file.
    
    Args:
        embeddings (np.ndarray): The embeddings to save.
        output_path (str): The path to the output file.
    """
    np.save(output_path, embeddings)

def parse_fasta(fasta_file: str) -> list[str]:
    """
    Parses a FASTA file and returns a list of sequences.
    
    Args:
        fasta_file (str): The path to the FASTA file.
        
    Returns:
        A list of sequences from the FASTA file.
    """
    sequences = []
    with open(fasta_file, 'r') as f:
        sequence = ""
        for line in f:
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                sequence = ""
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
    return sequences

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for RNA sequences using RNAErnie.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="LLM-EDA/RNAErnie",
        help="The name of the model on the Hugging Face Hub or path to a local directory."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sequences",
        nargs='+',
        help="A list of RNA sequences to embed."
    )
    group.add_argument(
        "--fasta_file",
        type=str,
        help="Path to a FASTA file containing RNA sequences."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="rna_embeddings.npy",
        help="The path to save the embeddings NumPy file."
    )
    args = parser.parse_args()

    if args.fasta_file:
        rna_sequences = parse_fasta(args.fasta_file)
    else:
        rna_sequences = args.sequences

    # 1. Load the model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    print("Model and tokenizer loaded.")
    
    # 2. Get the embeddings
    print(f"Generating embeddings for {len(rna_sequences)} sequences...")
    embeddings = get_embeddings(rna_sequences, model, tokenizer)
    print(f"Embeddings generated with shape: {embeddings.shape}")
    
    # 3. Save the embeddings
    print(f"Saving embeddings to {args.output_file}...")
    save_embeddings(embeddings.numpy(), args.output_file)
    print("Embeddings saved.")

if __name__ == "__main__":
    main()
