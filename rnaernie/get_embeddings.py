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

def save_embeddings(embeddings: np.ndarray, identifiers: list[str], output_path_base: str):
    """
    Saves the embeddings and their identifiers to a NumPy NPZ file and a TSV file.
    
    Args:
        embeddings (np.ndarray): The embeddings to save.
        identifiers (list[str]): The list of identifiers for the embeddings.
        output_path_base (str): The base path for the output files (without extension).
    """
    npz_path = f"{output_path_base}.npz"
    tsv_path = f"{output_path_base}.tsv"
    
    # Save as NPZ
    np.savez(npz_path, embeddings=embeddings, identifiers=identifiers)
    print(f"Embeddings saved to {npz_path}")

    # Save as TSV
    with open(tsv_path, 'w') as f:
        for identifier, embedding in zip(identifiers, embeddings):
            embedding_str = "\t".join(map(str, embedding))
            f.write(f"{identifier}\t{embedding_str}\n")
    print(f"Embeddings saved to {tsv_path}")

def parse_fasta(fasta_file: str) -> tuple[list[str], list[str]]:
    """
    Parses a FASTA file and returns a list of identifiers and a list of sequences.
    
    Args:
        fasta_file (str): The path to the FASTA file.
        
    Returns:
        A tuple containing a list of identifiers and a list of sequences from the FASTA file.
    """
    identifiers = []
    sequences = []
    with open(fasta_file, 'r') as f:
        sequence = ""
        identifier = ""
        for line in f:
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    identifiers.append(identifier)
                sequence = ""
                identifier = line[1:].strip().split()[0]
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
            identifiers.append(identifier)
    return identifiers, sequences

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
        "--output_path_base",
        type=str,
        default="rna_embeddings",
        help="The base path to save the output files (e.g., 'rna_embeddings' will create 'rna_embeddings.npz' and 'rna_embeddings.tsv')."
    )
    args = parser.parse_args()

    if args.fasta_file:
        identifiers, rna_sequences = parse_fasta(args.fasta_file)
    else:
        rna_sequences = args.sequences
        identifiers = [f"seq_{i}" for i in range(len(rna_sequences))]

    # 1. Load the model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    print("Model and tokenizer loaded.")
    
    # 2. Get the embeddings
    print(f"Generating embeddings for {len(rna_sequences)} sequences...")
    embeddings = get_embeddings(rna_sequences, model, tokenizer)
    print(f"Embeddings generated with shape: {embeddings.shape}")
    
    # 3. Save the embeddings
    print(f"Saving embeddings with base path {args.output_path_base}...")
    save_embeddings(embeddings.numpy(), identifiers, args.output_path_base)
    print("Embeddings saved.")

if __name__ == "__main__":
    main()
