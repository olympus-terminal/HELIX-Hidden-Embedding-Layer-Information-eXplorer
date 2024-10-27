"""
HELIX: Hidden Embedding Layer Information eXplorer
------------------------------------------------
A tool for visualizing protein sequence embeddings across transformer layers.

   _    _ _____ _     _______ __  __
  | |  | |  ___| |   |_   _|\\ \\/ /
  | |__| | |__ | |     | |  \\V / 
  |  __  |  __|| |     | |   > <  
  | |  | | |___| |_____| |_ / . \\ 
  |_|  |_|\\____/\\_____/\\___/_/ \\_\\

Author: [Your Name]
Date: October 2024
"""

import torch
import gc
import matplotlib.pyplot as plt
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.markers as mmarkers
import matplotlib as mpl
from tqdm import tqdm
import sys

def setup_visualization_params():
    """Initialize all matplotlib parameters for consistent styling"""
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 6,
        'axes.linewidth': 0.25,
        'grid.linewidth': 0.25,
        'lines.linewidth': 0.25,
        'xtick.major.width': 0.25,
        'ytick.major.width': 0.25,
        'axes.titlesize': 6,
        'legend.fontsize': 6,
        'figure.titlesize': 6,
        'legend.handlelength': 1,
        'legend.handleheight': 1,
        'legend.frameon': False
    })

def setup_model():
    """Initialize and return the model, tokenizer, and device"""
    print("Loading model and tokenizer...")
    model_name = "ChlorophyllChampion/duality100s-ckpt-30000_pythia70m-arc"
    
    # Load configuration and tokenizer
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("hmbyt5/byt5-small-english")
    
    # Load model weights
    safetensors_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    model = GPTNeoXForCausalLM(config)
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict, strict=False)
    
    # Setup device and move model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device).half()
    
    return model, tokenizer, device

def read_sequences_from_file(file_path, max_sequences=10):
    """Read and return up to max_sequences from file"""
    with open(file_path, 'r') as file:
        sequences = [line.strip() for line in file if line.strip()]
    print(f"Found {len(sequences)} sequences in {file_path}")
    return sequences[:max_sequences]

def aa_to_input_ids(sequence, tokenizer, device):
    """Convert amino acid sequence to input IDs"""
    toks = tokenizer.encode(sequence, add_special_tokens=False)
    return torch.as_tensor(toks).view(1, -1).to(device)

def analyze_sequence(model, input_ids):
    """Get hidden states from model for sequence"""
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return [h.squeeze(0).cpu().numpy() for h in outputs.hidden_states]

def setup_a4_figure(n_sequences):
    """Create figure with A4 dimensions and calculate grid layout"""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches
    
    if n_sequences <= 5:
        rows, cols = n_sequences, 1
    else:
        rows = (n_sequences + 1) // 2
        cols = 2
    
    return fig, rows, cols

def plot_sequence_layers(hidden_states, sequence, ax, sequence_name=""):
    """Plot layer-wise representations for a sequence"""
    # Setup amino acids and markers
    all_aa = 'ACDEFGHIKLMNPQRSTVWYX'
    unique_aa = sorted(set(all_aa + ''.join(set(sequence) - set(all_aa))))
    
    # Expanded marker list
    base_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '8', '*', 'H', '+', 'x', 'd',
                   '1', '2', '3', '4', '_', '|', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    markers = (base_markers * (len(unique_aa) // len(base_markers) + 1))[:len(unique_aa)]
    
    # Setup colors and markers
    n_colors = len(unique_aa)
    colors = plt.cm.viridis(np.linspace(0, 1, n_colors))
    aa_to_color = dict(zip(unique_aa, colors))
    aa_to_marker = dict(zip(unique_aa, markers))
    
    # Special colors for C and M from viridis
    viridis = plt.cm.viridis
    aa_to_color['C'] = viridis(0.9)
    aa_to_color['M'] = viridis(0.1)
    
    # Setup PCA and calculate layout
    pca = PCA(n_components=2)
    n_layers = len(hidden_states)
    n_rows = (n_layers + 3) // 4
    n_cols = min(4, n_layers)
    
    # Plot each layer
    for i, state in enumerate(hidden_states):
        reduced_state = pca.fit_transform(state)
        layer_ax = ax.inset_axes([
            (i % n_cols) / n_cols, 
            1 - ((i // n_cols + 1) / n_rows), 
            1/n_cols * 0.9, 
            1/n_rows * 0.9
        ])
        
        # Plot each amino acid
        for aa in unique_aa:
            mask = np.array([s == aa for s in sequence])
            if np.any(mask):
                layer_ax.scatter(
                    reduced_state[mask, 0], 
                    reduced_state[mask, 1],
                    c=[aa_to_color[aa]], 
                    marker=aa_to_marker[aa],
                    s=20,
                    label=aa if i == 0 else "",
                    alpha=0.7,
                    linewidth=0.25
                )
        
        # Style layer plot
        layer_ax.set_title(f'Layer {i}', pad=2)
        layer_ax.tick_params(axis='both', which='major', labelsize=6)
        layer_ax.set_xticks([])
        layer_ax.set_yticks([])
        
        for spine in layer_ax.spines.values():
            spine.set_linewidth(0.25)
    
    # Style main plot
    ax.set_title(sequence_name, pad=10)
    
    # Add legend to first sequence only
    if sequence_name.endswith('_0'):
        legend = ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            markerscale=0.8,
            handletextpad=0.5,
            borderaxespad=0
        )
        # Use correct method for legend handles
        for handle in legend.get_lines():
            handle.set_linewidth(0.25)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def process_sequences(file_paths, max_sequences=10, output_file='helix_output.pdf'):
    """Main function to process sequences and create visualizations"""
    # Setup
    setup_visualization_params()
    model, tokenizer, device = setup_model()
    
    # Process each file
    with PdfPages(output_file) as pdf:
        for file_path in file_paths:
            print(f"\nProcessing file: {file_path}")
            sequences = read_sequences_from_file(file_path, max_sequences)
            file_name = file_path.split('/')[-1].split('.')[0]
            
            fig, rows, cols = setup_a4_figure(len(sequences))
            plt.suptitle(f'Sequence Representations - {file_name}', y=0.98)
            
            # Process sequences with progress bar
            for idx, sequence in enumerate(tqdm(sequences, desc="Processing sequences")):
                ax = plt.subplot(rows, cols, idx + 1)
                input_ids = aa_to_input_ids(sequence, tokenizer, device)
                hidden_states = analyze_sequence(model, input_ids)
                plot_sequence_layers(
                    hidden_states, 
                    sequence, 
                    ax, 
                    f'{file_name}_seq_{idx}'
                )
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig, dpi=300)
            plt.close()
            
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"Completed processing {len(sequences)} sequences")

def main():
    """Entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="HELIX: Hidden Embedding Layer Information eXplorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python helix.py sequences.txt
  python helix.py --max_sequences 5 --output custom_output.pdf seq1.txt seq2.txt
        """
    )
    parser.add_argument('file_paths', nargs='+', 
                       help='Paths to files containing amino acid sequences (one per line)')
    parser.add_argument('--max_sequences', type=int, default=10, 
                       help='Maximum number of sequences to process per file')
    parser.add_argument('--output', type=str, default='helix_output.pdf',
                       help='Output PDF file name (default: helix_output.pdf)')
    args = parser.parse_args()
    
    print("""
   _    _ _____ _     _______ __  __
  | |  | |  ___| |   |_   _|\\ \\/ /
  | |__| | |__ | |     | |  \\V / 
  |  __  |  __|| |     | |   > <  
  | |  | | |___| |_____| |_ / . \\ 
  |_|  |_|\\____/\\_____/\\___/_/ \\_\\
    """)
    print("HELIX: Hidden Embedding Layer Information eXplorer")
    print("------------------------------------------------")
    
    process_sequences(args.file_paths, args.max_sequences, args.output)
    print(f"\nVisualization completed! Output saved to: {args.output}")

if __name__ == "__main__":
    main()
