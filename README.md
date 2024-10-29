# HELIX: Hidden Embedding Layer Information eXplorer

```ascii
   _    _ _____ _     _______ __  __
  | |  | |  ___| |   |_   _|\ \/ /
  | |__| | |__ | |     | |  \V / 
  |  __  |  __|| |     | |   > <  
  | |  | | |___| |_____| |_ / . \ 
  |_|  |_|\____/\_____/\___/_/ \_\
```

HELIX is a specialized visualization tool designed to explore and analyze protein sequence embeddings across transformer model layers. It generates detailed visualizations showing how different amino acids are represented in the embedding space throughout the model's layers, offering insights into the model's internal representations of protein sequences.

## Features

- **Multi-sequence Analysis**: Process multiple protein sequences in batch
- **Layer-wise Visualization**: Generate detailed PCA projections for each transformer layer
- **Amino Acid Tracking**: Unique markers and colors for each amino acid type
- **PDF Report Generation**: Automated creation of publication-ready figures in A4 format
- **GPU Acceleration**: CUDA support for faster processing
- **Memory Management**: Automatic garbage collection and CUDA cache clearing
- **Progress Tracking**: Built-in progress bars for long-running processes

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM

### Dependencies

```bash
pip install torch
pip install transformers
pip install safetensors
pip install matplotlib
pip install numpy
pip install scikit-learn
pip install huggingface_hub
pip install tqdm
```

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/username/helix.git
cd helix
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python helix.py sequences.txt
```

### Advanced Usage

```bash
python helix.py --max_sequences 5 --output custom_output.pdf seq1.txt seq2.txt
```

### Command Line Arguments

- `file_paths`: One or more files containing protein sequences (one per line)
- `--max_sequences`: Maximum number of sequences to process per file (default: 10)
- `--output`: Custom output PDF filename (default: helix_output.pdf)

### Input File Format

Input files should contain one protein sequence per line using standard amino acid single-letter codes:

```
MAEGEITTFTALTEKFNLPPGNYKKPKLLY
MLRFSSFFGEQKTKKEQVLQHFPQGQGPIV
```

## Technical Details

### Model Architecture

HELIX uses the `ChlorophyllChampion/duality100s-ckpt-30000_pythia70m-arc` model, a GPT-NeoX-based architecture fine-tuned for protein sequence analysis. The model configuration includes:

- Base Architecture: GPT-NeoX
- Tokenizer: ByteT5 (specialized for protein sequences)
- Precision: FP16 (half-precision)

### Visualization Framework

The visualization system employs several sophisticated techniques:

1. **Dimensionality Reduction**:
   - Principal Component Analysis (PCA) for 2D projections
   - Layer-wise independent transformations
   - Preserved variance optimization

2. **Plot Organization**:
   - A4 paper size optimization (8.27 × 11.69 inches)
   - Dynamic grid layout based on sequence count
   - Automated subplot positioning and scaling

3. **Visual Encoding**:
   - Amino acid-specific markers and colors
   - Viridis colormap for optimal perceptual linearity
   - Special highlighting for cysteine (C) and methionine (M)

### Memory Management

The tool implements several memory optimization strategies:

```python
# Memory cleanup after processing each file
gc.collect()
torch.cuda.empty_cache()
```

## Visualization Output

The generated PDF contains:

- One page per input file
- Multiple sequences per page (grid layout)
- For each sequence:
  - Layer-wise PCA projections
  - Amino acid position tracking
  - Consistent color and marker encoding
  - Layer-specific titles and annotations

## Performance Considerations

- GPU memory usage scales with sequence length
- Processing time is primarily dependent on:
  - Number of sequences
  - Sequence length
  - Available GPU memory
  - CPU speed (for PCA calculations)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use HELIX in your research, please cite:

```bibtex
@software{nelson2024helix,
  title={HELIX: Hidden Embedding Layer Information eXplorer},
  author={Nelson, David},
  year={2024},
  url={https://github.com/username/helix}
}
```

## Acknowledgments

- Transformers library by Hugging Face
- PyTorch team for CUDA optimization
- Matplotlib developers for visualization capabilities
- scikit-learn team for PCA implementation

## Contact

David Nelson - [GitHub Profile](https://github.com/username)

Project Link: [https://github.com/username/helix](https://github.com/username/helix)
