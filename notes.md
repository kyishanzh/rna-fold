# Notes on current SOTA RNA structure prediction models

## RibonanzaNet

A hybrid convolutional-transformer architecture for RNA structure prediction

### Key Components:
1. **Attention Mechanisms**
   - Multi-head attention with scaled dot-product attention
   - Distance-based attention masking
   - Triangular attention patterns for RNA-specific features

2. **Architecture Highlights**
   - Hybrid design combining CNNs and Transformers
   - Specialized activation (Mish) for better gradient flow
   - Generalized Mean Pooling (GeM) for adaptive feature aggregation
   - Pairwise feature processing for structural information

3. **Notable Features**
   - Handles variable length RNA sequences
   - Incorporates both local (CNN) and global (Transformer) context
   - Uses structured dropout for better regularization
   - Supports test-time augmentation with sequence flipping

4. **Technical Innovations**
   - ConvTransformerEncoderLayer: Custom layer combining convolution and self-attention
   - Pairwise dimension processing for capturing RNA base pair interactions
   - Temperature-scaled attention for better numerical stability
   - Layer normalization and residual connections throughout

5. **Data handling**
   - `Dataset.py`: Specialized data processing for RNA sequences
   - `dropout.py`: Custom dropout with structured regularization
   - Efficient batch processing with padding and masking

## trRosettaRNA

## RhoFold+

## AlphaFold3