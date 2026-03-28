# Multi-Agent Collaboration for Coherent Long-Video Music Synthesis

> **Notice**: This code repository is directly related to the manuscript *"Multi-Agent Collaboration for Coherent Long-Video Music Synthesis"* currently submitted to *The Visual Computer*. If you use this code or data in your research, please cite our paper as indicated below.

## Overview

This repository contains the official implementation of our hierarchical multi-agent framework for generating semantically consistent, temporally aligned, and stylistically coherent music for long videos. Our approach integrates storyboard-based semantic structuring, a dual-path feature fusion mechanism, and a closed-loop self-correction strategy to address the challenges of long-video music synthesis.

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{zhao2026multi,
  title={Multi-Agent Collaboration for Coherent Long-Video Music Synthesis},
  author={Zhao, Yi-Pin},
  journal={The Visual Computer},
  year={2026},
  publisher={Springer}
}
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- NVIDIA GPU with at least 16GB memory (four A100 80GB GPUs recommended for training)
- CUDA 11.7 or higher

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/long-video-music-synthesis.git
cd long-video-music-synthesis

# Create and activate conda environment
conda create -n lvms python=3.10
conda activate lvms

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Dependencies

Key packages are listed in `requirements.txt`, including:

- torch>=2.0.0
- transformers>=4.30.0
- diffusers>=0.21.0
- einops>=0.6.0
- librosa>=0.10.0
- opencv-python>=4.8.0
- scikit-learn>=1.3.0
- tqdm>=4.65.0
- wandb>=0.15.0 (optional, for experiment tracking)

## Data Preparation

### Datasets

Our method is evaluated on the following datasets:

1. **V2M-bench**: Standardized evaluation subset for video-conditioned music generation
2. **LVAS-Bench**: Long-form video benchmark with multi-scene sequences
3. **TikTok**: Short-form user-generated videos for cross-domain generalization
4. **AIST++**: Structured dance videos for motion-aligned audio evaluation

### Download and Preprocessing

```bash
# Download datasets (example for V2M-bench)
python scripts/download_v2m_bench.py --output_dir ./data/v2m_bench

# Preprocess videos: extract frames, compute optical flow, and generate scene boundaries
python scripts/preprocess_video.py \
  --input_dir ./data/raw_videos \
  --output_dir ./data/processed \
  --scene_threshold 0.15 \
  --keyframe_per_scene 16

# Generate structured music scripts using the storyboard module
python scripts/generate_music_script.py \
  --video_dir ./data/processed/videos \
  --output_dir ./data/scripts \
  --model_path pretrained/vlm_checkpoint.pth
```

### Data Structure

```
data/
├── v2m_bench/
│   ├── videos/          # Original video files
│   ├── audio/           # Ground-truth audio files
│   ├── scripts/         # Generated structured music scripts
│   └── splits/          # Train/val/test split definitions
├── lvas_bench/
│   ├── long_videos/     # Long-form video sequences
│   ├── annotations/     # Scene boundaries and semantic labels
│   └── metadata.json    # Dataset metadata
└── pretrained/          # Pretrained model weights
    ├── videomae.pth
    ├── imagebind.pth
    └── vlm_checkpoint.pth
```

## Usage

### Training

```bash
# Single-node multi-GPU training (4x A100)
python train.py \
  --config configs/default.yaml \
  --data_dir ./data/v2m_bench \
  --output_dir ./experiments/exp_001 \
  --gpus 0,1,2,3 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --epochs 100

# Key training options:
#   --use_script: Enable storyboard-based semantic guidance (default: True)
#   --dual_path: Enable dual-path feature fusion (default: True)
#   --enable_refinement: Enable tree-of-thought self-correction (default: True)
#   --lambda_semantic: Weight for semantic consistency loss (default: 1.0)
#   --beta_alignment: Weight for temporal alignment loss (default: 1.0)
```

### Inference

```bash
# Generate music for a single video
python infer.py \
  --video_path ./examples/input_video.mp4 \
  --script_path ./examples/music_script.json \
  --checkpoint_path ./experiments/exp_001/best_model.pth \
  --output_dir ./results \
  --enable_correction True \
  --quality_threshold 0.85 \
  --max_refinement_steps 5

# Batch inference on a dataset
python infer_batch.py \
  --data_config configs/v2m_test.yaml \
  --checkpoint_path ./experiments/exp_001/best_model.pth \
  --output_dir ./results/batch_eval \
  --num_workers 8
```

### Evaluation

```bash
# Evaluate generated audio using standard metrics
python evaluate.py \
  --gen_dir ./results/batch_eval \
  --ref_dir ./data/v2m_bench/audio \
  --video_dir ./data/v2m_bench/videos \
  --metrics fad kl fd density coverage ib desync \
  --output_file ./results/metrics.json

# Available metrics:
#   Audio quality: FAD, KL, FD
#   Distribution consistency: Density, Coverage
#   Semantic alignment: ImageBind Similarity (IB)
#   Temporal synchronization: De-Synchronization Error (DeSync)
```

## Acknowledgements

We thank the developers of VideoMAE, ImageBind, and Hugging Face Transformers for their excellent open-source contributions. This work was supported by Kookmin University.
