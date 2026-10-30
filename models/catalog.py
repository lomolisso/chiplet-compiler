"""
DNN model set for chiplet-accelerator workload characterization.

Grouped by architecture family (typical DNN taxonomy). Each entry gives what is
needed to fetch the model: the code repo and the exact weights/checkpoint.

status:     "existing" = already in our list, "added" = new for heterogeneity
framework:  framework of the upstream reference (all must be ported to JAX -> StableHLO)
"""

MODELS = [
    # ------------------------------------------------------------------
    # Convolutional Neural Networks (CNN)
    # ------------------------------------------------------------------
    {
        "key": "resnet50",
        "name": "ResNet-50 v1.5",
        "family": "CNN",
        "subfamily": "Image classification",
        "status": "existing",
        "framework": "TensorFlow/PyTorch",
        "repo": "https://github.com/mlcommons/inference (vision/classification_and_detection)",
        "weights": "MLPerf Inference ResNet-50 v1.5 reference checkpoint (Zenodo, linked from repo README)",
    },
    {
        "key": "mobilenetv4_conv_m",
        "name": "MobileNetV4-Conv-Medium",
        "family": "CNN",
        "subfamily": "Efficient image classification (depthwise-separable)",
        "status": "existing",
        "framework": "PyTorch (timm)",
        "repo": "https://github.com/huggingface/pytorch-image-models",
        "weights": "timm: mobilenetv4_conv_medium",
    },
    {
        "key": "yolo11",
        "name": "YOLO11-L",
        "family": "CNN",
        "subfamily": "Single-stage object detection",
        "status": "existing",
        "framework": "PyTorch (Ultralytics)",
        "repo": "https://github.com/ultralytics/ultralytics",
        "weights": "yolo11l.pt",
    },
    {
        "key": "unet3d",
        "name": "3D U-Net (KiTS19)",
        "family": "CNN",
        "subfamily": "3D volumetric segmentation (medical)",
        "status": "existing",
        "framework": "PyTorch",
        "repo": "https://github.com/mlcommons/inference (vision/medical_imaging/3d-unet-kits19)",
        "weights": "Zenodo record 5597155",
    },

    # ------------------------------------------------------------------
    # Hybrid CNN-Transformer
    # ------------------------------------------------------------------
    {
        "key": "mobilenetv4_hybrid_m",
        "name": "MobileNetV4-Hybrid-Medium",
        "family": "Hybrid CNN-Transformer",
        "subfamily": "Efficient image classification (conv + mobile attention)",
        "status": "existing",
        "framework": "PyTorch (timm)",
        "repo": "https://github.com/huggingface/pytorch-image-models",
        "weights": "timm: mobilenetv4_hybrid_medium",
    },

    # ------------------------------------------------------------------
    # Transformers
    # ------------------------------------------------------------------
    {
        "key": "bert",
        "name": "BERT-Large (SQuAD v1.1)",
        "family": "Transformer",
        "subfamily": "Encoder-only",
        "status": "existing",
        "framework": "PyTorch (HF transformers)",
        "repo": "https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
        "weights": "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
    },
    {
        "key": "whisper",
        "name": "Whisper large-v3",
        "family": "Transformer",
        "subfamily": "Encoder-decoder (speech-to-text)",
        "status": "existing",
        "framework": "PyTorch (HF transformers)",
        "repo": "https://huggingface.co/openai/whisper-large-v3",
        "weights": "openai/whisper-large-v3",
    },
    {
        "key": "llama",
        "name": "Llama-3.1-8B-Instruct",
        "family": "Transformer",
        "subfamily": "Decoder-only LLM (dense)",
        "status": "existing",
        "framework": "PyTorch (HF transformers)",
        "repo": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
        "weights": "meta-llama/Llama-3.1-8B-Instruct (gated: accept license on HF)",
    },
    {
        "key": "gemma4",
        "name": "Gemma 4 26B-A4B",
        "family": "Transformer",
        "subfamily": "Decoder-only LLM (Mixture-of-Experts)",
        "status": "existing",
        "framework": "JAX/PyTorch",
        "repo": "https://huggingface.co/google/gemma-4-26b-a4b",
        "weights": "google/gemma-4-26b-a4b",
    },
    {
        "key": "qwen3vl",
        "name": "Qwen3-VL-235B-A22B-Instruct",
        "family": "Transformer",
        "subfamily": "Multimodal vision-language (ViT encoder + MoE decoder)",
        "status": "existing",
        "framework": "PyTorch (HF transformers)",
        "repo": "https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct",
        "weights": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    },
    {
        "key": "rtdetr_v2",
        "name": "RT-DETRv2-R50",
        "family": "Transformer",
        "subfamily": "Detection transformer (deformable attention)",
        "status": "added",
        "framework": "PyTorch (HF transformers)",
        "repo": "https://github.com/lyuwenyu/RT-DETR",
        "weights": "PekingU/rtdetr_v2_r50vd",
    },
    {
        "key": "sam2",
        "name": "SAM 2.1 Hiera-Large",
        "family": "Transformer",
        "subfamily": "Vision foundation model (hierarchical ViT + streaming memory attention)",
        "status": "added",
        "framework": "PyTorch",
        "repo": "https://github.com/facebookresearch/sam2",
        "weights": "facebook/sam2.1-hiera-large",
    },
    {
        "key": "alphafold2",
        "name": "AlphaFold 2 (monomer, model_1_ptm)",
        "family": "Transformer",
        "subfamily": "Evoformer / pair-representation attention (protein structure)",
        "status": "existing",
        "framework": "JAX (Haiku)",
        "repo": "https://github.com/google-deepmind/alphafold",
        "weights": "DeepMind AlphaFold parameters, model_1_ptm (download script in repo)",
    },

    # ------------------------------------------------------------------
    # State-space and long-convolution sequence models
    # ------------------------------------------------------------------
    {
        "key": "mamba2",
        "name": "Mamba-2 2.7B",
        "family": "State-space model",
        "subfamily": "Selective SSM (SSD)",
        "status": "existing",
        "framework": "PyTorch",
        "repo": "https://github.com/state-spaces/mamba",
        "weights": "state-spaces/mamba2-2.7b (HF-format reference: AntonV/mamba2-2.7b-hf)",
    },
    {
        "key": "evo2",
        "name": "Evo 2 7B",
        "family": "Long-convolution hybrid",
        "subfamily": "StripedHyena 2 (FFT long convolutions + attention), DNA language model",
        "status": "added",
        "framework": "PyTorch (Vortex)",
        "repo": "https://github.com/ArcInstitute/evo2",
        "weights": "arcinstitute/evo2_7b (lighter alternative: arcinstitute/evo2_1b_base)",
    },

    # ------------------------------------------------------------------
    # Recurrent Neural Networks (RNN)
    # ------------------------------------------------------------------
    {
        "key": "rnnt",
        "name": "RNN-T (LSTM transducer, LibriSpeech)",
        "family": "RNN",
        "subfamily": "LSTM encoder/predictor + joint network (speech recognition)",
        "status": "added",
        "framework": "PyTorch",
        "repo": "https://github.com/mlcommons/inference (speech_recognition/rnnt; retired benchmark, may live under retired_benchmarks/)",
        "weights": "Zenodo record 3662521: DistributedDataParallel_1576581068.9962234-epoch-100.pt",
    },

    # ------------------------------------------------------------------
    # Graph Neural Networks (GNN)
    # ------------------------------------------------------------------
    {
        "key": "rgat",
        "name": "R-GAT (IGBH)",
        "family": "GNN",
        "subfamily": "Relational graph attention, sampled neighborhoods",
        "status": "existing",
        "framework": "PyTorch (DGL/PyG)",
        "repo": "https://github.com/mlcommons/inference (graph/R-GAT)",
        "weights": "MLPerf Inference RGAT.pt",
    },
    {
        "key": "graphcast",
        "name": "GraphCast (operational, 0.25 deg, 13 levels)",
        "family": "GNN",
        "subfamily": "Encoder-processor-decoder message passing on a static mesh (weather)",
        "status": "added",
        "framework": "JAX (Haiku)",
        "repo": "https://github.com/google-deepmind/graphcast",
        "weights": "GCS bucket dm_graphcast: 'GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz' (HF mirror: shermansiu/dm_graphcast_operational)",
    },

    # ------------------------------------------------------------------
    # Recommendation models
    # ------------------------------------------------------------------
    {
        "key": "dlrm",
        "name": "DLRMv2 (DLRM-DCNv2), Criteo 1TB multi-hot",
        "family": "Recommendation",
        "subfamily": "Embedding tables + MLP + DCNv2 cross network",
        "status": "existing",
        "framework": "PyTorch (TorchRec)",
        "repo": "https://github.com/mlcommons/inference (recommendation/dlrm_v2)",
        "weights": "MLPerf Inference DLRMv2 checkpoint (download instructions in repo)",
    },
    {
        "key": "wide_deep",
        "name": "Wide & Deep (Outbrain)",
        "family": "Recommendation",
        "subfamily": "Embedding tables + linear (wide) + MLP (deep)",
        "status": "existing",
        "framework": "TensorFlow 2",
        "repo": "https://github.com/NVIDIA/DeepLearningExamples (TensorFlow2/Recommendation/WideAndDeep)",
        "weights": "NVIDIA DeepLearningExamples Wide & Deep TF2 checkpoint (NGC)",
    },

    # ------------------------------------------------------------------
    # Diffusion models
    # ------------------------------------------------------------------
    {
        "key": "sdxl",
        "name": "Stable Diffusion XL Base 1.0",
        "family": "Diffusion",
        "subfamily": "Latent diffusion, UNet denoiser",
        "status": "existing",
        "framework": "PyTorch (diffusers)",
        "repo": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        "weights": "stabilityai/stable-diffusion-xl-base-1.0",
    },
    {
        "key": "flux",
        "name": "FLUX.1-schnell",
        "family": "Diffusion",
        "subfamily": "Latent diffusion / rectified flow, transformer (DiT/MMDiT) denoiser",
        "status": "added",
        "framework": "PyTorch (diffusers)",
        "repo": "https://github.com/black-forest-labs/flux",
        "weights": "black-forest-labs/FLUX.1-schnell (Apache-2.0, ungated)",
    },

    # ------------------------------------------------------------------
    # Point-cloud / 3D perception
    # ------------------------------------------------------------------
    {
        "key": "pointpainting",
        "name": "PointPainting (DeepLabV3+ ResNet-50 OS16 + PointPillars), Waymo",
        "family": "Point-cloud / 3D perception",
        "subfamily": "Camera-lidar fusion, pillar-based 3D detection",
        "status": "existing",
        "framework": "PyTorch",
        "repo": "https://github.com/mlcommons/mlperf_automotive",
        "weights": "MLPerf Automotive 3D object detection reference checkpoints",
    },
    {
        "key": "centerpoint",
        "name": "CenterPoint (voxel 0.075, sparse 3D conv backbone), nuScenes",
        "family": "Point-cloud / 3D perception",
        "subfamily": "Voxel-based 3D detection with submanifold sparse convolution",
        "status": "added",
        "framework": "PyTorch (OpenPCDet, spconv)",
        "repo": "https://github.com/open-mmlab/OpenPCDet",
        "weights": "OpenPCDet model zoo, cfg tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml (Waymo weights are not released)",
    },
    {
        "key": "ptv3",
        "name": "Point Transformer V3 (base), nuScenes semantic segmentation",
        "family": "Point-cloud / 3D perception",
        "subfamily": "Serialized (space-filling-curve) windowed point attention",
        "status": "added",
        "framework": "PyTorch (Pointcept)",
        "repo": "https://github.com/Pointcept/Pointcept",
        "weights": "Pointcept/PointTransformerV3, subfolder nuscenes-semseg-pt-v3m1-0-base (Waymo weights are not released)",
    },
]
