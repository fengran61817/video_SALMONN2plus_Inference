#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import os
from pathlib import Path
from types import SimpleNamespace  

os.environ["DEEPSPEED_MII_INSTALL"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from qwenvl.inference.config import InferenceConfig 
from qwenvl.inference.model_loader import load_inference_model
from qwenvl.inference.pipeline import QwenVLPipeline
from qwenvl.inference.utils import save_results
from transformers import AutoTokenizer
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from transformers import WhisperFeatureExtractor
from qwenvl.data.dataset import LazySupervisedDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 推理")
    parser.add_argument("--config", type=str, default="scripts/default.yaml",
                   help="YAML配置文件路径")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    

    config = InferenceConfig.from_yaml(args.config)
    
    model_path = args.model_path or config.model.base_path
    tokenizer_path = args.tokenizer_path or config.model.tokenizer_path
    data_path = args.data_path or config.data.data_path
    output_path = args.output_path or config.output.output_path
    max_new_tokens = args.max_new_tokens or config.generation.max_new_tokens
    do_sample = args.do_sample or config.generation.do_sample
    
    for path_attr, path_val in [("model", model_path), ("data", data_path)]:
        if not Path(path_val).exists():
            print(f"❌ {path_attr} not found: {path_val}")
            sys.exit(1)
    
    model_base_path = tokenizer_path or model_path
    print(f"Loading components from: {model_base_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_base_path,
        padding_side="right",
        use_fast=False,
    )
    
    image_processor = Qwen2VLImageProcessorFast.from_pretrained(model_base_path)
    audio_processor = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
    )
    
    from qwenvl.inference.config import ModelConfig
    model_config = ModelConfig(
        base_path=model_path,
        tokenizer_path=tokenizer_path,
        lora_path=config.model.lora_path,
        dtype=config.model.dtype,
        device=config.model.device,
        no_audio=config.model.no_audio,
        use_liger=config.model.use_liger
    )
    model = load_inference_model(model_config, tokenizer)
    
    data_args = SimpleNamespace(
        dataset_use=data_path,
        model_type=config.data.model_type,
        max_pixels=config.data.max_pixels,
        min_pixels=config.data.min_pixels,
        video_max_frame_pixels=config.data.video_max_frame_pixels,
        video_min_frame_pixels=config.data.video_min_frame_pixels,
        base_interval=config.data.base_interval,
        video_min_frames=config.data.video_min_frames,
        video_max_frames=config.data.video_max_frames,
        image_processor=image_processor,
        audio_processor=audio_processor,
        run_test=True,
        train_type="sft"
    )
    
    print(f"Creating dataset from {data_path}...")
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    print(f" Dataset created with {len(dataset)} samples")
    
    from qwenvl.inference.config import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=config.generation.top_p,
        temperature=config.generation.temperature,
        num_samples=config.generation.num_samples
    )
    
    pipeline = QwenVLPipeline(model, tokenizer, gen_config)
    results = pipeline(dataset, progress_bar=not args.no_progress)
    
    save_results(results, output_path)
    
    _generate_report(results)

def _generate_report(results: list):
    total = len(results)
    usable = sum(1 for r in results if r.get("should_use", True))
    videos = sum(1 for r in results if r.get("video"))
    audio_enabled = sum(1 for r in results if r.get("use_audio"))
    
    print("\nInference Report")
    print(f"   Total samples: {total}")
    print(f"   Usable samples: {usable}")
    print(f"   Video samples: {videos}")
    print(f"   Audio-enabled: {audio_enabled}")
    print(f"\nInference completed successfully")

if __name__ == "__main__":
    main()
