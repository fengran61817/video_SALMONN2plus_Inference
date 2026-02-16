# qwenvl/inference/data.py
"""
推理数据集 - 精确复现 LazySupervisedDataset._get_item() 的 run_test=True 行为
"""
from typing import Dict, Any, Optional, List
import torch
from pathlib import Path
from transformers import AutoTokenizer
from qwenvl.data.dataset import LazySupervisedDataset

class InferenceDataset:
    """直接复用训练数据集的推理逻辑"""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data_path: str,
        image_processor,
        audio_processor,
        model_type: str = "qwen2.5vl",
    ):
        # 创建模拟 data_args 启用 run_test 模式
        class FakeDataArgs:
            dataset_use = data_path
            run_test = True
            train_type = "sft"
            model_type = model_type
            max_pixels = 512 * 512
            min_pixels = 256 * 256
            video_max_frame_pixels = 1664 * 28 * 28
            video_min_frame_pixels = 256 * 28 * 28
            base_interval = 4
            video_min_frames = 4
            video_max_frames = 8
            
            @property
            def image_processor(self):
                return image_processor
            
            @property
            def audio_processor(self):
                return audio_processor
        
        self.dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=FakeDataArgs())
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i) -> Dict[str, Any]:
        try:
            # === 关键：打印原始数据 ===
            raw_sample = self.dataset.list_data_dict[i]
            print(f"\n[DEBUG] Processing sample {i}:")
            print(f"  Raw keys: {list(raw_sample.keys())}")
            if "conversations" in raw_sample:
                print(f"  Conversations: {len(raw_sample['conversations'])} turns")
                for j, conv in enumerate(raw_sample["conversations"]):
                    print(f"    [{j}] {conv.get('from', 'N/A')}: {conv.get('value', '')[:50]}...")
            
            sample = self.dataset._get_item(i)
            
            # 验证关键字段
            required_fields = ["input_ids", "prompt", "ref"]
            missing_fields = [f for f in required_fields if f not in sample]
            if missing_fields:
                raise ValueError(f"Missing fields: {missing_fields}")
            
            print(f"  ✅ Success! Keys: {list(sample.keys())}")
            if "input_ids" in sample:
                print(f"  input_ids shape: {sample['input_ids'].shape}")
            return sample
            
        except Exception as e:
            print(f"\n[ERROR] Sample {i} failed with exception:")
            import traceback
            traceback.print_exc()
            
            # 尝试 fallback
            import random
            for _ in range(3):
                fallback_idx = random.randint(0, len(self) - 1)
                try:
                    fallback = self.dataset._get_item(fallback_idx)
                    if all(f in fallback for f in ["input_ids", "prompt", "ref"]):
                        fallback["should_use"] = False
                        print(f"[WARNING] Using fallback sample {fallback_idx}")
                        return fallback
                except:
                    continue
            
            # 最后手段
            print(f"[CRITICAL] Creating minimal sample for {i}")
            return {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
                "prompt": {"from": "human", "value": "[DEBUG_FAILED]"},
                "ref": "[DEBUG_FAILED]",
                "should_use": False,
                "video": raw_sample.get("video") if 'raw_sample' in locals() else None,
                "audio": None,
                "use_audio": raw_sample.get("use_audio", False) if 'raw_sample' in locals() else False
            }