# qwenvl/inference/pipeline.py
"""
推理Pipeline - 适配训练脚本输出的特殊字段格式
关键：prompt 是字典，ref 是字符串，input_ids 必须存在
"""
from typing import List, Dict, Any, Optional, Union
import torch
from tqdm import tqdm
from qwenvl.inference.config import InferenceConfig, GenerationConfig
from qwenvl.inference.utils import decode_tokens

class QwenVLPipeline:
    def __init__(
        self,
        model: "video_SALMONN2_plus",
        tokenizer,
        config: Union[InferenceConfig, GenerationConfig],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config.generation if isinstance(config, InferenceConfig) else config
        self.device = next(model.parameters()).device
    
    @torch.no_grad()
    def __call__(
        self,
        dataset: "InferenceDataset",
        progress_bar: bool = True,
    ) -> List[Dict[str, Any]]:
        results = []
        iterable = tqdm(range(len(dataset)), desc="Inference") if progress_bar else range(len(dataset))
        
        for i in iterable:
            sample = dataset[i]
            
            # 关键：验证 input_ids 存在（训练脚本推理模式保证此字段）
            if "input_ids" not in sample:
                print(f"[ERROR] Sample {i} missing input_ids! Keys: {list(sample.keys())}")
                result = {
                    "video": sample.get("video"),
                    "image": sample.get("image"),
                    "audio": sample.get("audio"),
                    "use_audio": sample.get("use_audio", False),
                    "should_use": sample.get("should_use", True),
                    "prompt": sample.get("prompt", {"from": "human", "value": "[MISSING]"}),
                    "ref": sample.get("ref", "[MISSING]"),
                    "pred": "[GENERATION_FAILED: missing input_ids]",
                }
                results.append(result)
                continue
            
            # 准备模型输入（仅保留必要字段）
            model_inputs = {
                "input_ids": sample["input_ids"].to(self.device),
                "attention_mask": sample["attention_mask"].to(self.device),
            }
            
            # 添加多模态字段（如果存在）
            multimodal_fields = [
                "pixel_values", "pixel_values_videos", 
                "audio_feature", "temporal_indices",
                "image_grid_thw", "video_grid_thw"
            ]
            for field in multimodal_fields:
                if field in sample and sample[field] is not None:
                    if isinstance(sample[field], torch.Tensor):
                        model_inputs[field] = sample[field].to(self.device)
                    else:
                        model_inputs[field] = sample[field]
            
            # 生成参数（避免警告）
            gen_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            if self.config.do_sample:
                gen_kwargs.update({
                    "do_sample": True,
                    "top_p": self.config.top_p,
                    "temperature": self.config.temperature,
                })
            else:
                gen_kwargs["do_sample"] = False
            
            try:
                outputs = self.model.generate(**model_inputs, **gen_kwargs)
                
                # 解码（与训练脚本完全一致）
                output_ids = outputs[0][model_inputs["input_ids"].shape[1]:]
                pred_text = decode_tokens(output_ids, self.tokenizer)
                
                # 构建结果（精确匹配训练脚本输出格式）
                result = {
                    "video": sample.get("video"),
                    "image": sample.get("image"),
                    "audio": sample.get("audio"),
                    "use_audio": sample.get("use_audio", False),
                    "should_use": sample.get("should_use", True),
                    "prompt": sample["prompt"],  # 字典: {"from": "human", "value": "..."}
                    "ref": sample["ref"],        # 字符串
                    "pred": pred_text,
                }
                results.append(result)
                
            except Exception as e:
                print(f"[ERROR] Generation failed for sample {i}: {e}")
                result = {
                    "video": sample.get("video"),
                    "image": sample.get("image"),
                    "audio": sample.get("audio"),
                    "use_audio": sample.get("use_audio", False),
                    "should_use": sample.get("should_use", True),
                    "prompt": sample["prompt"],
                    "ref": sample["ref"],
                    "pred": f"[GENERATION_FAILED: {str(e)}]",
                }
                results.append(result)
        
        return results