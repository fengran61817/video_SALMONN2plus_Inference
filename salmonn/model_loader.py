# qwenvl/inference/model_loader.py

from typing import Optional
import torch
from peft import PeftModel
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from qwenvl.inference.liger_utils import apply_liger_kernel_to_qwen2_5_vl  # ✅ 修复导入

def load_inference_model(
    config: "ModelConfig",  # 使用字符串避免循环导入
    tokenizer=None,  # 用于验证tokenizer一致性（可选）
) -> video_SALMONN2_plus:
    
    if config.use_liger:
        apply_liger_kernel_to_qwen2_5_vl()
        print("[Inference] Liger kernels applied")
    
    # 关键：与训练完全相同的加载参数
    model = video_SALMONN2_plus.from_pretrained(
        config.base_path,
        attn_implementation="sdpa",
        torch_dtype=config.torch_dtype,
        device_map="cpu",  # 避免自动多卡初始化
        low_cpu_mem_usage=True,
    )
    
    # LoRA合并 - 与训练脚本完全一致的逻辑
    if config.lora_path and config.lora_path != "No":
        print(f"[Inference] Merging LoRA from {config.lora_path}")
        audio_layers = None
        
        # 与训练相同的audio.layers处理
        if not config.no_audio and hasattr(model.audio, 'layers'):
            audio_layers = model.audio.layers
            del model.audio.layers
        
        model = PeftModel.from_pretrained(
            model, 
            config.lora_path,
            torch_dtype=config.torch_dtype,
            device_map="cpu"
        )
        
        if not config.no_audio and audio_layers is not None:
            model.model.audio.layers = audio_layers
        
        model = model.merge_and_unload()
        print("[Inference] LoRA merged successfully")
    
    # 禁用音频模块（与训练行为一致）
    if config.no_audio and hasattr(model, 'audio'):
        print("[Inference] Audio module disabled")
        del model.audio
    
    # 移至目标设备并设置eval模式
    model = model.to(config.device).eval()
    
    # 验证tokenizer一致性（预防训练/推理不一致）
    if tokenizer is not None:
        _validate_tokenizer_consistency(model, tokenizer, config.base_path)
    
    print(f"[Inference] Model loaded on {config.device} with dtype {config.dtype}")
    return model

def _validate_tokenizer_consistency(model, tokenizer, model_path):
    """验证tokenizer与模型配置一致性 - 预防静默错误"""
    try:
        from transformers import AutoTokenizer
        ref_tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 检查关键属性
        checks = [
            ("vocab_size", tokenizer.vocab_size == ref_tokenizer.vocab_size),
            ("pad_token_id", tokenizer.pad_token_id == ref_tokenizer.pad_token_id),
            ("model_max_length", tokenizer.model_max_length == ref_tokenizer.model_max_length),
        ]
        
        for name, ok in checks:
            if not ok:
                print(f"[WARNING] Tokenizer mismatch in {name}: "
                      f"inference={getattr(tokenizer, name)} vs model={getattr(ref_tokenizer, name)}")
    except Exception as e:
        print(f"[WARNING] Tokenizer validation skipped: {e}")