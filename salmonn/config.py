# qwenvl/inference/config.py
import torch
from typing import Optional, Literal
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    base_path: str = Field(..., description="微调模型路径")
    tokenizer_path: Optional[str] = Field(None, description="基础模型路径（用于tokenizer和processors）")
    lora_path: Optional[str] = Field("No", description="LoRA权重路径")
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    device: str = "cuda:0"
    no_audio: bool = False
    use_liger: bool = True
    
    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }[self.dtype]

class GenerationConfig(BaseModel):
    max_new_tokens: int = Field(1024, ge=1)
    do_sample: bool = False
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    temperature: float = Field(0.7, ge=0.0)
    num_samples: int = Field(1, ge=1)

# qwenvl/inference/config.py
class DataConfig(BaseModel):
    data_path: str  
    dataset_use: str = None  
    model_type: str = "qwen2.5vl"
    max_pixels: int = 512 * 512
    min_pixels: int = 256 * 256
    video_max_frame_pixels: int = 1664 * 28 * 28
    video_min_frame_pixels: int = 256 * 28 * 28
    base_interval: int = 4
    video_min_frames: int = 4
    video_max_frames: int = 8
    
    def __init__(self, **data):

        if "data_path" in data and "dataset_use" not in data:
            data["dataset_use"] = data["data_path"]
        super().__init__(**data)

class OutputConfig(BaseModel):
    output_path: str
    indent: int = 2

class InferenceConfig(BaseModel):
    model: ModelConfig
    generation: GenerationConfig
    data: DataConfig
    output: OutputConfig
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "InferenceConfig":
        import yaml
        with open(path) as f:
            return cls(**yaml.safe_load(f))