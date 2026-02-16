
from typing import List, Union
from transformers import PreTrainedTokenizer

def decode_tokens(
    token_ids: Union[List[int], "torch.Tensor"],
    tokenizer: PreTrainedTokenizer,
) -> str:
    
    if hasattr(token_ids, "cpu"):
        token_ids = token_ids.cpu()
    
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False 
    )

def load_samples(
    path: str,
    format: str = "qwenvl",
) -> List[dict[str, any]]:
    
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    
    if format == "qwenvl":

        return data
    elif format == "custom":

        return _convert_custom_format(data)
    else:
        raise ValueError(f"Unsupported format: {format}")

def _convert_custom_format(data: any) -> List[dict[str, any]]:

    raise NotImplementedError("Custom format conversion not implemented")

def save_results(
    results: List[dict[str, any]],
    path: str,
    indent: int = 2,
    ensure_ascii: bool = False,
):

    import json
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=indent, ensure_ascii=ensure_ascii)
    
    print(f"Results saved to {path} ({len(results)} samples)")