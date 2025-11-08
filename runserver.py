import math, os, importlib, sys

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

import torch

from typing import List, Optional, Union

_ALL_MODELS_ = {
    "MiniLM-L6-v2": {
        "dim": 384,
        "description": "MiniLM-L6-v2 embedding model",
        "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    }
}

def _resolve(
    name
) -> torch.nn.Module:
    import hashlib, importlib
    def base48(
        n: int, pad: int = 2
    ) -> str:
        digits = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghkjmnpqrstuvwxyz'
        if n == 0:
            return digits[0].rjust(pad, digits[0])
        res = ""
        while n > 0:
            remainder = n % len(digits)
            res = digits[remainder] + res
            n //= len(digits)
        return res.rjust(pad, digits[0])
    def md5(
        s
    ) -> str:
        if sys.version_info >= (3, 9):
            md5_ = hashlib.md5(usedforsecurity=False)
        else:
            md5_ = hashlib.md5()
        md5_.update(s.encode("utf-8"))
        return base48(int.from_bytes(md5_.digest()))
    path = os.path.abspath(name + ".py").lower()
    spec = importlib.util.spec_from_file_location(md5(name), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._model()

for m in _ALL_MODELS_:
    _ALL_MODELS_[m]['model'], _ALL_MODELS_[m]['encoding'] = _resolve(m)

class EmbeddingRequest(BaseModel):
    r""" Request body for embedding generation. """
    model: str
    dim: Optional[int] = 384
    input: Union[str, List[str]] = []
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None

class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]

class EmbeddingResponse(BaseModel):
    model: str
    usage: Optional[int] = None
    object: str = "list"
    data: List[EmbeddingData]
    model_config = {
            "exclude_none": True
        }

app = FastAPI()

def encode(model, text: str) -> List[float]:
    model, encoding = _ALL_MODELS_[model]["model"], _ALL_MODELS_[model]["encoding"]
    encoded = encoding(
        [text.strip()],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        token_embeddings = model(
            input_ids = encoded.data['input_ids'], 
            token_type_ids = encoded.data['token_type_ids'],
            attention_mask = encoded.data['attention_mask']
        )
        def mean_pooling(token_embeddings, attention_mask):
            token_embeddings = token_embeddings.cpu()
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)
        sentence_embeddings = mean_pooling(token_embeddings, encoded['attention_mask'])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.squeeze(0).tolist()

@app.post(
    "/v1/embeddings",
    response_model = EmbeddingResponse,
    response_model_exclude_none = True
)
def embeddings(payload: EmbeddingRequest):
    if payload.model not in _ALL_MODELS_:
        raise HTTPException(status_code=400, detail=f"The specified model '{payload.model}' is not supported.")
    if payload.encoding_format != "float":
        raise HTTPException(status_code=400, detail=f"The specified encoding format '{payload.encoding_format}' is not supported.")
    if payload.dim != _ALL_MODELS_[payload.model]["dim"]:
        raise HTTPException(status_code=400, detail=f"The specified dimension '{payload.dim}' is not supported.")
    if isinstance(payload.input, str):
        inputs = [payload.input]
    else:
        inputs = payload.input
    data: List[EmbeddingData] = []
    for _index, text in enumerate(inputs):
           if not isinstance(text, str):
               raise HTTPException(status_code=400, detail="The specified input format is not supported.")
           embedding = encode(payload.model, text)
           data.append(
               EmbeddingData(
                   index=_index,
                   embedding=embedding
               )
           )
    return EmbeddingResponse(
        data=data,
        model=payload.model,
    )