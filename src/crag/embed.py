"""Sentence embeddings with ONNX Runtime (no PyTorch needed, runs on any CPU).

Default model: all-MiniLM-L6-v2, int8-quantised ONNX export (Xenova/all-MiniLM-L6-v2,
`onnx/model_quantized.onnx`, ~23 MB). Mean pooling + L2 normalisation, exactly as the
sentence-transformers reference implementation.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
HF_BASE = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main"
FILES = ["tokenizer.json", "onnx/model_quantized.onnx"]
CACHE_DIR = ROOT / ".cache"


def ensure_model(model_dir: Path = MODEL_DIR) -> Path:
    for f in FILES:
        dst = model_dir / f
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{HF_BASE}/{f}", dst)
    return model_dir


class OnnxEmbedder:
    name = "all-MiniLM-L6-v2-int8"

    def __init__(self, model_dir: Path = MODEL_DIR, max_len: int = 256, threads: int = 0):
        ensure_model(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_dir / "onnx" / "model_quantized.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[i : i + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feeds)[0]  # (b, seq, dim)
            m = mask[..., None].astype(np.float32)
            emb = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out.append(emb / np.linalg.norm(emb, axis=1, keepdims=True))
        return np.vstack(out).astype(np.float32)

    def encode_cached(self, texts: list[str], tag: str) -> np.ndarray:
        """Cache document embeddings on disk keyed by content hash."""
        h = hashlib.sha1(("\n".join(texts) + self.name).encode()).hexdigest()[:12]
        path = CACHE_DIR / f"{tag}_{h}.npy"
        if path.exists():
            return np.load(path)
        CACHE_DIR.mkdir(exist_ok=True)
        emb = self.encode(texts)
        np.save(path, emb)
        return emb
