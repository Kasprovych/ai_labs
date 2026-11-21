"""Model installation: downloads the local models from Hugging Face Hub and checks that they work.

Every model is a set of files (weights in .safetensors, tokenizer, configuration) stored in the
Hugging Face cache (~/.cache/huggingface/hub). `snapshot_download` fetches only missing files, so
running the setup again is fast and safe.
"""

import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

logger = logging.getLogger(__name__)

# Only files needed for inference; skips e.g. ONNX/OpenVINO exports some repositories also contain
INFERENCE_FILE_PATTERNS = ("*.json", "*.safetensors", "*.model", "*.txt", "*.jinja")


@dataclass(frozen=True)
class InstalledModel:
    """Where a model was installed and how big it is."""
    role: str
    model_id: str
    local_path: Path
    size_gb: float
    weight_files: int
    seconds: float


class ModelInstaller:
    def install(self, role: str, model_id: str) -> InstalledModel:
        from huggingface_hub import snapshot_download  # heavy import, kept local

        started = time.perf_counter()
        local_path = Path(snapshot_download(model_id, allow_patterns=list(INFERENCE_FILE_PATTERNS)))
        files = [path for path in local_path.rglob("*") if path.is_file()]
        # Files in the snapshot are symlinks into the shared blob store; stat() follows the link
        size_gb = sum(path.stat().st_size for path in files) / 1024 ** 3
        installed = InstalledModel(role=role, model_id=model_id, local_path=local_path, size_gb=size_gb,
                                   weight_files=sum(path.suffix == ".safetensors" for path in files),
                                   seconds=time.perf_counter() - started)
        logger.info("%-11s %-42s %5.2f GB  %s", role, model_id, size_gb, local_path)
        return installed


def describe_hardware() -> List[str]:
    """Facts about the machine that matter for running the models locally."""
    import mlx.core as mx  # heavy import, kept local

    info = mx.device_info() if hasattr(mx, "device_info") else {}
    memory_gb = info.get("memory_size", 0) / 1024 ** 3
    return [f"Platform: {platform.platform()}", f"Processor: {platform.processor() or platform.machine()}",
            f"MLX {mx.__version__}, default device: {mx.default_device()}",
            f"Unified memory available to MLX: {memory_gb:.0f} GB" if memory_gb else "Unified memory: unknown"]


def smoke_test_models(embedding_model: str, llm_model: str, prompts: Sequence[str]) -> List[str]:
    """Loads the embedding model and the LLM once and runs a tiny request through each."""
    from src.rag.embeddings import E5Embedder
    from src.rag.llm import ChatMessage, MlxLanguageModel, ROLE_USER

    lines = []
    started = time.perf_counter()
    embedder = E5Embedder(embedding_model)
    vector = embedder.embed_query(prompts[0])
    lines.append(f"Embedding model loaded in {time.perf_counter() - started:.1f} s; "
                 f"query vector: {vector.shape[0]} numbers, L2 norm {float((vector ** 2).sum() ** 0.5):.3f}")
    started = time.perf_counter()
    llm = MlxLanguageModel(llm_model, temperature=0.0)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reply = llm.generate([ChatMessage(ROLE_USER, prompts[1])], max_tokens=40)
    lines.append(f"LLM loaded in {load_seconds:.1f} s; test reply in {time.perf_counter() - started:.1f} s: {reply}")
    return lines
