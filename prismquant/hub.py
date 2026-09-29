"""Download one complete, revision-pinned checkpoint and its rotation factors."""
from __future__ import annotations

import json
from pathlib import Path
from importlib.resources import files
from typing import Any


def list_models() -> dict[str, dict[str, Any]]:
    """Return the released model catalog, including every available checkpoint."""
    return json.loads(files('prismquant').joinpath('models.json').read_text())


def checkpoint_spec(model: str, checkpoint: str | None = None) -> tuple[dict, str, dict]:
    catalog = list_models()
    if model not in catalog:
        raise ValueError(f'Unknown model {model!r}; choose from {", ".join(catalog)}')
    spec = catalog[model]
    name = checkpoint or spec['default_checkpoint']
    if name not in spec['checkpoints']:
        raise ValueError(f'Unknown checkpoint {name!r} for {model}; available: {list(spec["checkpoints"])}')
    return spec, name, spec['checkpoints'][name]


def validate_snapshot(snapshot: str | Path, model: str, checkpoint: str | None = None) -> Path:
    root = Path(snapshot)
    spec, name, variant = checkpoint_spec(model, checkpoint)
    missing = [p for p in variant['required_files'] if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError(f'Incomplete {model}/{name}: {len(missing)} missing files; first: {missing[:3]}')
    config_path = root / 'config.json'
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        entry = config.get('models', {}).get(model, {})
        if config.get('format') != 'prismquant-reference-v1':
            raise ValueError('Unsupported PrismQuant artifact format')
        expected = dict(base_model=spec['base_model'], architecture=spec['architecture'],
                        storage_dtype=spec['compute_dtype'], default_checkpoint=spec['default_checkpoint'])
        if any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError('Hub configuration does not match the model registry')
    metadata = json.loads((root / 'checkpoints' / model / name / 'DONE.json').read_text())
    if metadata['model'] != model or metadata['rotation'] != variant['rotation']:
        raise ValueError('Checkpoint metadata does not match the requested model/rotation')
    if metadata['model_id'] != spec['base_model']:
        raise ValueError('Checkpoint base-model identity does not match the registry')
    return root


def download_checkpoint(model: str, checkpoint: str | None = None, *, cache_dir=None,
                        token=None, local_files_only: bool = False) -> Path:
    """Fetch only this variant's weights and required factors, never the whole family.

    The registry pins an immutable Hub revision. Base model weights/tokenizer are
    fetched separately by ``load_model`` and retain the upstream license.
    """
    from huggingface_hub import snapshot_download
    spec, name, variant = checkpoint_spec(model, checkpoint)
    root = snapshot_download(spec['repo_id'], revision=spec['revision'],
                             allow_patterns=variant['required_files'], cache_dir=cache_dir,
                             token=token, local_files_only=local_files_only, max_workers=4)
    return validate_snapshot(root, model, name)
