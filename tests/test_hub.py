import json
from pathlib import Path
import pytest
from prismquant.hub import checkpoint_spec, validate_snapshot, download_checkpoint


def make_snapshot(root, model):
    spec,name,var=checkpoint_spec(model)
    for relative in var['required_files']:
        p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    (root/'checkpoints'/model/name/'DONE.json').write_text(json.dumps(dict(model=model,rotation=var['rotation'],model_id=spec['base_model'])))
    (root/'config.json').write_text(json.dumps({'format':'prismquant-reference-v1', 'models':{model:dict(base_model=spec['base_model'], architecture=spec['architecture'], storage_dtype=spec['compute_dtype'], default_checkpoint=spec['default_checkpoint'])}}))
    return spec,name,var


def test_missing_factor_is_reported_before_model_loading(tmp_path):
    model='qwen3_0.6b_base';spec,name,var=make_snapshot(tmp_path,model)
    assert validate_snapshot(tmp_path,model)==tmp_path
    (tmp_path/var['required_files'][-1]).unlink()
    with pytest.raises(FileNotFoundError,match='missing files'):validate_snapshot(tmp_path,model)


def test_download_selects_only_one_variant_and_pins_revision(tmp_path,monkeypatch):
    model='qwen3_0.6b_base';spec,name,var=make_snapshot(tmp_path,model);observed={}
    def snapshot(repo,**kwargs):observed.update(repo=repo,**kwargs);return str(tmp_path)
    monkeypatch.setattr('huggingface_hub.snapshot_download',snapshot)
    assert download_checkpoint(model,local_files_only=True)==tmp_path
    assert observed['repo']==spec['repo_id'] and observed['revision']==spec['revision']
    assert observed['allow_patterns']==var['required_files'] and observed['local_files_only']
    assert all('hadamard' not in p for p in observed['allow_patterns'])


def test_wrong_model_metadata_is_rejected(tmp_path):
    model='qwen3_0.6b_base';spec,name,_=make_snapshot(tmp_path,model)
    p=tmp_path/'checkpoints'/model/name/'DONE.json';d=json.loads(p.read_text());d['model']='wrong';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='metadata'):validate_snapshot(tmp_path,model)


def test_wrong_manifest_is_rejected(tmp_path):
    model = 'qwen3_0.6b_base'
    make_snapshot(tmp_path, model)
    path = tmp_path/'config.json'
    config = json.loads(path.read_text())
    config['models'][model]['base_model'] = 'unrelated/model'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='configuration'):
        validate_snapshot(tmp_path, model)
