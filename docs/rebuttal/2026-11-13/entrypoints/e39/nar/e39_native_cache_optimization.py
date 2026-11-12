"""Auditable removal of one unused allocation in pinned FlatQuant deployment.

No official file is edited. The opt-in adapter removes only the random matrix
assignment immediately overwritten by the trained K transform. Every other AST
node is preserved. Its one observable side effect is to stop advancing the RNG
for that unused draw; use only with the deterministic evaluation model.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import inspect
from pathlib import Path
import textwrap


OFFICIAL_SOURCE = Path('/home/yanlongc/.prismquant/FlatQuant-native-torch26-cu124/deploy/transformers/kv_cache.py')
OFFICIAL_SOURCE_SHA256 = 'f425438b9ec8b15cd5157772b3356605b87b7f81eef75e4e8b6ed49d62d63da8'
MODES = ('official', 'remove-dead-random')
_INSTALLED = {}
_DEAD = ('self.head_dim = torch.randn([self.org_head_dim, self.org_head_dim], '
         'requires_grad=False).to(self.trans_dtype).to(self.device)')
_OVERWRITE = 'self.head_dim = self.trans_matrix_k'


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _ast(node):
    return ast.dump(node, include_attributes=False)


def _remove_one_dead_assignment(source):
    """Return source with exactly the pinned adjacent dead assignment removed."""
    original = ast.parse(textwrap.dedent(source))
    if len(original.body) != 1 or not isinstance(original.body[0], ast.FunctionDef):
        raise ValueError('Expected precisely one ordinary update method')
    changed = copy.deepcopy(original)
    dead, overwrite = (_ast(ast.parse(line).body[0]) for line in (_DEAD, _OVERWRITE))
    matches = []
    for node in ast.walk(changed):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for i, statement in enumerate(value):
                if isinstance(statement, ast.stmt) and _ast(statement) == dead:
                    if i + 1 >= len(value) or _ast(value[i + 1]) != overwrite:
                        raise ValueError('Random matrix is no longer immediately overwritten')
                    matches.append((value, i))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one pinned dead assignment, got {len(matches)}')
    statements, index = matches[0]
    removed = statements.pop(index)
    transformed_ast = _ast(changed)
    # Reinserting the exact removed node must reconstruct the original AST.
    statements.insert(index, removed)
    if _ast(changed) != _ast(original):
        raise AssertionError('Unexpected change beyond the single dead assignment')
    statements.pop(index)
    transformed = ast.unparse(changed) + '\n'
    return transformed, dict(
        original_function_sha256=_digest(textwrap.dedent(source)),
        transformed_function_sha256=_digest(transformed),
        original_function_ast_sha256=_digest(_ast(original)),
        transformed_function_ast_sha256=_digest(transformed_ast),
        removed_statement=_DEAD, following_overwrite=_OVERWRITE,
        removed_statement_count=1, all_remaining_ast_nodes_unchanged=True)


def _compile_optimized_update(original):
    if original.__closure__:
        raise ValueError('The pinned method must not capture a closure')
    source, audit = _remove_one_dead_assignment(inspect.getsource(original))
    namespace = {}
    filename = str(Path(__file__).resolve()) + '::generated_update'
    exec(compile(source, filename, 'exec'), original.__globals__, namespace)
    optimized = namespace[original.__name__]
    for attr in ('__name__', '__qualname__', '__module__', '__doc__', '__annotations__'):
        setattr(optimized, attr, getattr(original, attr))
    return optimized, audit


def install_native_cache_optimization(mode='official'):
    """Select official or one-statement-pruned update, returning provenance.

    Repeated calls are idempotent. Selecting ``official`` restores the original
    function after an optimized run, permitting both variants in one process.
    Source identity is checked before either mode; no CUDA call is made here.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown cache optimization {mode!r}; expected {MODES}')
    module = importlib.import_module('deploy.transformers.kv_cache')
    path = Path(module.__file__).resolve()
    if path != OFFICIAL_SOURCE.resolve():
        raise ValueError(f'Unexpected official cache source: {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != OFFICIAL_SOURCE_SHA256:
        raise ValueError('Pinned official cache source SHA256 changed')
    cls = module.MultiLayerPagedKVCache4Bit
    if cls not in _INSTALLED:
        original = cls.update
        if Path(inspect.getsourcefile(original)).resolve() != path:
            raise ValueError('Cache update method was already replaced externally')
        optimized, audit = _compile_optimized_update(original)
        _INSTALLED[cls] = dict(original=original, optimized=optimized, audit=audit)
    saved = _INSTALLED[cls]
    if cls.update not in (saved['original'], saved['optimized']):
        raise ValueError('Cache update method changed after adapter installation')
    cls.update = saved['original'] if mode == 'official' else saved['optimized']
    return dict(mode=mode, applied=mode != 'official',
        label='official native' if mode == 'official' else 'native with unused random allocation removed',
        official_source=str(path), official_source_sha256=digest,
        adapter_source=str(Path(__file__).resolve()),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        official_files_modified=False, numerical_arithmetic_changed=False,
        cache_layout_changed=False, metadata_algorithm_changed=False,
        rng_note=('Optimized mode omits the unused random draw and therefore does not advance its RNG state; '
                  'evaluation outputs use the immediately following trained transform in both modes'),
        transformation_audit=saved['audit'])
