"""Public model catalog, selective download and text-completion commands."""
import argparse
import json
from .hub import list_models, download_checkpoint


def main():
    parser = argparse.ArgumentParser(prog='prismquant', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('models', help='List released models and checkpoint variants')
    for name in ['download', 'generate']:
        p = commands.add_parser(name)
        p.add_argument('--model', required=True, choices=list_models())
        p.add_argument('--checkpoint', help='Exact variant name from prismquant models')
        p.add_argument('--cache-dir')
        p.add_argument('--local-files-only', action='store_true')
        if name == 'generate':
            p.add_argument('--prompt', required=True)
            p.add_argument('--max-new-tokens', type=int, default=64)
            p.add_argument('--device', default='cuda')
            p.add_argument('--device-map', choices=['auto', 'balanced'])
    args = parser.parse_args()
    if args.command == 'models':
        print(json.dumps({k: {key: val for key, val in v.items() if key != 'checkpoints'} |
                          {'checkpoints': list(v['checkpoints'])} for k, v in list_models().items()}, indent=2))
        return
    options = dict(cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    if args.command == 'download':
        print(download_checkpoint(args.model, args.checkpoint, **options))
        return
    from .model import load_model
    loaded = load_model(args.model, args.checkpoint, device=args.device, device_map=args.device_map, **options)
    try:
        print(loaded.generate(args.prompt, max_new_tokens=args.max_new_tokens))
    finally:
        loaded.close()
