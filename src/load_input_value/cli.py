"""Small command surface; heavy dependencies are imported only when requested."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog='liv', description='Load input value research companion')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('reproduce', 'validate'):
        cmd = sub.add_parser(name, help='Rebuild or validate the public paper artifacts')
        cmd.add_argument('--root', type=Path, default=Path.cwd(), help='Repository checkout root')
        cmd.add_argument('--output-dir', type=Path)
    experiment = sub.add_parser('experiment', help='Explicit private-data training or synthetic smoke')
    actions = experiment.add_subparsers(dest='action', required=True)
    run = actions.add_parser('run', help='Full historical protocol; requires private CSV and training extra')
    run.add_argument('--data', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--config', type=Path, default=Path('configs/paper.json'))
    smoke = actions.add_parser('smoke', help='Small synthetic plumbing check; never paper reproduction')
    smoke.add_argument('--output', type=Path, required=True)
    smoke.add_argument('--config', type=Path, default=Path('configs/paper.json'))
    reload = sub.add_parser('_reload', help=argparse.SUPPRESS)
    reload.add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in ('reproduce', 'validate'):
            script = args.root.resolve() / 'scripts/reproduce_artifacts.py'
            if not script.is_file():
                raise FileNotFoundError('Public artifact script not found; run from a repository checkout or supply --root.')
            command = [sys.executable, str(script)]
            if args.command == 'validate':
                command.append('--check-only')
            if args.output_dir:
                command.extend(['--output-dir', str(args.output_dir.resolve())])
            subprocess.run(command, cwd=args.root, check=True)
            return 0
        if args.command == '_reload':
            from .training import reload_check
            result = reload_check(args.directory)
        elif args.action == 'run':
            # Explain absent private data before importing optional scientific libraries.
            if not args.data.is_file():
                raise FileNotFoundError('Private input CSV not found. Supply --data PATH; see docs/data-contract.md.')
            from .experiments import run
            result = run(args.data, args.output, args.config)
        else:
            from .experiments import smoke
            result = smoke(args.output, args.config)
        print(json.dumps(result, indent=2))
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, ImportError) as error:
        print(f'liv: {error}', file=sys.stderr)
        if isinstance(error, ImportError):
            print('Install the optional training environment with: python -m pip install ".[train]"', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
