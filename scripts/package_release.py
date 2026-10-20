#!/usr/bin/env python3
"""Build a reproducible source archive from the checked-in release allowlist."""
import argparse
import gzip
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def collect_files(root, manifest):
    root = Path(root).resolve()
    files = set()
    for entry in manifest['include']:
        relative = PurePosixPath(entry)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid release path: {entry}')
        source = root / entry
        if source.is_symlink():
            raise ValueError(f'Release entries must be source files, not links: {entry}')
        if not source.exists():
            raise FileNotFoundError(source)
        candidates = source.rglob('*') if source.is_dir() else [source]
        for path in candidates:
            rel = path.relative_to(root)
            if any(part in manifest['exclude_directories'] or part.endswith('.egg-info') for part in rel.parts):
                continue
            if path.suffix in manifest['exclude_suffixes']:
                continue
            if path.suffix not in manifest['source_suffixes'] and path.name != '.gitignore':
                continue
            if path.is_symlink():
                raise ValueError(f'Release source must not be a link: {rel}')
            if path.is_file():
                files.add(rel.as_posix())
    return sorted(files)


def build_archive(root, manifest, output):
    root = Path(root).resolve()
    output = Path(output)
    files = collect_files(root, manifest)
    contents = {name: (root / name).read_bytes() for name in files}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing an existing release.
    with output.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as archive:
                for name, data in sorted(contents.items()):
                    info = tarfile.TarInfo('ReWild4D/' + name)
                    info.size = len(data)
                    info.mode = 0o755 if (root / name).stat().st_mode & 0o111 else 0o644
                    archive.addfile(info, io.BytesIO(data))
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='Print selected paths without writing an archive')
    parser.add_argument('--output', type=Path, help='New .tar.gz output file (must not already exist)')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'release_manifest.json').read_text())
    if args.list:
        print('\n'.join(collect_files(ROOT, manifest)))
    elif args.output is not None:
        count = build_archive(ROOT, manifest, args.output)
        print(f'Packaged {count} source files into {args.output}')
    else:
        parser.error('provide --list or --output')


if __name__ == '__main__':
    main()
