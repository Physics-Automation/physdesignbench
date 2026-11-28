"""Reusable manual Docker Hub release operations for Harbor task packages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from .core import ROOT, DIGEST, SOURCE_URL, contained, execute, inventory, pin, tree_hash, validate_manifest, write_json
from .security import inspect_image


def build(args):
    manifest = inventory(args.task, args.version)
    if args.dry_run:
        print(json.dumps(manifest,indent=2));return
    if manifest['source_dirty'] and not args.allow_dirty:
        raise ValueError('Commit source before a release build, or use --allow-dirty for a non-publishable local build')
    for image in manifest['images']:
        task = ROOT/'tasks'/image['task_id']
        argv = ['docker','buildx','build','--platform',image['platform'],'--load','--provenance=false',
                '--file',contained(task,image['dockerfile']),'--tag',image['reference'],
                '--label','org.opencontainers.image.source='+SOURCE_URL,
                '--label','org.opencontainers.image.revision='+manifest['source_commit'],
                '--label','org.opencontainers.image.version='+manifest['release_version']]
        if image.get('input_hash_kind'):
            from scripts.evaluator_build_inputs import learner_build_inputs_sha256, evaluator_build_inputs_sha256
            function = {'learner': learner_build_inputs_sha256, 'evaluator': evaluator_build_inputs_sha256}[image['input_hash_kind']]
            argv += ['--label', image['input_hash_label']+'='+function(task)]
        for key,value in image.get('build_args',{}).items():
            if any(s in key.upper() for s in ['TOKEN','PASSWORD','API_KEY','SECRET']):
                raise ValueError('Secrets must not be build arguments')
            argv += ['--build-arg',key+'='+str(value)]
        execute([*argv,contained(task,image['context'])])
        image['build_context'] = image['context']
    manifest['state'] = 'built';write_json(args.manifest,manifest)


def check(args):
    manifest = json.loads(args.manifest.read_text());validate_manifest(manifest)
    if manifest['state'] not in {'built','checked'}:
        raise ValueError('Image-content checks require a completed local build')
    checks = [inspect_image(image) for image in manifest['images']]
    manifest['image_checks'] = checks;manifest['state'] = 'checked'
    write_json(args.manifest,manifest)


def registry_digest(reference, *, missing_ok=False):
    result = execute(['docker','buildx','imagetools','inspect',reference,'--format','{{.Manifest.Digest}}'],capture=True,check=False)
    if result.returncode:
        message = (result.stdout+result.stderr).lower()
        if missing_ok and not any(s in message for s in ['unauthorized','denied','forbidden','credential','timeout','tls','connection','authentication']) and (any(s in message for s in ['manifest unknown','manifest_unknown','no such manifest']) or 'not found' in message and reference.lower() in message):
            return None
        raise RuntimeError('Registry inspection failed; authentication/network errors are not missing images')
    digest = result.stdout.strip()
    if not DIGEST.fullmatch(digest):raise ValueError('Malformed registry digest')
    return digest


def publish(args):
    if not args.execute:raise ValueError('Publishing requires --execute')
    manifest = json.loads(args.manifest.read_text());validate_manifest(manifest)
    if manifest['source_dirty'] or execute(['git','status','--porcelain'],capture=True).stdout:
        raise ValueError('Publishing requires clean, committed source')
    if manifest['state'] != 'checked' or len(manifest.get('image_checks',[])) != len(manifest['images']):
        raise ValueError('Every image must pass content and offline software checks before publication')
    for image in manifest['images']:
        if tree_hash(ROOT/'tasks'/image['task_id']) != image['task_tree_sha256']:
            raise ValueError('Task changed since build')
        info = json.loads(execute(['docker','image','inspect',image['reference']],capture=True).stdout)[0]
        check_result = next(c for c in manifest['image_checks'] if c['image']==image['reference'])
        if info['Id'] != check_result['image_id']:raise ValueError('Image changed since inspection')
        if registry_digest(image['reference'], missing_ok=True):
            raise ValueError('Version tag already exists; refusing to overwrite an immutable release')
    completed = []
    try:
        for image in manifest['images']:
            execute(['docker','push',image['reference']])
            image['digest'] = registry_digest(image['reference'])
            immutable = image['repository']+'@'+image['digest']
            execute(['docker','pull','--platform',image['platform'],immutable])
            info = json.loads(execute(['docker','image','inspect',immutable],capture=True).stdout)[0]
            original = next(c['image_id'] for c in manifest['image_checks'] if c['image']==image['reference'])
            if info['Id'] != original:raise ValueError('Pulled image differs from inspected image')
            completed.append(immutable)
    except Exception:
        write_json(args.manifest.with_name('publication-incomplete.json'),
                   {'state':'incomplete','completed_images':completed,'release_version':manifest['release_version']})
        raise
    manifest['state'] = 'published';validate_manifest(manifest,published=True)
    write_json(args.output,manifest)
    print('\n'.join(completed))


def resolve(args):
    """Recover a complete manifest for checked images already in the registry."""
    manifest = json.loads(args.manifest.read_text());validate_manifest(manifest)
    if manifest['source_dirty'] or manifest['state'] != 'checked':
        raise ValueError('Registry recovery requires checked images from a clean source build')
    if len(manifest.get('image_checks', [])) != len(manifest['images']):
        raise ValueError('Every image must be inspected before registry recovery')
    for image in manifest['images']:
        if tree_hash(ROOT/'tasks'/image['task_id']) != image['task_tree_sha256']:
            raise ValueError('Task changed since the recorded build')
        image['digest'] = registry_digest(image['reference'])
        immutable = image['repository']+'@'+image['digest']
        execute(['docker','pull','--platform',image['platform'],immutable])
        actual = json.loads(execute(['docker','image','inspect',immutable],capture=True).stdout)[0]['Id']
        expected = next(c['image_id'] for c in manifest['image_checks'] if c['image']==image['reference'])
        if actual != expected:raise ValueError('Published image differs from the inspected image')
    manifest.update(state='published',manifest_recovered_from_registry=True)
    validate_manifest(manifest,published=True)
    write_json(args.output,manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ['inventory','build']:
        p=sub.add_parser(name);p.add_argument('--task',action='append',required=True)
        p.add_argument('--version',required=True);p.add_argument('--manifest',type=Path,default=ROOT/'.release/build.json')
        p.add_argument('--dry-run',action='store_true');p.add_argument('--allow-dirty',action='store_true')
    p=sub.add_parser('check');p.add_argument('--manifest',type=Path,required=True)
    p=sub.add_parser('publish');p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'.release/image-manifest.json')
    p=sub.add_parser('resolve');p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'.release/image-manifest.json')
    p=sub.add_parser('pin');p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p=sub.add_parser('validate');p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'.release/validation.json')
    p.add_argument('--harbor',action='store_true');p.add_argument('--scientific',action='store_true')
    p.add_argument('--reference',type=Path);p.add_argument('--expected-j',type=float)
    args=parser.parse_args()
    try:
        if args.command=='inventory':
            document=inventory(args.task,args.version)
            if args.dry_run:print(json.dumps(document,indent=2))
            else:write_json(args.manifest,document)
        elif args.command=='build':build(args)
        elif args.command=='check':check(args)
        elif args.command=='publish':publish(args)
        elif args.command=='resolve':resolve(args)
        elif args.command=='pin':print(pin(json.loads(args.manifest.read_text()),args.destination))
        elif args.command=='validate':
            from .validation import validate
            validate(args)
    except (ValueError,RuntimeError,KeyError,OSError) as error:
        print(f'Release failed: {error}',file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
