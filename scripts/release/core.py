"""Task/image inventories, immutable manifests and release-only task copies."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import tomlkit
import yaml

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = 'docker.io/physicsautomation/physdesignbench'
SOURCE_URL = 'https://github.com/Physics-Automation/physdesignbench'
DIGEST = re.compile(r'^sha256:[0-9a-f]{64}$')
VERSION = re.compile(r'^v?\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?$')


def execute(args, *, capture=False, check=True, env=None):
    result = subprocess.run([str(a) for a in args], cwd=ROOT, text=True,
                            capture_output=capture, check=False, env=env)
    if check and result.returncode:
        # Docker errors can contain credential-bearing URLs: leave full details
        # in the tool's own output, never copy environment contents into reports.
        raise RuntimeError(f'Command failed ({result.returncode}): {args[0]} {args[1]}')
    return result


def write_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def version(value):
    if not VERSION.fullmatch(value):
        raise ValueError('Use an explicit semantic release version, for example 0.1.0-pilot.1')
    return value.removeprefix('v')


def tag(task, role, release):
    value = re.sub(r'[^a-z0-9_.-]+', '-', f'{task}-{role}-v{version(release)}'.lower()).strip('.-')
    if len(value) > 128:
        value = value[:111]+'-'+hashlib.sha256(value.encode()).hexdigest()[:16]
    return value


def contained(root, value):
    path = root/value
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes its task: {value}')
    if not path.exists():
        raise ValueError(f'Missing task input: {value}')
    return path


def tree_hash(root):
    result = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if any(p in {'__pycache__', '.pytest_cache', '.git'} for p in path.parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink in release input: {path.relative_to(root)}')
        if path.is_file():
            result.update(path.relative_to(root).as_posix().encode()+b'\0')
            result.update(str(path.stat().st_mode & 0o777).encode()+b'\0')
            result.update(path.read_bytes())
    return result.hexdigest()


def lookup(document, dotted):
    current = document
    for name in dotted.split('.'):
        current = current[name]
    return current


def assign(document, dotted, value):
    current = document
    parts = dotted.split('.')
    for name in parts[:-1]:
        current = current[name]
    current[parts[-1]] = value


def inventory(task_ids, release):
    release = version(release)
    commit = execute(['git', 'rev-parse', 'HEAD'], capture=True).stdout.strip()
    dirty = bool(execute(['git', 'status', '--porcelain'], capture=True).stdout)
    if task_ids == ['all']:
        task_ids = sorted(p.parent.name for p in (ROOT/'tasks').glob('*/release.toml'))
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError('Select unique task IDs with --task, or --task all')
    images, tasks = [], []
    for task_id in task_ids:
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', task_id):
            raise ValueError('Invalid task ID')
        task = contained(ROOT/'tasks', task_id)
        spec = tomlkit.parse((task/'release.toml').read_text()).unwrap()
        config = tomlkit.parse((task/'task.toml').read_text()).unwrap()
        compose_path = task/'environment/docker-compose.yaml'
        compose = yaml.safe_load(compose_path.read_text()) if compose_path.exists() else {'services': {}}
        platform = spec['release']['platform']
        if platform != 'linux/amd64':
            raise ValueError('This release pipeline currently validates linux/amd64 only')
        if config.get('verifier', {}).get('environment_mode') == 'separate' and not spec['release'].get('public_evaluator_reviewed'):
            raise ValueError('Separate evaluator requires a documented public-distribution review')
        config_targets, service_targets = set(), set()
        for role, image in spec['images'].items():
            if not re.fullmatch(r'[a-z][a-z0-9-]*', role):
                raise ValueError('Invalid image role')
            context, dockerfile = contained(task, image['context']), contained(task, image['dockerfile'])
            if not context.is_dir() or not dockerfile.is_file():
                raise ValueError('Invalid build surface')
            for target in image.get('config_paths', []):
                lookup(config, target);config_targets.add(target)
            for service in image.get('compose_services', []):
                if service not in compose['services']:
                    raise ValueError(f'Unknown Compose service: {service}')
                service_targets.add(service)
            image_tag = tag(task_id, role, release)
            images.append({**image, 'task_id': task_id, 'image_role': role,
                           'registry': 'docker.io', 'repository': REPOSITORY,
                           'tag': image_tag, 'reference': REPOSITORY+':'+image_tag,
                           'digest': None, 'source_commit': commit, 'platform': platform,
                           'release_version': release, 'task_tree_sha256': tree_hash(task)})
        required_targets = {'environment.docker_image'}
        if config.get('verifier', {}).get('environment_mode') == 'separate':
            required_targets.add('verifier.environment.docker_image')
        if not required_targets <= config_targets:
            raise ValueError('A Harbor environment has no release image binding')
        for service, definition in compose['services'].items():
            if ('build' in definition or 'image' in definition) and service not in service_targets:
                raise ValueError(f'Unmanaged Compose image/build surface: {service}')
        tasks.append({'task_id': task_id, 'task_tree_sha256': tree_hash(task)})
    return {'schema_version': 1, 'state': 'planned', 'source_commit': commit,
            'source_dirty': dirty, 'release_version': release, 'tasks': tasks,
            'images': images, 'scientific_validation': 'not_run'}


def validate_manifest(manifest, *, published=False):
    required = {'task_id','image_role','registry','repository','tag','digest','source_commit',
                'dockerfile','context','platform','release_version'}
    seen = set()
    for image in manifest['images']:
        if not required <= image.keys():
            raise ValueError('Manifest image is incomplete')
        if image['repository'] != REPOSITORY or image['registry'] != 'docker.io':
            raise ValueError('Unexpected registry destination')
        if not re.fullmatch(r'[0-9a-f]{40}', image['source_commit']):
            raise ValueError('Manifest requires a full source commit')
        if image['tag'] != tag(image['task_id'], image['image_role'], image['release_version']):
            raise ValueError('Noncanonical image tag')
        if image['tag'] in seen:
            raise ValueError('Duplicate image tag')
        seen.add(image['tag'])
        if published and (not DIGEST.fullmatch(image.get('digest') or '') or
                          image.get('build_context') != image['context']):
            raise ValueError('Published manifest lacks an immutable digest or build context')
    if published and (manifest['state'] != 'published' or manifest['source_dirty']):
        raise ValueError('A partial or dirty build is not a published release')


def replace_references(node, substitutions):
    if isinstance(node, dict):
        return {key: replace_references(value, substitutions) for key, value in node.items()}
    if isinstance(node, list):
        return [replace_references(v, substitutions) for v in node]
    if isinstance(node, str):
        return substitutions.get(node, node)
    return node


def pin(manifest, destination):
    validate_manifest(manifest, published=True)
    destination = Path(destination).resolve()
    if destination == ROOT or destination.is_relative_to(ROOT/'tasks'):
        raise ValueError('Refusing to overwrite development tasks')
    if destination.exists():
        raise ValueError('Release destination already exists; use a new empty destination')
    staging = destination.with_name(destination.name+'.incomplete')
    if staging.exists():
        raise ValueError('Remove or inspect the previous incomplete release copy first')
    staging.mkdir(parents=True)
    try:
        for entry in manifest['tasks']:
            task_id = entry['task_id'];source = ROOT/'tasks'/task_id
            if tree_hash(source) != entry['task_tree_sha256']:
                raise ValueError('Task inputs changed after the images were built')
            target = staging/'tasks'/task_id
            shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            config = tomlkit.parse((target/'task.toml').read_text())
            compose_path = target/'environment/docker-compose.yaml'
            compose = yaml.safe_load(compose_path.read_text()) if compose_path.exists() else None
            substitutions = {}
            for image in [i for i in manifest['images'] if i['task_id']==task_id]:
                ref = image['repository']+'@'+image['digest']
                for path in image.get('config_paths', []):
                    old = lookup(config, path)
                    substitutions[str(old)] = ref
                    assign(config, path, ref)
                variables = {image.get('image_variable'): ref, image.get('digest_variable'): image['digest']}
                for key, value in variables.items():
                    if key:
                        substitutions['${'+key+':-}'] = value
                for service in image.get('compose_services', []):
                    definition = compose['services'][service]
                    definition.pop('build', None);definition['image'] = ref
            # Runtime provenance uses the evaluator's registry manifest digest,
            # not a mutable alias or the Docker configuration/image ID.
            for image in [i for i in manifest['images'] if i['task_id']==task_id and 'verifier.environment.docker_image' in i.get('config_paths', [])]:
                ref = image['repository']+'@'+image['digest']
                env = config['verifier']['environment'].get('env', {})
                if 'PICBENCH_EVALUATOR_IMAGE' in env: env['PICBENCH_EVALUATOR_IMAGE'] = ref
                if 'PICBENCH_EVALUATOR_IMAGE_DIGEST' in env: env['PICBENCH_EVALUATOR_IMAGE_DIGEST'] = image['digest']
                for service in image.get('compose_services', []):
                    env = compose['services'][service].get('environment', {})
                    if 'PICBENCH_EVALUATOR_IMAGE' in env:env['PICBENCH_EVALUATOR_IMAGE'] = ref
                    if 'PICBENCH_EVALUATOR_IMAGE_DIGEST' in env:env['PICBENCH_EVALUATOR_IMAGE_DIGEST'] = image['digest']
            # Preserve all parsed scientific/resource configuration values while
            # changing only image references in the release copy.
            config = tomlkit.parse(tomlkit.dumps(config))
            for section in [config.get('environment',{}).get('env',{}), config.get('verifier',{}).get('environment',{}).get('env',{})]:
                for key, value in list(section.items()):
                    if str(value) in substitutions:section[key] = substitutions[str(value)]
            (target/'task.toml').write_text(tomlkit.dumps(config))
            if compose:
                compose = replace_references(compose, substitutions)
                compose_path.write_text(yaml.safe_dump(compose, sort_keys=False))
        write_json(staging/'image-manifest.json', manifest)
        staging.rename(destination)
    except Exception:
        # The incomplete directory is deliberately retained for investigation.
        raise
    return destination
