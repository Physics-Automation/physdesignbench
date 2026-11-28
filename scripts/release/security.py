"""Inspect build contexts and every saved image layer without exposing secrets."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

from .core import ROOT, execute

PATTERNS = [re.compile(p) for p in [
    rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\r?\n[A-Za-z0-9+/=]{32,}',
    rb'(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{36}(?![A-Za-z0-9_])', rb'github_pat_[A-Za-z0-9_]{40,}',
    rb'sk-proj-[A-Za-z0-9_-]{30,}', rb'\bAKIA[A-Z0-9]{16}\b',
]]
# Exact public AWS documentation example IDs; no path-wide exemptions.
PUBLIC_EXAMPLE_HASHES = {
    '1a5d44a2dca19669d72edf4c4f1c27c4c1ca4b4408fbb17f6ce4ad452d78ddb3',
    'a9792a142a82899acc92c05206f4d26e80fb39d037bbfdbd4b17bd58c87dd248',
    '843e06a72dff62a6b86729f8f55cfcfdd2d102105f2fba88b7401be239827d3e',
    'c6ea27c534f993d31f0aef882e3d200e7b87470c379ae79c8f9b19d3bd363dc9',
}

def credential_pattern_match(content):
    return any(hashlib.sha256(match.group()).hexdigest() not in PUBLIC_EXAMPLE_HASHES
               for pattern in PATTERNS for match in pattern.finditer(content))

# Byte-identical to Debian's signed libgnutls30t64 3.8.9-3+deb13u4 amd64
# package; the library embeds public self-test keys, not runtime credentials.
PUBLIC_FIXTURE_FILES = {
    'usr/lib/x86_64-linux-gnu/libgnutls.so.30.40.3':
        '4b14edde42e7cbeefd8a6d88d035e096f2e6412a535388bb1d073a1c196e2822',
}

BAD_NAMES = {'.env','.netrc','.npmrc','id_rsa','id_ed25519'}
SECRET_NAMES = ['TIDY3D_API_KEY','SIMCLOUD_APIKEY','DOCKERHUB_TOKEN','GITHUB_TOKEN','OPENROUTER_API_KEY']


def inspect_image(image):
    ref = image['reference']
    metadata = json.loads(execute(['docker','image','inspect',ref], capture=True).stdout)[0]
    if metadata['Architecture'] != image['platform'].split('/')[-1]:
        raise ValueError('Built image platform differs from the release platform')
    labels = metadata['Config'].get('Labels') or {}
    if labels.get('org.opencontainers.image.revision') != image['source_commit']:
        raise ValueError('Image revision label does not match the source commit')
    for setting in metadata['Config'].get('Env') or []:
        key, _, value = setting.partition('=')
        if key in SECRET_NAMES and value:
            raise ValueError(f'Credential variable baked into {image["image_role"]} image: {key}')
    configured = [os.environ[key].encode() for key in SECRET_NAMES if len(os.environ.get(key,'')) >= 12]
    violations = set();files = 0
    with tempfile.TemporaryDirectory(prefix='physdesignbench-layer-audit-') as temporary:
        archive_path = Path(temporary)/'image.tar'
        execute(['docker','image','save','-o',archive_path,ref])
        with tarfile.open(archive_path) as archive:
            manifest = json.load(archive.extractfile('manifest.json'))
            for layer in manifest[0]['Layers']:
                with tarfile.open(fileobj=archive.extractfile(layer), mode='r|*') as contents:
                    for member in contents:
                        name = member.name.removeprefix('./').lstrip('/')
                        if Path(name).name in BAD_NAMES or Path(name).name.startswith('.env.') or any(name==p or name.startswith(p+'/') for p in image.get('forbidden_paths', [])):
                            violations.add(name)
                        if not member.isfile():continue
                        files += 1
                        stream = contents.extractfile(member);previous = b''
                        file_hash = hashlib.sha256();pattern_hit = False
                        while block := stream.read(1024*1024):
                            file_hash.update(block)
                            window = previous+block
                            pattern_hit = pattern_hit or credential_pattern_match(window)
                            if any(secret in window for secret in configured):
                                violations.add(name)
                            previous = window[-512:]
                        if pattern_hit and PUBLIC_FIXTURE_FILES.get(name) != file_hash.hexdigest():
                            violations.add(name)
    if violations:
        raise ValueError('Image-content audit failed; paths only (values redacted): '+', '.join(sorted(violations)[:20]))
    for command in image.get('smoke_commands', []):
        if not isinstance(command,list) or not command or not all(isinstance(x,str) for x in command):
            raise ValueError('Smoke commands must be explicit argument arrays')
        execute(['docker','run','--rm','--network','none','--platform',image['platform'],ref,*command])
    return {'image':ref,'image_id':metadata['Id'],'layer_files_inspected':files,
            'credential_scan':'passed','forbidden_paths':'passed','offline_smoke':'passed'}
