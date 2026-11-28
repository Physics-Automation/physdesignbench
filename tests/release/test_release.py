import copy
import json
from pathlib import Path

import pytest
import tomlkit
import yaml

from scripts.release import core
from scripts.release.__main__ import registry_digest
from scripts.release.security import PATTERNS

TASK = 'anisotropic-swg-beamsplitter-v0'


def planned():
    return core.inventory([TASK], '0.1.0-pilot.1')


def published():
    result = planned();result.update(state='published',source_dirty=False)
    for i,image in enumerate(result['images']):
        image.update(digest='sha256:'+str(i+1)*64, build_context=image['context'])
    return result


def test_two_real_build_contexts_and_harbor_roles():
    result = planned()
    assert {i['context'] for i in result['images']} == {'environment','.'}
    assert {i['image_role'] for i in result['images']} == {'learner','evaluator'}
    assert all(i['platform']=='linux/amd64' for i in result['images'])


def test_tags_are_valid_deterministic_and_bounded():
    assert core.tag(TASK,'learner','v0.1.0') == core.tag(TASK,'learner','0.1.0')
    assert len(core.tag('a'*160,'sidecar','0.1.0')) <= 128
    with pytest.raises(ValueError):core.version('latest')
    with pytest.raises(ValueError):core.version('1.0; touch /tmp/oops')


def test_pin_preserves_science_budgets_and_source_tree(tmp_path):
    manifest = published();source = core.ROOT/'tasks'/TASK
    before = core.tree_hash(source)
    target = core.pin(manifest,tmp_path/'release')/'tasks'/TASK
    original = tomlkit.parse((source/'task.toml').read_text()).unwrap()
    released = tomlkit.parse((target/'task.toml').read_text()).unwrap()
    assert released['agent'] == original['agent']
    assert released['artifacts'] == original['artifacts']
    assert released['verifier']['timeout_sec'] == original['verifier']['timeout_sec']
    for path in ['instruction.md','evaluator/runtime/evaluator.py','evaluator/runtime/task_profile.json','solution/solve.sh','tests/test.sh']:
        assert (source/path).read_bytes() == (target/path).read_bytes()
    assert core.tree_hash(source) == before
    for role in manifest['images']:
        for key in role['config_paths']:assert core.lookup(released,key).endswith('@'+role['digest'])
    compose = yaml.safe_load((target/'environment/docker-compose.yaml').read_text())
    assert 'build' not in compose['services']['evaluator']
    assert compose['services']['evaluator']['image'].endswith('@sha256:'+'2'*64)
    assert released['verifier']['environment']['env']['PICBENCH_EVALUATOR_IMAGE_DIGEST']=='sha256:'+'2'*64
    assert released['verifier']['environment']['env']['TIDY3D_API_KEY']=='${TIDY3D_API_KEY:-}'


def test_partial_manifests_and_overwrites_are_rejected(tmp_path):
    data=published();data['images'][0]['digest']=None
    with pytest.raises(ValueError):core.pin(data,tmp_path/'partial')
    with pytest.raises(ValueError):core.pin(published(),core.ROOT/'tasks'/TASK)
    folder=tmp_path/'existing';folder.mkdir()
    with pytest.raises(ValueError):core.pin(published(),folder)


def test_context_path_escape_and_symlinks_are_rejected(tmp_path):
    with pytest.raises(ValueError):core.contained(core.ROOT/'tasks', '../README.md')
    (tmp_path/'link').symlink_to(core.ROOT/'README.md')
    with pytest.raises(ValueError):core.tree_hash(tmp_path)


def test_registry_auth_failure_is_not_treated_as_an_unused_tag(monkeypatch):
    from scripts.release import __main__ as cli
    class Result:
        returncode=1;stdout='';stderr='error getting credentials: credential helper not found'
    monkeypatch.setattr(cli,'execute',lambda *a,**k:Result())
    with pytest.raises(RuntimeError):registry_digest('docker.io/example/image:0.1.0',missing_ok=True)


def test_registry_missing_manifest_can_be_new_release(monkeypatch):
    from scripts.release import __main__ as cli
    class Result:
        returncode=1;stdout='';stderr='manifest unknown'
    monkeypatch.setattr(cli,'execute',lambda *a,**k:Result())
    assert registry_digest('docker.io/example/image:0.1.0',missing_ok=True) is None


def test_known_credential_shapes_are_detected_without_real_credentials():
    example=b'sk-proj-'+b'a'*40
    assert any(p.search(example) for p in PATTERNS)
    assert not any(p.search(b'agenticpic-layout-token') for p in PATTERNS)


def test_secret_scanner_distinguishes_public_examples_and_library_strings():
    from scripts.release.security import credential_pattern_match
    assert credential_pattern_match(b'AKIA' + b'1'*16)
    assert not credential_pattern_match(b'AKIAIOSFODNN7EXAMPLE')
    assert not credential_pattern_match(b'AAAAAKIA' + b'B'*32)
    assert not credential_pattern_match(b'-----BEGIN PRIVATE KEY-----')
    assert credential_pattern_match(b'-----BEGIN PRIVATE KEY-----\n'+b'A'*64)


def test_single_image_task_without_compose_can_be_pinned(tmp_path, monkeypatch):
    root = tmp_path/'repo'; task = root/'tasks/example';task.mkdir(parents=True)
    (task/'Dockerfile').write_text('FROM scratch\n')
    (task/'task.toml').write_text('[environment]\ndocker_image="example:dev"\ncpus=1\n')
    (task/'release.toml').write_text('''[release]
platform="linux/amd64"
[images.learner]
context="."
dockerfile="Dockerfile"
config_paths=["environment.docker_image"]
''')
    monkeypatch.setattr(core,'ROOT',root)
    def git_result(args, **kwargs):
        class Result:
            stdout = 'a'*40 if args[1]=='rev-parse' else ''
        return Result()
    monkeypatch.setattr(core,'execute',git_result)
    manifest=core.inventory(['example'],'1.0.0')
    manifest.update(state='published',source_dirty=False)
    manifest['images'][0].update(digest='sha256:'+'1'*64,build_context='.')
    target=core.pin(manifest,tmp_path/'package')/'tasks/example'
    result=tomlkit.parse((target/'task.toml').read_text())
    assert result['environment']['docker_image'].endswith('@sha256:'+'1'*64)
    assert result['environment']['cpus']==1


def test_registry_recovery_never_emits_partial_success(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.release import __main__ as cli
    data=published();data['state']='checked'
    data['image_checks']=[{'image':i['reference'],'image_id':'sha256:'+'a'*64} for i in data['images']]
    source=tmp_path/'build.json';source.write_text(json.dumps(data))
    output=tmp_path/'complete.json';seen=[]
    def digest(reference):
        seen.append(reference)
        if len(seen)>1:raise RuntimeError('Second image not available')
        return 'sha256:'+'1'*64
    class Result:
        stdout=json.dumps([{'Id':'sha256:'+'a'*64}])
    monkeypatch.setattr(cli,'registry_digest',digest)
    monkeypatch.setattr(cli,'execute',lambda *a,**k:Result())
    with pytest.raises(RuntimeError):cli.resolve(SimpleNamespace(manifest=source,output=output))
    assert not output.exists()


def test_store_specific_ids_must_identify_the_same_published_manifest(monkeypatch):
    image={'repository':core.REPOSITORY,'digest':'sha256:'+'1'*64}
    config='sha256:'+'2'*64
    class Result:
        stdout=json.dumps({'config':{'digest':config}})
    monkeypatch.setattr(core,'execute',lambda *a,**k:Result())
    assert core.matching_image_ids(image['digest'],config,image)
    assert core.matching_image_ids(config,image['digest'],image)
    assert not core.matching_image_ids('sha256:'+'3'*64,config,image)


def test_scientific_preflight_without_credentials_starts_no_provider_job(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.release import validation
    source=tmp_path/'manifest.json';source.write_text(json.dumps(published()))
    reference=tmp_path/'reference.py';reference.write_text('raise RuntimeError()\n')
    output=tmp_path/'validation.json'
    monkeypatch.delenv('TIDY3D_API_KEY',raising=False)
    monkeypatch.delenv('SIMCLOUD_APIKEY',raising=False)
    def no_execution(*args,**kwargs):raise AssertionError('A command ran before credential preflight')
    monkeypatch.setattr(validation,'execute',no_execution)
    args=SimpleNamespace(manifest=source,package=tmp_path,scientific=True,reference=reference,output=output)
    with pytest.raises(ValueError,match='no provider evaluation was started'):validation.validate(args)
    assert json.loads(output.read_text())['scientific_validation']=='blocked_missing_runtime_credentials'
