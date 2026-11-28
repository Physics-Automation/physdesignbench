"""Published-image, Harbor and optional single-reference scientific validation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import tomlkit
import yaml

from .core import ROOT, execute, lookup, validate_manifest, write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harbor_case(package, *, task_id, case, reference=None):
    """Use Harbor's existing oracle interface with a validation-only solution hook."""
    directory = ROOT/'.release/validation-cases'/case/task_id
    if directory.exists():raise ValueError('Validation case already exists; retain its evidence and choose a fresh output directory')
    shutil.copytree(package/'tasks'/task_id,directory)
    solution = directory/'solution'
    if reference:
        shutil.copy2(reference,solution/'reference.py')
        script = '#!/bin/sh\nset -eu\ncp /solution/reference.py /workspace/design.py\n'
        final_sha = sha(reference)
    else:
        probe_source = "raise RuntimeError('release_probe_invalid')\n"
        final_source = "raise RuntimeError('release_final_invalid')\n"
        script = ('#!/bin/sh\nset -eu\n'
                  "cat > /workspace/design.py <<'PROBE'\n"+probe_source+'PROBE\n'
                  'set +e\nrun_probe > /logs/agent/release-probe.json 2>&1\nprobe_status=$?\nset -e\n'
                  "cat > /workspace/design.py <<'FINAL'\n"+final_source+'FINAL\n')
        final_sha = hashlib.sha256(final_source.encode()).hexdigest()
    (solution/'solve.sh').write_text(script);(solution/'solve.sh').chmod(0o755)
    jobs = ROOT/'.release/validation-jobs'/case
    execute([sys.executable,'-m','harbor.cli.main','run','--path',directory,'--agent','oracle',
             '--env','docker','--n-concurrent','1','--max-retries','0','--no-force-build','--jobs-dir',jobs])
    candidates = list(jobs.rglob('artifacts/eval.json'))
    candidates = [p for p in candidates if '/logs/artifacts/' not in str(p)] or candidates
    if not candidates:raise ValueError('Harbor did not collect a final evaluation artifact')
    evaluations = [json.loads(p.read_text()) for p in candidates]
    final = next((e for e in evaluations if e.get('submission_sha256')==final_sha or
                  e.get('frozen_submission_sha256')==final_sha or
                  e.get('evaluation_transaction',{}).get('submission_sha256')==final_sha),None)
    if final is None:raise ValueError('Final evaluation is not bound to the actual submitted source')
    if final.get('evaluation_transaction',{}).get('mode') != 'final':
        raise ValueError('A probe result was reused as final grading')
    record = {'case':case,'submission_sha256':final_sha,'eval_sha256':sha(next(p for p,e in zip(candidates,evaluations) if e==final)),
              'gate_pass':final.get('gate_pass'),'score_eligible':final.get('score_eligible'),
              'objective':final.get('official_score'),'evaluation_transaction':final.get('evaluation_transaction')}
    if reference:
        if not final.get('gate_pass') or not final.get('score_eligible'):
            raise ValueError('Reference final is invalid or score-ineligible; inspect retained Harbor artifacts')
        if not isinstance(final.get('official_score'),(int,float)) or not math.isfinite(final['official_score']):
            raise ValueError('Reference final lacks a finite objective')
    else:
        if final.get('gate_pass') or final.get('score_eligible'):
            raise ValueError('Invalid submission was accepted')
        probes = list(jobs.rglob('release-probe.json'))
        if not probes:
            raise ValueError('The native probe interface was not exercised')
        try:
            feedback = json.loads(probes[-1].read_text())
        except (ValueError, OSError):
            raise ValueError('Probe returned no valid JSON envelope; inspect retained Harbor artifacts')
        expected_probe_sha = hashlib.sha256(probe_source.encode()).hexdigest()
        if feedback.get('submission_sha256') != expected_probe_sha or feedback.get('probe_id') != 1:
            raise ValueError('Probe feedback is not bound to the probe submission')
        if feedback.get('valid') is not False or feedback.get('gate_pass') is not False:
            raise ValueError('Invalid probe was accepted')
        record['probe_interface'] = 'passed'
        record['probe_submission_sha256'] = expected_probe_sha
    return record


def validate(args):
    from harbor.models.task.config import TaskConfig
    manifest = json.loads(args.manifest.read_text());validate_manifest(manifest,published=True)
    package = args.package.resolve()
    if (args.scientific or args.reference) and not args.scientific:
        raise ValueError('Reference simulation requires explicit --scientific')
    if args.scientific and (not args.reference or not args.reference.is_file()):
        raise ValueError('Scientific validation requires an existing reference source file')
    report = {'schema_version':1,'release_version':manifest['release_version'],'images':[],
              'harbor_tasks':[],'harbor_cases':[],'scientific_validation':'not_run'}
    write_json(args.output,{**report,'state':'incomplete'})
    for image in manifest['images']:
        ref = image['repository']+'@'+image['digest']
        execute(['docker','pull','--platform',image['platform'],ref])
        inspected = json.loads(execute(['docker','image','inspect',ref],capture=True).stdout)[0]
        expected = next(c['image_id'] for c in manifest['image_checks'] if c['image']==image['reference'])
        if inspected['Id'] != expected:raise ValueError('Published image bytes differ from inspected build')
        report['images'].append({'reference':ref,'pull':'passed','image_id':expected})
    for task in manifest['tasks']:
        task_id=task['task_id'];directory=package/'tasks'/task_id
        config=tomlkit.parse((directory/'task.toml').read_text()).unwrap()
        TaskConfig.model_validate(config)
        compose_path=directory/'environment/docker-compose.yaml'
        compose=yaml.safe_load(compose_path.read_text()) if compose_path.exists() else {'services':{}}
        for image in [i for i in manifest['images'] if i['task_id']==task_id]:
            ref=image['repository']+'@'+image['digest']
            for key in image.get('config_paths',[]):
                if lookup(config,key)!=ref:raise ValueError('Harbor configuration is not digest pinned')
            for service in image.get('compose_services',[]):
                if compose['services'][service].get('image')!=ref or 'build' in compose['services'][service]:
                    raise ValueError('Compose service requires a local build or mutable image')
        original=tomlkit.parse((ROOT/'tasks'/task_id/'task.toml').read_text()).unwrap()
        for key in ['agent.timeout_sec','verifier.timeout_sec','environment.cpus','environment.memory_mb',
                    'environment.network_mode','verifier.environment.cpus','verifier.environment.memory_mb',
                    'verifier.environment_mode','verifier.collect','artifacts']:
            try:
                expected = lookup(original,key)
            except KeyError:
                continue
            if lookup(config,key)!=expected:raise ValueError('Release changed Harbor semantics: '+key)
        report['harbor_tasks'].append({'task_id':task_id,'load':'passed','resource_and_artifact_contract':'unchanged'})
        if args.harbor or args.scientific:
            report['harbor_cases'].append(harbor_case(package,task_id=task_id,case='invalid'))
        if args.scientific:
            result=harbor_case(package,task_id=task_id,case='reference',reference=args.reference.resolve())
            if args.expected_j is not None and not math.isclose(result['objective'],args.expected_j,rel_tol=1e-6,abs_tol=1e-9):
                raise ValueError('Reference objective differs from expected J')
            report['harbor_cases'].append(result)
            report['scientific_validation']='reference_final_passed'
    report['state']='passed';write_json(args.output,report)
    print(json.dumps(report,indent=2))
