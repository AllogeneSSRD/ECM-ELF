"""Prove result encoding, CRC and ledger ordering moved without protocol changes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from test_stage1_host_modules import normalize

ROOT=Path(__file__).resolve().parents[2]


def body(text,name):
    # Empty default arguments contain braces; find the definition after its
    # balanced parameter list rather than treating those defaults as a body.
    masked=re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*',lambda m:' '*len(m[0]),text,flags=re.S)
    def end(at,left,right):
        depth=0
        for i in range(at,len(masked)):
            depth+=(masked[i]==left)-(masked[i]==right)
            if not depth:return i
        raise AssertionError(name)
    for m in re.finditer(r'\b'+name+r'\s*\(',masked):
        at=masked.index('(',m.start());i=end(at,'(',')')+1
        while i<len(masked) and masked[i].isspace():i+=1
        if i<len(masked) and masked[i]=='{':return normalize(text[i:end(i,'{','}')+1])
    raise AssertionError(name)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old=lambda name:subprocess.check_output(['git','show','e2ead4d:'+name],cwd=ROOT).decode('utf-8-sig')
    read=lambda name:(ROOT/name).read_text(encoding='utf-8-sig')
    folder='src/ecm/storage/results/'
    impl=read(folder+'publication.cpp');public=read('include/ecm/storage/results/publication.h')
    functions={
        'ecm_result_submission.h':['quoted','decimal','normalized_integer','utc_timestamp','reference_fft','target_exponent','crc32','crc_hex','with_result_checksum','submission_json'],
        'ecm_internal_result.h':['internal_json'],
        'ecm_result_paths.h':['legacy_submission_ledger','submission_ledger','has_result_ledger','has_result_ledger_directory'],
        'ecm_submission_writer.h':['atomic_text','read_submission_entry','migrate_result_ledger','stage1_factor_key','require_ledger_format','read_submission_bytes','append_submissions','submission_order','seal_submissions','refresh_submission_if_owned','prepare_submission_ledger','ensure_submission_output','publish_submission','render_internal','prepare_internal_ledger','publish_internal_batch','publish_internal','refresh_internal']}
    checked={}
    for file,names in functions.items():
        previous=old(folder+file)
        assert not (ROOT/folder/file).exists(),file
        for name in names:
            expected,actual=body(previous,name),body(impl,name);assert expected==actual,name
            checked[name]=hashlib.sha256(actual.encode()).hexdigest()
    def structure(s,first,last):return normalize(s[s.index(first):s.index(last,s.index(first))+len(last)])
    assert structure(old(folder+'ecm_result_submission.h'),'struct Submission {','\n};')==structure(public,'struct Submission {','\n};')
    assert structure(old(folder+'ecm_submission_writer.h'),'struct SubmissionLock {','\n};')==structure(impl,'struct SubmissionLock {','\n};')
    assert structure(old(folder+'ecm_submission_writer.h'),'struct SubmissionEntry {','};')==structure(impl,'struct SubmissionEntry {','};')
    expected=body(old('src/ecm/stage1/task.cpp'),'stage1_results_complete').replace('opt.persistence.results_file','path')
    assert body(impl,'contains_stage1_curves')==expected
    assert 'stage1_results_complete' not in read('src/ecm/stage1/task.cpp')
    for name in ['src/ecm/montgomery_stage1/cuda/workflow.cpp','src/ecm/montgomery_stage1/opencl/workflow.cpp','src/ecm/edwards_stage1/cuda/workflow.cpp']:
        s=read(name);assert 'ecm_results::contains_stage1_curves(opt.persistence.results_file,' in s
    assert 'SubmissionLock' not in public and not re.search(r'#\s*include[^\n]*(?:windows|src/|cuda|cgbn)',public)
    assert not re.search(r'inline |std::ofstream|std::ifstream|directory_iterator|mpz_gcd\(|WriteFile\(',public)
    inventory=a.output/'inventory.json'
    subprocess.run(['cmake','-DECM_CUDA_SOURCES_OUTPUT='+str(inventory),'-P',str(ROOT/'tools/build/internal/cuda_sources.cmake')],check=True)
    groups=json.loads(inventory.read_text())['groups']
    for group in ['STAGE1_COMMON','STAGE2_HOST']:assert groups[group].count(folder+'publication.cpp')==1
    cpu=read('tools/build/internal/ecm_cpu_stage2.cmake');assert '"${_ecm_repo}/'+folder+'publication.cpp"' in cpu
    result=dict(complete=True,baseline='e2ead4d',function_bodies=checked,unchanged_functions=len(checked),submission_layout_unchanged=True,lock_and_entry_layout_unchanged=True,complete_receipt_scan_moved_to_writer=True,public_os_handle_types=0,public_implementation_bodies=0,implementation_lines=len(impl.splitlines()),cuda_runs=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='function_bodies'}))


if __name__=='__main__':main()
