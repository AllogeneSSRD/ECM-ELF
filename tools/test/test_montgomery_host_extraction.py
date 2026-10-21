"""Check host math/storage extraction without changing GPU timing or payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from test_stage1_host_modules import normalize,body

ROOT=Path(__file__).resolve().parents[2]
FOLDER='src/ecm/backend/cuda/stage1/montgomery/'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old=lambda name:subprocess.check_output(['git','show','2e550a4:'+FOLDER+name],cwd=ROOT).decode('utf-8-sig')
    read=lambda name:(ROOT/FOLDER/name).read_text(encoding='utf-8-sig')
    previous=old('adapter.cu');runtime=read('adapter.cu');math=read('host_math.cpp');io=read('host_io.cpp');header=read('host_io.h')
    checked={}
    math_names=['to_mpz','from_mpz','findfactor','find_np0','allocate_and_set_s_bits','set_p_2p','set_p_2p_suyama','set_p_2p_param2','p2_jac_dbl','p2_jac_add_affine','p2_scalar_mul_base','process_results']
    io_names=['stdout_is_tty','print_progress','emit_progress_line','get_checkpoint_filename','save_checkpoint','load_checkpoint','open_dump_csv','dump_curve_state_csv']
    for target,names in [(math,math_names),(io,io_names)]:
        for name in names:
            expected,current=body(previous,name),body(target,name)
            assert expected==current,name
            checked[name]=hashlib.sha256(current.encode()).hexdigest()
    assert body(io,'should_emit_progress_line')==body(previous,'should_emit_progress_line').replace('g_last_progress_line_ms','last_ms')
    assert 'static const double kProgressLineGapMs = 200.0;' in io
    assert 'static double g_last_progress_line_ms = -1.0e9;' in runtime
    assert body(runtime,'cgbn_ecm_stage1').replace('should_emit_progress_line(batches_complete,(double)*gputime,g_last_progress_line_ms)','should_emit_progress_line(batches_complete,(double)*gputime)')==body(previous,'cgbn_ecm_stage1'),'runtime order changed'
    for name in ['cgbn_check','verify_size_of_n','ecm_build_tier_list_default','ecm_build_tier_list_param0','ecm_cuda_tier_list','ecm_cuda_tier_occupancy']:
        assert body(runtime,name)==body(previous,name),name
    before_prac=old('prac/prac_runtime.inl');prac=read('prac/prac_runtime.inl')
    assert body(io,'resident_checkpoint_write')==body(before_prac,'resident_checkpoint_write')
    a0=before_prac.index('            std::string saved;ecm_stage1_read_blob(path,saved);')
    b0=before_prac.index('        }\n        if (!resumed)',a0);restore=before_prac[a0:b0]
    expected=normalize(restore)
    prefix=normalize('const auto total=h.total,bytes=h.bytes;const auto bits=h.bits,tpi=h.tpi,curves=h.curves,torsion=h.torsion;bool resumed=false;')
    assert body(io,'resident_checkpoint_restore')=='{'+prefix+expected+'returnresumed;}'
    reinserted=prac.replace('resumed=resident_checkpoint_restore(path,h,data,sigma,prac);',restore)
    assert body(reinserted,'cgbn_stage1_resident')==body(before_prac,'cgbn_stage1_resident'),'resident runtime order changed'
    def structure(text,start,end):return normalize(text[text.index(start):text.index(end,text.index(start))+len(end)])
    assert structure(previous,'typedef struct {','} checkpoint_header_t;')==structure(header,'typedef struct {','} checkpoint_header_t;')
    assert structure(before_prac,'struct ResidentCheckpoint {','};')==structure(header,'struct ResidentCheckpoint {','};')
    assert structure(before_prac,'struct ResidentResources {','};')==structure(prac,'struct ResidentResources {','};')
    assert '#ifndef ECM_CUDA_FULL_BUILD\n#define IS_DEV_BUILD\n#endif' in math
    for name in ['host_math.h','host_io.h','host_math.cpp','host_io.cpp']:
        assert not re.search(r'#\s*include[^\n]*(?:cgbn\.h|cuda\.h|cuda_runtime|launch\.h)',read(name)),name
    inventory=a.output/'inventory.json'
    subprocess.run(['cmake','-DECM_CUDA_SOURCES_OUTPUT='+str(inventory),'-P',str(ROOT/'tools/build/internal/cuda_sources.cmake')],check=True)
    groups=json.loads(inventory.read_text())['groups']
    for name in ['host_math.cpp','host_io.cpp']:assert groups['STAGE1_CUDA'].count(FOLDER+name)==1
    result=dict(complete=True,baseline='2e550a4',unchanged_function_bodies=checked,unchanged_functions=len(checked)+2,checkpoint_layouts_unchanged=2,ladder_runtime_order_unchanged=True,resident_runtime_order_unchanged=True,carry_geometry_and_device_resources_retained=True,adapter_lines=len(runtime.splitlines()),math_cpp_lines=len(math.splitlines()),io_cpp_lines=len(io.splitlines()),cuda_runs=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='unchanged_function_bodies'}))


if __name__=='__main__':main()
