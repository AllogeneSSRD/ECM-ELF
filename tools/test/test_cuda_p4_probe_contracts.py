"""Finite P4 probe seams; real production math and independent oracles stay intact."""
from source_layout import current_source,current_files,relocation_text
import argparse,json,re,subprocess
from pathlib import Path
from test_cuda_p2_structure import mask
from test_cuda_p2_host import tokens
from test_cuda_p4_isolation import normalized,expand
from stage2_native_source import ROOT,units,signature


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    frozen=lambda p:subprocess.check_output(['git','show','2a2606e:'+p],cwd=ROOT).decode('utf-8-sig')
    text=lambda p:relocation_text(p)
    # Nothing in the production Stage1 or NTT algorithms/host lifetime moved.
    unchanged=['src/cuda/stage1','kernels/cuda/stage1','src/cuda/ntt','kernels/cuda/ntt',
               'include/ecm/cuda/tiers.def','include/ecm/cuda/ntt_parameters.h',
               'tools/test/stage2_s4_program_fixture.cpp']
    files=subprocess.check_output(['git','ls-tree','-r','--name-only','2a2606e','--',*unchanged],cwd=ROOT).decode().splitlines()
    for name in files:assert frozen(name)==relocation_text(name),'production body changed beyond relocation: '+name
    before=frozen('tools/test/stage1_host_wait_probe.cu');after=text('tools/test/stage1_host_wait_probe.cu')
    begin=before.index('#define ECM_TIERS_RESTRICTED');end=before.index('\nstatic void require',begin)
    before=before[:begin]+before[end:]
    call='ecm_edwards_cuda_detail::DIAGNOSTIC_JOIN(run_,ECM_HOST_WAIT_TIER)({n.v,scalar.v,states.data(),count,done,chunk,nullptr,nullptr,&plan,4,&metrics,nullptr})'
    newcall='ecm_cuda_diagnostics::host_wait_run(hooks,n.v,scalar.v,states.data(),count,done,chunk,plan,4,&metrics)'
    assert before.count(call)==after.count(newcall)==1
    before=before.replace(call,'HOST_WAIT_CALL()');after=after.replace(newcall,'HOST_WAIT_CALL()')
    hooks='''        const ecm_cuda_diagnostics::HostWaitHooks hooks{diagnostic_event_create,diagnostic_event_record,
            diagnostic_event_sync,diagnostic_device_sync,diagnostic_memcpy,diagnostic_last_error,diagnostic_elapsed};
'''
    assert after.count(hooks)==1;after=after.replace(hooks,'')
    assert normalized(before)==normalized(after),'host-wait hooks, arguments, point export or report changed'
    adapter=text('tools/test/adapters/edwards_host_wait.cu')
    assert adapter.count('#include "../../../src/cuda/stage1/edwards_runtime.inl"')==1
    assert '{n,scalar,states,count,done,chunk,nullptr,nullptr,&plan,doubling,metrics,nullptr}' in adapter
    assert not re.search(r'\bcuda(?:Malloc|Free|Memcpy|DeviceSynchronize|EventSynchronize)\s*\(',mask(adapter))
    assert 'active_hooks=previous;' in adapter and 'active_hooks=&hooks;' in adapter
    # The rotation body never had a production instantiation. Its control and
    # field primitives now have explicit experimental ownership and namespace.
    old=frozen('kernels/cuda/stage2/giant_chain.cuh')
    variant=text('tools/experiments/cuda/stage2_tree/giant_chain_variant.cuh')
    prefix=old[:old.index('// Probe-only buffer rotation')]
    prefix=prefix[prefix.index('template <int NW'):]
    rotation=old[old.index('// Probe-only buffer rotation'):old.index('\ntemplate<int NW,bool XADD6>\nvoid ecm_launch_')]
    assert normalized(variant)==normalized(prefix+rotation)
    assert 's2g_chain_rotate_kernel' not in text('kernels/cuda/stage2/giant_chain.cuh')
    point=frozen('kernels/cuda/stage2/point.cuh');at=point.index('__global__ void s2g_ladder_kernel');at=point.rfind('template ',0,at)
    assert normalized(point[:at])==normalized(text('tools/experiments/cuda/stage2_tree/giant_point.cuh'))
    assert normalized(frozen('kernels/cuda/stage2/point_mersenne.cuh'))==normalized(text('tools/experiments/cuda/stage2_tree/giant_point_mersenne.cuh'))
    owner=text('tools/experiments/cuda/stage2_tree/giant_variant.cu')
    assert 'namespace ecm_stage2_giant_experiment {' in owner
    assert '#include "giant_point.cuh"' in owner and 'src/cuda/' not in owner
    # Independent GMP recurrence/export and every CLI validation remain exact.
    old=frozen('tools/test/stage2_giant_point_probe.cu');new=text('tools/test/stage2_giant_point_probe.cu')
    def without_launch(s):
        for kind,_,_,value in units(s):
            if kind=='func' and signature(value)[0]=='launch':return s.replace(value,'')
        raise AssertionError('missing giant launch seam')
    reset='S2GDeviceConstants constants{};CK(ecm_stage2_copy_constants(&constants,sizeof(constants)));'
    assert new.count(reset)==1;new=new.replace(reset,'')
    assert normalized(without_launch(old))==normalized(without_launch(new))
    assert 'if constexpr(NW==64 || NW==128 || NW==256)' in text('tools/test/stage2_giant_point_probe.cu')
    assert 'ecm_launch_s2g_chain_kernel<NW,true>(cfg,' in text('tools/test/stage2_giant_point_probe.cu')
    # Pack data generation, canaries and GMP indexing stay independent. Only
    # the two literal CUDA launches become calls into the actual arithmetic CU.
    old=frozen('tools/bench/stage2_pack_zero_probe.cu');new=text('tools/bench/stage2_pack_zero_probe.cu')
    no_includes=lambda s:re.sub(r'^\s*#\s*include[^\n]*\n','',s,flags=re.M)
    assert tokens(no_includes(old))==tokens(no_includes(new))
    old=frozen('kernels/cuda/stage2/pack.cuh');start=old.index('namespace stage2_pack {');end=old.index('\nnamespace stage2_pack {',start+1)
    assert normalized(old[start:end])==normalized(text('kernels/cuda/shared/pack_index.cuh'))
    new=text('kernels/cuda/stage2/pack.cuh').replace('#include "../shared/pack_index.cuh"','')
    assert normalized(old[:start]+old[end:])==normalized(new)
    result=dict(complete=True,unchanged_production_modules=unchanged,
        seams=['host_wait_hook_table','registered_giant_actual_object','experimental_giant_rotation_and_widths',
               'pack_actual_arithmetic_object','shared_pack_quotient'],independent_oracles_unchanged=True)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,seams=len(result['seams']))))


if __name__=='__main__':main()
