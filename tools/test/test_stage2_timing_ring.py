"""Exercise current production Shape timing methods with deterministic CUDA doubles.

Only the host control seam is extracted; no arithmetic implementation is copied.
The full worker A/B separately verifies device execution and oracle coverage.
"""
import argparse
import json
from pathlib import Path
import subprocess
from stage2_native_source import units
from test_cuda_p2_structure import balanced, mask

ROOT = Path(__file__).resolve().parents[2]
STAGE2 = ROOT/'src/ecm/stage2/cuda'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build', type=Path, default=ROOT/'build_stage2_local')
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.build/'build_manifest.json').read_text(encoding='utf-8-sig'))
    header = (STAGE2/'s4.h').read_text(encoding='utf-8')
    start = header.index('struct Shape {')
    end = balanced(mask(header), header.index('{', start), '{', '}') + 2
    shape = header[start:end]
    methods = '\n'.join(value for kind, _, _, value in units((STAGE2/'s4.cpp').read_text(encoding='utf-8'))
                        if kind == 'func' and 'S4Reduce::Shape::dt_' in value.split('{', 1)[0])
    order = 's.dt_sequence[i]=i+1; s.dt_next_sequence=9;' if 'dt_sequence' in shape else ''
    wrap = r'''
  {
    waits.clear();S4Reduce::Shape s;
    for(int i=0;i<8;++i){marks[i]=i;s.dt_used[i]=true;s.dt_ev[i][0]=s.dt_ev[i][1]=&marks[i];s.dt_sequence[i]=i+10;}
    s.dt_sequence[7]=1;
    require(s.dt_slot()==7 && waits==std::vector<int>({7}),"oldest slot must follow submission order after index wrap");
  }
''' if 'dt_sequence' in shape else ''
    program = r'''
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstdlib>
using cudaEvent_t=int*;
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorNotReady=34;
int query_error=0;
int queries=0;
bool ready=false;
std::vector<int> waits;
int destroyed=0;
cudaError_t cudaEventQuery(cudaEvent_t){++queries;return query_error ? query_error : ready?cudaSuccess:cudaErrorNotReady;}
cudaError_t cudaEventSynchronize(cudaEvent_t e){waits.push_back(*e);return 0;}
cudaError_t cudaEventElapsedTime(float*ms,cudaEvent_t,cudaEvent_t){*ms=1;return 0;}
cudaError_t cudaEventDestroy(cudaEvent_t){++destroyed;return 0;}
bool fifo=true;
namespace stage2_wait {
bool timing_fifo(){return fifo;}
cudaError_t event_sync(cudaEvent_t e){return cudaEventSynchronize(e);}
}
#define CK(x) do {if((x)!=cudaSuccess)throw std::runtime_error("cuda failure");}while(0)
struct S4Reduce {
@SHAPE@
};
@METHODS@
S4Reduce::Shape::~Shape(){dt_destroy();}
void require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
int main(){try{
  int marks[8];
  {
    S4Reduce::Shape s;
    for(int i=0;i<8;++i){marks[i]=i;s.dt_used[i]=true;s.dt_ev[i][0]=s.dt_ev[i][1]=&marks[i];@ORDER@}
    const int slot=s.dt_slot();
    require(queries==1,"pending oldest must prevent redundant queries of newer marks");
    require(waits.size()==1 && waits[0]==0 && slot==0,"full ring must wait only oldest mark");
    require(destroyed==2 && !s.dt_used[0] && s.dt_used[1],"newer pending slots must stay live");
    require(s.t_reduce==0.001 && s.dt_blocks==1,"oldest time counted once");
    s.dt_flush();
    require(waits.size()==8 && s.t_reduce>0.007999 && s.t_reduce<0.008001,"final drain must count every mark once");
    s.dt_flush();require(waits.size()==8,"second drain must be empty");
  }
@WRAP@
  {
    waits.clear();S4Reduce::Shape s;s.dt_used[0]=true;s.dt_ev[0][0]=s.dt_ev[0][1]=&marks[0];ready=true;
    s.dt_reap();require(!s.dt_used[0] && waits.empty() && s.t_reduce==0.001,"ready reap must not wait or lose timing");ready=false;
  }
  {
    fifo=false;waits.clear();S4Reduce::Shape s;
    for(int i=0;i<8;++i){s.dt_used[i]=true;s.dt_ev[i][0]=s.dt_ev[i][1]=&marks[i];}
    require(s.dt_slot()==0 && waits.size()==8,"A/B legacy switch must retain full-ring drain");fifo=true;
  }
  {
    const int before=destroyed;S4Reduce::Shape s;s.dt_ev[0][0]=&marks[0];s.dt_ev[0][1]=&marks[1];
    s.dt_destroy();require(destroyed==before+2,"partially created pairs must not leak");
  }
  {
    S4Reduce::Shape s;s.dt_used[0]=true;s.dt_ev[0][0]=s.dt_ev[0][1]=&marks[0];
    query_error=719;bool rejected=false;
    try{s.dt_reap();}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"query failure must not be treated as NotReady");query_error=0;
  }
  std::cout<<"timing ring: oldest-only, final drain, no double count, CUDA error propagation passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
'''.replace('@SHAPE@', shape).replace('@METHODS@', methods).replace('@ORDER@', order).replace('@WRAP@', wrap)
    cpp = out/'fixture.cpp'
    cpp.write_text(program, encoding='utf-8')
    exe = out/'fixture.exe'
    cmd = out/'compile.cmd'
    cmd.write_text('@echo off\n'+manifest['compile_setup']+'\nif errorlevel 1 exit /b 1\n"'+
        manifest['host_compiler']+'" /nologo /std:c++17 /EHsc /O2 /utf-8 "'+str(cpp)+
        '" /Fe:"'+str(exe)+'" /Fo:"'+str(out/'fixture.obj')+'"\n', encoding='utf-8')
    built = subprocess.run(['cmd', '/d', '/c', str(cmd)], capture_output=True, timeout=60)
    (out/'compile.log').write_bytes(built.stdout+built.stderr)
    if built.returncode:
        raise RuntimeError('Compile failed; see compile.log')
    run = subprocess.run([str(exe)], capture_output=True, timeout=10)
    (out/'runtime.log').write_bytes(run.stdout+run.stderr)
    print((run.stdout+run.stderr).decode(errors='replace'))
    if run.returncode:
        raise SystemExit(run.returncode)
    policy_exe = out/'policy.exe'
    policy_cmd = out/'policy_compile.cmd'
    policy_cmd.write_text('@echo off\n'+manifest['compile_setup']+'\nif errorlevel 1 exit /b 1\n"'+
        manifest['host_compiler']+'" /nologo /std:c++17 /EHsc /O2 /utf-8 "'+
        str(ROOT/'tools/test/stage2_host_wait_policy_test.cpp')+'" /Fe:"'+str(policy_exe)+
        '" /Fo:"'+str(out/'policy.obj')+'"\n', encoding='utf-8')
    built = subprocess.run(['cmd', '/d', '/c', str(policy_cmd)], capture_output=True, timeout=60)
    (out/'policy_compile.log').write_bytes(built.stdout+built.stderr)
    if built.returncode:
        raise RuntimeError('Policy compile failed')
    run = subprocess.run([str(policy_exe)], capture_output=True, timeout=10)
    (out/'policy_runtime.log').write_bytes(run.stdout+run.stderr)
    print((run.stdout+run.stderr).decode(errors='replace'))
    raise SystemExit(run.returncode)


if __name__ == '__main__':
    main()
