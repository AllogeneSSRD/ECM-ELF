"""Compile the authoritative root digest kernel in a small, isolated CUDA harness.

The kernel text is extracted verbatim; no second maintained implementation.
Run only on the selected device. Writes provenance and the test executable.
"""
import argparse, hashlib, json, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--device', type=int, default=1)
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
source = repo / 'tools/bench/stage2_tree_gpu.cu'
raw = source.read_bytes()
text = raw.decode('utf-8').replace('\r\n', '\n')
start = text.index('// Two order-sensitive 64-bit checksums')
end = text.index('struct FoldDeviceStats {', start)
kernel = text[start:end]
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
prefix = r'''
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(2);}} while(0)
'''
harness = r'''
int main(int argc,char **argv) {
    CK(cudaSetDevice(argc>1?std::atoi(argv[1]):1));
    unsigned long long *input=nullptr,*digest=nullptr;
    CK(cudaMalloc(&input,1048619ull*8));CK(cudaMalloc(&digest,16));
    unsigned cases=0;unsigned long long words=0;
    for(size_t count:{0u,1u,255u,256u,257u,1025u,262149u,1048619u}) {
        std::vector<unsigned long long> data(count);
        for(size_t i=0;i<count;++i)data[i]=i%11?0x9e3779b97f4a7c15ull*(i+3):~0ull;
        if(count)CK(cudaMemcpy(input,data.data(),count*8,cudaMemcpyHostToDevice));
        for(unsigned long long base:{0ull,(1ull<<40)+17}) {
            unsigned long long expected[2]={0,0},position=base;
            groot_input_digest_host(data,position,expected);
            for(bool segmented:{false,true}) {
                CK(cudaMemset(digest,0,16));
                const size_t segment=segmented?65539:std::max(count,(size_t)1);
                for(size_t offset=0;offset<count;offset+=segment) {
                    const size_t n=std::min(count-offset,segment);
                    const unsigned blocks=(unsigned)std::min((size_t)1024,(n+255)/256);
                    groot_input_digest_kernel<<<blocks,256>>>(input+offset,n,base+offset,digest);
                    CK(cudaGetLastError());
                }
                unsigned long long actual[2];CK(cudaMemcpy(actual,digest,16,cudaMemcpyDeviceToHost));
                if(actual[0]!=expected[0] || actual[1]!=expected[1]) {
                    std::fprintf(stderr,"FAIL count=%llu base=%llu segmented=%d\n",(unsigned long long)count,base,(int)segmented);return 3;
                }
                ++cases;words+=count;
            }
        }
    }
    CK(cudaFree(input));CK(cudaFree(digest));
    std::printf("root_digest_fixture: cases=%u words=%llu bad=0\n",cases,words);return 0;
}
'''
(out / 'root_digest.cu').write_text(prefix + kernel + harness, encoding='utf-8')
vc = sorted(Path('C:/Program Files/Microsoft Visual Studio').glob('*/*/VC/Auxiliary/Build/vcvars64.bat'))
assert vc, 'vcvars64.bat missing'
(out / 'build.cmd').write_text(f'@echo off\ncall "{vc[0]}"\nif errorlevel 1 exit /b %errorlevel%\nnvcc -std=c++17 -O3 -arch=sm_89 root_digest.cu -o root_digest.exe\nexit /b %errorlevel%\n', encoding='utf-8', newline='\r\n')
build = subprocess.run(['cmd.exe', '/c', 'build.cmd'], cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(out / 'build.log').write_bytes(build.stdout)
assert build.returncode == 0, build.stdout.decode('utf-8', errors='replace')
run = subprocess.run([str(out / 'root_digest.exe'), str(a.device)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(out / 'fixture.log').write_bytes(run.stdout)
assert run.returncode == 0 and b'cases=32' in run.stdout, run.stdout
manifest = {'source': str(source), 'source_sha256': hashlib.sha256(raw).hexdigest(),
    'kernel_sha256': hashlib.sha256(kernel.encode()).hexdigest(), 'device': a.device,
    'exe_sha256': hashlib.sha256((out / 'root_digest.exe').read_bytes()).hexdigest(),
    'passed': 32, 'failed': 0, 'stdout': run.stdout.decode()}
(out / 'summary.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(run.stdout.decode().strip())
