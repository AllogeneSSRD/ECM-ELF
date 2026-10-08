"""Check the exact frozen resident-root reversal kernel, including zero padding.

This probe covers word mapping and disjoint ranges in one allocation. Complete
polynomial arithmetic is checked by bench_stage2_gscale.py --target scaled-root.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from bench_stage2_production import freeze, sha

DRIVER = r'''
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdlib>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));return 2;} } while(0)
KERNEL_BODY
int main(int argc,char **argv) {
  int device=argc>1?std::atoi(argv[1]):1;CHECK(cudaSetDevice(device));
  cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,device));
  if(prop.major!=8 || prop.minor!=9)return 2;
  char uuid[33];const char *hex="0123456789abcdef";
  for(int j=0;j<16;++j){unsigned char c=(unsigned char)prop.uuid.bytes[j];uuid[2*j]=hex[c>>4];uuid[2*j+1]=hex[c&15];}uuid[32]=0;
  unsigned long long cases=0,words=0;
  for(size_t W:{1u,8u,9u,20u,35u,70u,128u,129u,256u})
  for(size_t P:{1u,2u,3u,17u,129u}) {
    std::vector<size_t> counts{1,P>1?P-1:1,P};
    std::sort(counts.begin(),counts.end());counts.erase(std::unique(counts.begin(),counts.end()),counts.end());
    for(size_t count:counts)for(unsigned zero=0;zero<2;++zero) {
      const size_t n=P*W,output=n+16;
      const unsigned long long sentinel=0xd79c6eb25a1843ffull;
      std::vector<unsigned long long> before(2*n+24,sentinel),want,got;
      for(size_t i=0;i<count*W;++i)before[8+i]=zero?0:(0x9e3779b97f4a7c15ull*(i+1)^((unsigned long long)W<<32)^P);
      want=before;got.resize(before.size());
      for(size_t i=0;i<n;++i){const size_t c=P-1-i/W;want[output+i]=c<count?before[8+c*W+i%W]:0;}
      unsigned long long *memory=nullptr;CHECK(cudaMalloc(&memory,before.size()*8));
      CHECK(cudaMemcpy(memory,before.data(),before.size()*8,cudaMemcpyHostToDevice));
      scaled_root_reverse_kernel<<<(unsigned)((n+255)/256),256>>>(memory+8,count,P,(int)W,memory+output);
      CHECK(cudaGetLastError());CHECK(cudaMemcpy(got.data(),memory,got.size()*8,cudaMemcpyDeviceToHost));
      CHECK(cudaFree(memory));
      if(got!=want){std::fprintf(stderr,"mapping/guard mismatch W=%zu P=%zu count=%zu zero=%u\n",W,P,count,zero);return 3;}
      ++cases;words+=n;
    }
  }
  std::printf("{\"device\":%d,\"uuid\":\"%s\",\"cases\":%llu,\"words\":%llu,\"bad\":0}\n",device,uuid,cases,words);
  return 0;
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args()
    exe = a.exe.resolve(); out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise ValueError('use a fresh output directory')
    identity = freeze(exe)
    relative = 'tools/bench/stage2_tree_gpu.cu' if 'tools/bench/stage2_tree_gpu.cu' in identity['sources'] else 'src/cuda/ecm_cuda_stage2.cu'
    source = exe.parent / 'sources' / relative
    if sha(source) != identity['sources'][relative]: raise ValueError('frozen source changed')
    text = source.read_text()
    start = text.index('__global__ void scaled_root_reverse_kernel(')
    end = text.index('struct FoldDeviceStats {', start)
    kernel = text[start:end]
    probe = out / 'reverse_probe.cu'; probe.write_text(DRIVER.replace('KERNEL_BODY', kernel), newline='\n')
    vc = next(Path('C:/Program Files/Microsoft Visual Studio').glob('*/Community/VC/Auxiliary/Build/vcvars64.bat'))
    binary = out / 'reverse_probe.exe'
    if any(any(c in str(s) for c in '&|<>%!^\r\n"') for s in (vc, probe, binary)): raise ValueError('unsafe compiler path')
    command = f'call "{vc}" >nul 2>&1\nnvcc -std=c++17 -O3 -arch=sm_89 "{probe}" -o "{binary}"\nexit /b %errorlevel%\n'
    build = out / 'build.cmd'; build.write_text('@echo off\n' + command)
    with (out / 'build.log').open('wb') as stream:
        subprocess.run(['cmd.exe', '/d', '/c', str(build)], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=120)
    result = subprocess.run([str(binary), str(a.device)], capture_output=True, check=True, timeout=120)
    (out / 'probe.log').write_bytes(result.stdout + result.stderr)
    record = json.loads(result.stdout)
    if record['cases'] != 216 or record['bad']: raise ValueError('incomplete reversal probe')
    if a.device == 1 and record['uuid'] != '8a67b1f8ef1c3177a822813a7ac2224d': raise ValueError('GPU1 identity differs')
    if freeze(exe) != identity or sha(source) != identity['sources'][relative]: raise ValueError('identity changed')
    (out / 'collector.py').write_bytes(Path(__file__).read_bytes())
    data = dict(complete=True, identity=identity, tool_sha256=sha(__file__), kernel_source_sha256=sha(source),
                probe_source_sha256=sha(probe), probe_binary_sha256=sha(binary), result=record,
                scope='Exact frozen kernel body; padding, zero data, word permutation and source/output guards. Not the full ECM arithmetic or production resource profile.')
    (out / 'summary.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__': main()
