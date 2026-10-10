"""Cross-check the independent native Stage1 oracle against plain Python."""
import argparse,copy,hashlib,importlib.util,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exe=a.reference.resolve();digest=hashlib.sha256(exe.read_bytes()).hexdigest()
    python=module('python',ROOT/'tools/stat/suyama_mont_ref.py');tool=module('tool',ROOT/'tools/bench/tune_stage1_cost.py')
    scalar_cases=point_cases=generic_points=bad_cases=0
    for b1 in [2,3,4,8,9,25,27,31,49,64,65,999,1000,10000,1048578]:
        proc=subprocess.run([str(exe),'--scalar',str(b1)],capture_output=True,text=True,timeout=120)
        assert proc.returncode==0,proc.stderr
        assert int(proc.stdout,16)==python.lcm_1_to(b1),b1
        (out/f'scalar_{b1}.json').write_text(json.dumps(dict(b1=b1,bits=int(proc.stdout,16).bit_length(),matched=True)))
        scalar_cases+=1
    for bits,b1,count in [(p,20,2) for p in tool.PRIMES]+[(521,1000,8),(4423,1000,2),(521,10000,2)]:
        for mode in ['lcm','choose12']:
            proc=subprocess.run([str(exe),str(bits),str(b1),'26',str(count),mode],capture_output=True,text=True,timeout=120)
            (out/f'm{bits}_b{b1}_{mode}.jsonl').write_text(proc.stdout,encoding='utf-8');assert proc.returncode==0,proc.stderr
            values=tool.read_native_reference(proc.stdout,bits,b1,26,count,mode)
            n=(1<<bits)-1;s=python.lcm_1_to(b1)*(12 if mode=='choose12' else 1)
            for sigma,value in enumerate(values,26):
                _,a24,x,z=python.suyama_curve(sigma,n);x,z=python.ladder(s,x,z,a24,n)
                assert value==x*pow(z,-1,n)%n,(bits,b1,mode,sigma)
                point_cases+=1
    # Composite targets, alternate first sigma and two different N of equal width.
    for n,b1,first,count in [(1000000007*1000000009,20,40,4),
        (1000000007*1000000033,20,40,4),(((1<<521)-1)*((1<<127)-1),1000,26,2)]:
        for mode in ['lcm','choose12']:
            proc=subprocess.run([str(exe),'--n',format(n,'x'),str(b1),str(first),str(count),mode],
                capture_output=True,text=True,timeout=120)
            assert proc.returncode==0,proc.stderr
            (out/f'n{n.bit_length()}_{n%1000}_{mode}.jsonl').write_text(proc.stdout,encoding='utf-8')
            values=tool.read_native_reference(proc.stdout,n.bit_length(),b1,first,count,mode,n)
            s=python.lcm_1_to(b1)*(12 if mode=='choose12' else 1)
            for sigma,value in enumerate(values,first):
                _,a24,x,z=python.suyama_curve(sigma,n);x,z=python.ladder(s,x,z,a24,n)
                assert value==x*pow(z,-1,n)%n,(n,b1,mode,sigma)
                generic_points+=1
            try:tool.read_native_reference(proc.stdout,n.bit_length(),b1,first,count,mode,n+2)
            except ValueError:bad_cases+=1
            else:raise AssertionError('same-width different modulus accepted')
    proc=subprocess.run([str(exe),'521','20','26','2','lcm'],capture_output=True,text=True,timeout=120)
    rows=[json.loads(line) for line in proc.stdout.splitlines()]
    mutations=[rows[:-1],rows[:-2]+rows[-1:],[],list(reversed(rows))]
    for section,key,value in [(0,'bits',522),(0,'b1',21),(0,'exponent','choose12'),(0,'curves',1),
        (0,'n_hex','3'),(0,'scalar_bits',0),(1,'sigma',27),(1,'x_hex',rows[0]['n_hex']),
        (1,'x_hex','-1'),(-1,'seconds',0),(-1,'algorithm','cuda'),(-1,'curves',1),(-1,'seconds',float('inf'))]:
        bad=copy.deepcopy(rows);bad[section][key]=value;mutations.append(bad)
    for bad in mutations:
        try:tool.read_native_reference('\n'.join(json.dumps(row) for row in bad),521,20,26,2,'lcm')
        except (ValueError,KeyError,TypeError):bad_cases+=1
        else:raise AssertionError('bad oracle accepted')
    for args in [['8191','20','26','1','lcm'],['521','1','26','1','lcm'],['521','20','5','1','lcm'],
        ['521','20','26','0','lcm'],['521','20','26','4097','lcm'],['521','20','26','1','other'],
        ['521','260000001','26','1','lcm'],['-1','20','26','1','lcm'],['--scalar','0']]:
        proc=subprocess.run([str(exe),*args],capture_output=True,text=True,timeout=10)
        assert proc.returncode!=0,args;bad_cases+=1
    for text in ['','-5','0x11','3','4','xyz','1'+'0'*4096]:
        proc=subprocess.run([str(exe),'--n',text,'20','26','1','lcm'],capture_output=True,text=True,timeout=10)
        assert proc.returncode!=0,text;bad_cases+=1
    # A failed inversion must not emit a successful completion, even after a header.
    proc=subprocess.run([str(exe),'--n','f','20','6','1','lcm'],capture_output=True,text=True,timeout=10)
    assert proc.returncode!=0 and '"type":"complete"' not in proc.stdout;bad_cases+=1
    assert hashlib.sha256(exe.read_bytes()).hexdigest()==digest
    result=dict(passed=True,binary_sha256=digest,scalar_cases=scalar_cases,independent_points=point_cases,
        generic_points=generic_points,rejected=bad_cases)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
