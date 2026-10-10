"""CPU-only target and oracle scope checks for the Stage1 cost producer."""
import argparse,copy,importlib.util,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('tool',ROOT/'tools/bench/tune_stage1_cost.py')
    tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    accepted=rejected=0
    for text,n in [('5',5),('00005',5),('0X11',17),('0x'+'f'*4096,(1<<16384)-1),
        (str((1<<16384)-1),(1<<16384)-1)]:
        assert tool.target_integer(text)==n;accepted+=1
    for text in ['','0','3','4','-5','+5','1e10','2^521-1',' 5','5 ','0x','0x-5','0xffg','5;cmd',
        '1'+'0'*4933,str((1<<16384)+1),True]:
        try:tool.target_integer(text)
        except ValueError:rejected+=1
        else:raise AssertionError(('bad literal accepted',text))
    known=tool.benchmark_targets([521,2203],None)
    assert known==[(521,(1<<521)-1,'known_mersenne_prime',521),
        (2203,(1<<2203)-1,'known_mersenne_prime',2203)];accepted+=1
    assert tool.benchmark_targets([],['17','31'])==[(5,17,'validated_no_factor_stage1',0),
        (5,31,'validated_no_factor_stage1',0)];accepted+=1
    for values in [['17','19'],['31','0x1f'],['17','4']]:
        try:tool.benchmark_targets([],values)
        except ValueError:rejected+=1
        else:raise AssertionError('duplicate/invalid target scope accepted')
    n=1000000007*1000000009;bits=n.bit_length()
    rows=[dict(type='stage1_reference',bits=bits,b1=20,sigma_first=40,curves=2,exponent='lcm',
        n_hex=format(n,'x'),scalar_bits=28),dict(type='point',sigma=40,x_hex='0'),
        dict(type='point',sigma=41,x_hex='1'),dict(type='complete',curves=2,algorithm='plain_gmp_ladder',seconds=.1)]
    text='\n'.join(json.dumps(row) for row in rows)
    assert tool.read_native_reference(text,bits,20,40,2,'lcm',n)==[0,1];accepted+=1
    for changed in [n+2,(1<<bits)-1]:
        try:tool.read_native_reference(text,bits,20,40,2,'lcm',changed)
        except ValueError:rejected+=1
        else:raise AssertionError('same width different N accepted')
    for section,key,value in [(0,'n_hex',format(n+2,'x')),(0,'sigma_first',26),
        (1,'sigma',26),(2,'x_hex',format(n,'x')),(-1,'curves',1)]:
        bad=copy.deepcopy(rows);bad[section][key]=value
        try:tool.read_native_reference('\n'.join(json.dumps(row) for row in bad),bits,20,40,2,'lcm',n)
        except ValueError:rejected+=1
        else:raise AssertionError('bad generic oracle accepted')
    # CLI grid failures occur before reading executable paths or touching the GPU.
    common=[sys.executable,str(ROOT/'tools/bench/tune_stage1_cost.py'),'--stage1','missing1.exe',
        '--stage2','missing2.exe','--device','1','--output',str(out/'unused'),'--profile',str(out/'unused.toml')]
    cases=[['--target-n','17','--exponents','521'],['--target-n','17','19'],['--target-n','4'],
        ['--target-n','17','--sigma-first','5'],['--target-n','17','--sigma-first','9007199254740991','--batch','8']]
    for i,args in enumerate(cases):
        proc=subprocess.run(common+args,capture_output=True,text=True,timeout=20)
        assert proc.returncode==2 and not (out/'unused').exists(),(args,proc.stderr)
        (out/f'cli_reject_{i}.log').write_text(proc.stdout+proc.stderr,encoding='utf-8');rejected+=1
    result=dict(passed=True,accepted=accepted,rejected=rejected,gpu_access=False)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
