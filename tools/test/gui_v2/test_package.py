"""Relocated package smoke, without SDK/layers/build output on PATH."""
import argparse,ctypes,hashlib,json,os,shutil,struct,subprocess,time
from pathlib import Path
from win32_probe import window,close,user
from renderer_probe import renderer_clean
def imports(path):
    data=path.read_bytes();pe=struct.unpack_from('<I',data,0x3c)[0];assert data[pe:pe+4]==b'PE\0\0'
    sections=struct.unpack_from('<H',data,pe+6)[0];optional_size=struct.unpack_from('<H',data,pe+20)[0];optional=pe+24
    assert struct.unpack_from('<H',data,pe+4)[0]==0x8664 and struct.unpack_from('<H',data,optional)[0]==0x20b
    table=optional+optional_size
    def offset(rva):
        for index in range(sections):
            virtual_size,base,raw_size,raw=struct.unpack_from('<IIII',data,table+40*index+8)
            if base<=rva<base+max(virtual_size,raw_size):return raw+rva-base
        raise AssertionError(f'Unmapped PE RVA {rva}')
    rva=struct.unpack_from('<I',data,optional+120)[0];result=[];cursor=offset(rva)
    while any(data[cursor:cursor+20]):
        start=offset(struct.unpack_from('<I',data,cursor+12)[0]);result.append(data[start:data.index(0,start)].decode('ascii'));cursor+=20
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir();source=a.package.resolve();package=out/'relocated-中文';shutil.copytree(source,package);exe=package/'ecm_gui_v2.exe'
    manifest=json.loads((package/'manifest.json').read_text());assert not manifest['contains_compute_workers']
    for name,digest in manifest['files_sha256'].items():assert hashlib.sha256((package/name).read_bytes()).hexdigest()==digest,name
    modules=imports(exe);assert 'd3d11.dll' in [name.lower() for name in modules]
    assert not any(name.lower() in ['vulkan-1.dll','opencl.dll','webview2loader.dll'] for name in modules)
    assert not list(package.rglob('vulkan-1.dll')) and not list(package.rglob('*validation*.dll'))
    windows=ctypes.create_unicode_buffer(32768);assert ctypes.WinDLL('kernel32').GetWindowsDirectoryW(windows,len(windows))
    env=os.environ.copy();env['PATH']=str(Path(windows.value)/'System32');env.pop('VULKAN_SDK',None);env.pop('VK_LAYER_PATH',None);env.pop('VK_INSTANCE_LAYERS',None)
    assert manifest['group_order_helper'] and not manifest['pari_gp_redistributed']
    helper=package/'ecm_group_order_ui.exe';assert 'gmp-10.dll' in [name.lower() for name in imports(helper)]
    request=out/'order-request.json';response=out/'order-response.json'
    request.write_text(json.dumps(dict(factor='1009',sigma='26',profile=0,backend=2,gp='')),encoding='utf-8')
    subprocess.run([str(helper),str(request),str(response)],cwd=out,env=env,check=True,timeout=15)
    order=json.loads(response.read_text());assert order['status']=='complete' and order['backend']=='pari' and order['point_minimality']
    if manifest.get('native_pari_library'):
        assert order['native_engine']=='pari' and order['all_proven']
        request.write_text(json.dumps(dict(factor='6277101735386680763835789423207666416083908700390324961279',sigma='26',profile=0,backend=2,gp='')),encoding='utf-8')
        subprocess.run([str(helper),str(request),str(response)],cwd=out,env=env,check=True,timeout=8)
        order=json.loads(response.read_text())
        assert order['status']=='complete' and order['native_engine']=='pari' and order['all_proven']
        assert order['group_order']=='6277101735386680763835789423165887359574653668381181441896'
        assert 'libpari.dll' not in [name.lower() for name in modules],'Math library must stay outside the GUI process'
    startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    cases={};cfg=out/'workflow.toml';cfg.write_text('[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n');original=cfg.read_bytes()
    def run(name,flags,expected=0,show=False,binary=exe):
        log=out/(name+'.log');cmd=[str(binary),'--config',str(cfg),'--ui-state',str(out/(name+'.toml')),'--diagnostics',str(log),'--frames','12',*flags]
        proc=subprocess.Popen(cmd,cwd=out,env=env,startupinfo=startup,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if show:hwnd=window(proc.pid,'ecm_gui_v2');user.ShowWindow(hwnd,4)
        proc.wait(timeout=30);assert proc.returncode==expected,(name,proc.returncode);text=log.read_text(encoding='utf-8')
        assert renderer_clean(text,'dx11') if expected==0 else 'No Vulkan loader available' in text
        cases[name]=dict(exit=proc.returncode);return text
    assert 'frame: 1' in run('normal',[])
    unicode_config=out/'中文工作流.toml';unicode_config.write_bytes(cfg.read_bytes());unicode_log=out/'unicode.log'
    run_unicode=subprocess.run([str(exe),'--config',str(unicode_config),'--ui-state',str(out/'unicode-ui.toml'),'--diagnostics',str(unicode_log),'--frames','8','--no-nvml'],cwd=out,env=env,startupinfo=startup,timeout=30)
    assert run_unicode.returncode==0 and 'app: config=' in unicode_log.read_text(encoding='utf-8');cases['unicode-workflow-read']=True
    assert 'No Vulkan loader available' in run('no-loader',['--renderer','vulkan','--no-loader'],1)
    assert 'language=en cjk=0' in run('missing-cjk',['--language','zh','--simulate-missing-cjk'])
    assert 'worker_processes=0' in run('no-nvml',['--no-nvml'])
    assert 'pet: mood=' in run('pet',['--pet'],show=True)
    for skin in manifest.get('pet_skins',[])[1:]:
        assert f'pet_skin: id={skin} ' in run('pet-'+skin,['--pet','--pet-skin',skin],show=True)
    no_pet=out/'without-pet';no_pet.mkdir();shutil.copy2(exe,no_pet/exe.name)
    assert 'pet_error: Pet atlas missing' in run('missing-pet',['--pet'],show=True,binary=no_pet/exe.name)
    corrupt=out/'corrupt.toml';corrupt.write_text('[state]\nschema=999\n');before=corrupt.read_bytes()
    log=out/'invalid-state.log';result=subprocess.run([str(exe),'--ui-state',str(corrupt),'--diagnostics',str(log),'--frames','8','--no-nvml'],cwd=out,env=env,startupinfo=startup,timeout=30)
    assert result.returncode==0 and corrupt.read_bytes()==before and 'UI state not loaded' in log.read_text(encoding='utf-8');cases['invalid-state-retained']=True
    corrupt.write_text('[ui]\ndock="0"\n');before=corrupt.read_bytes()
    result=subprocess.run([str(exe),'--ui-state',str(corrupt),'--diagnostics',str(out/'invalid-layout.log'),'--frames','8','--no-nvml'],cwd=out,env=env,startupinfo=startup,timeout=30)
    assert result.returncode==0 and corrupt.read_bytes()==before;cases['invalid-layout-retained']=True
    # A persisted file is actually parsed again, not merely left on disk.
    state=out/'normal.toml';state_before=state.read_text(encoding='utf-8');assert '[ui]' in state_before
    log=out/'restart.log';result=subprocess.run([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--frames','8','--no-nvml'],cwd=out,env=env,startupinfo=startup,timeout=30)
    assert result.returncode==0 and 'UI state not loaded' not in log.read_text(encoding='utf-8') and 'UI state not saved' not in log.read_text(encoding='utf-8');cases['restart']=True
    assert cfg.read_bytes()==original
    report=dict(passed=True,cases=cases,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),manifest_files_verified=len(manifest['files_sha256']),pe_imports=modules,business_unchanged=True,path_only_system32=True,limits=['Same machine with installed graphics driver and MSVC runtime; not a separate clean OS','No actual driver removal; missing-loader/CJK are explicit fault seams'])
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
