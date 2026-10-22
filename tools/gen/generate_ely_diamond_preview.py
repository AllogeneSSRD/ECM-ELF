"""Prepare web-only hollow diamond outlines without touching the native registry."""
from pathlib import Path
import argparse,hashlib,json,xml.etree.ElementTree as ET
from generate_gui_v2_pet import layers,frame
from generate_gui_v2_pet_skins import catalog,compose,MOODS

ROOT=Path(__file__).resolve().parents[2]
RECIPE=ROOT/'assets/web/ely_variants/diamond-pupils.svg'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    spec=catalog();source=ROOT/'third_party/gui_v2_pet/rig.rs'
    base=layers(source.read_text(encoding='utf-8'))
    overrides={g.attrib['data-part-suffix']:''.join(ET.tostring(n,encoding='unicode') for n in g) for g in ET.parse(RECIPE).getroot()}
    assert set(overrides)=={'pupil-crystal-l','pupil-crystal-r'}
    for skin in spec['skins']:
        parts=compose(base,spec,skin);replaced=[]
        for part in parts:
            suffix=part['name'].removeprefix(skin['id']+'-')
            if suffix in overrides:
                part['shapes']=overrides[suffix];replaced.append(suffix)
        assert set(replaced)==set(overrides)
        styles=';'.join('--'+key+':'+value for key,value in skin['palette'].items())
        svgs=[frame(parts,mood,i).replace(' width="224"',f' style="{styles}" width="224"',1) for mood in MOODS for i in range(8)]
        html='<!doctype html><meta charset="utf-8"><style>html,body{margin:0;background:transparent}main{display:grid;grid-template-columns:repeat(8,224px);width:1792px}svg{display:block}</style><main>'+''.join(svgs)+'</main>'
        out=args.out/skin['id'];out.mkdir()
        (out/'atlas.html').write_text(html,encoding='utf-8',newline='\n')
        inputs=[source,RECIPE,Path(__file__),ROOT/'tools/gen/generate_gui_v2_pet.py',ROOT/'tools/gen/generate_gui_v2_pet_skins.py',ROOT/'tools/gen/render_gui_v2_pet.cjs',ROOT/'assets/gui_v2/skins/manifest.json',ROOT/'assets/gui_v2/skins'/spec['shared_svg'],ROOT/'assets/gui_v2/skins'/skin['svg']]
        metadata=dict(web_demo_only=True,variant='hollow_light_diamond_pupils',pupil_fill='none',pupil_stroke='iris-glow',skin=skin['id'],english=skin['english'],columns=8,rows=7,frame_width=224,frame_height=248,moods=MOODS,modified_parts=sorted(replaced),source_files_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},provenance='Web-only hollow light pupil outlines over the unchanged iris. ELYTerminal MIT face/animation attribution retained; no native registry or runtime atlas changes.')
        (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Prepared three web-only diamond-pupil sheets; native registry untouched')

if __name__=='__main__':main()
