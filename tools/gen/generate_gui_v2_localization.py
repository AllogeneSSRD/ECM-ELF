"""Generate legacy localization-key aliases and check readable default XML keys."""
from pathlib import Path
import argparse,hashlib,json,re,xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
ALIASES='config/gui_v2_localization_aliases.json'
HEADER='src/ecm/application/gui_v2/generated/localization_aliases.h'
RECEIPT='config/gui_v2_localization.generated.json'
LANGUAGES=['src/ecm/application/gui_v2/localization/'+name+'.xml' for name in ['english','chineseSimplified']]

def products():
    aliases=json.loads((ROOT/ALIASES).read_text(encoding='utf-8'))['aliases']
    sets=[]
    for name in LANGUAGES:
        items=ET.parse(ROOT/name).findall('.//Item');keys=[item.get('id') for item in items]
        assert len(keys)==len(set(keys)),f'Duplicate keys in {name}'
        for key in keys:
            assert re.fullmatch(r'[a-z][a-z0-9]*(?:[_.][a-z][a-z0-9]*)*',key),f'Unreadable key: {key}'
            assert not re.search(r'_[0-9a-f]{8}$|^message_\d+$',key),f'Legacy key in default XML: {key}'
        assert not any('\ufffd' in item.get('name','') for item in items),name
        sets.append(set(keys))
    assert sets[0]==sets[1],'Default XML keys must match'
    assert len(set(aliases.values()))==len(aliases),'Legacy aliases must map to distinct default keys'
    assert set(aliases.values())<=sets[0],'Alias points to a missing default key'
    for path in (ROOT/'src/ecm/application/gui_v2').rglob('*'):
        if path.suffix not in ['.h','.cpp'] or path.relative_to(ROOT).as_posix()==HEADER:continue
        source=path.read_text(encoding='utf-8')
        quoted=set(re.findall(r'"([^"\n]*)"',source))
        assert not quoted.intersection(aliases),f'Legacy source reference in {path}'
        for key in re.findall(r'\btext\("([a-zA-Z0-9_.]+)"',source):
            assert key in sets[0],f'Missing default key {key} in {path}'
    lines=['// Generated; edit config/gui_v2_localization_aliases.json.','#pragma once','namespace ecmgui_v2::ui {',
           'struct LocalizationAlias { const char* legacy; const char* readable; };','inline constexpr LocalizationAlias localization_aliases[]={']
    for old,new in sorted(aliases.items()):lines.append('    {'+json.dumps(old)+','+json.dumps(new)+'},')
    lines+=['};','}']
    content='\n'.join(lines)+'\n'
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [ALIASES,*LANGUAGES,Path(__file__).relative_to(ROOT).as_posix()]}
    hashes[HEADER]=hashlib.sha256(content.encode()).hexdigest()
    return {HEADER:content,RECEIPT:json.dumps({'sha256':hashes},indent=2)+'\n'}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    for name,content in products().items():
        path=ROOT/name
        if args.check:assert path.read_bytes()==content.encode(),f'Stale localization definitions: {name}'
        else:path.write_text(content,encoding='utf-8',newline='\n')
    print('GUI v2 localization keys current' if args.check else 'GUI v2 localization aliases generated')

if __name__=='__main__':main()
