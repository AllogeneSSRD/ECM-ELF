"""Extract the pinned ELYTerminal vector rig and prepare an offline frame atlas.

No browser or SVG parser ships in the application. Run the paired Node renderer
only when rebuilding this checked-in asset. Shapes stay attributed to upstream.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[2]


def arguments(text):
    values, start, depth = [], 0, 0
    for index, char in enumerate(text):
        if char in '([{': depth += 1
        if char in ')]}': depth -= 1
        if char == ',' and depth == 0:
            values.append(text[start:index].strip()); start = index + 1
    values.append(text[start:].strip())
    return values


def layers(source):
    region = source.split('const PARTS: &[Part] = &[', 1)[1].split('\n];', 1)[0]
    result = []
    for line in region.strip().splitlines():
        strings = []
        def quote(match):
            strings.append(match[1] if match[1] is not None else json.loads(match[0]))
            return f'@{len(strings)-1}@'
        simple = re.sub(r'r#"(.*?)"#|"(?:\\.|[^"\\])*"', quote, line.strip().rstrip(','))
        def string(value): return ''.join(strings[int(i)] for i in re.findall(r'@(\d+)@', value))
        def vector(value): return [float(x) for x in value.strip('[]() ').split(',')]
        if simple.startswith('Part {'):
            fields = dict(item.split(':', 1) for item in arguments(simple[6:-1]) if item)
            fields = {key.strip(): value.strip() for key, value in fields.items()}
            name, color, frame, pivot, bone, shapes = (fields[k] for k in ['name','color','frame','pivot','bone','shapes'])
            alpha = float(fields.get('alpha', 1))
        else:
            kind, rest = simple.split('(', 1)
            args = arguments(rest[:-1]); name, color = args[:2]; shapes = args[-1]; alpha = 1
            pivot = '(256,236)'
            if kind == 'part': frame, bone = args[2:4]
            elif kind == 'eye':
                x, side = float(args[2]), int(args[3]); frame = str([x-19,164,38,48]); pivot = str([x,192]); bone = f'Bone::Iris({side})'
            elif kind == 'mouth': frame = args[2]; bone = 'Bone::Mouth(' + args[3] + ')'; pivot = '(256,224)'
            elif kind == 'mark': frame, pivot = args[2:4]; bone = 'Bone::Mark(' + args[4] + ')'
            else: raise ValueError(kind)
        result.append(dict(name=string(name),color=int(color,16),frame=vector(frame),pivot=vector(pivot),bone=bone,shapes=string(shapes),alpha=alpha))
    assert len(result) > 40 and len({p['name'] for p in result}) == len(result)
    return result


def frame(parts, mood, index):
    phase = index * math.tau / 8
    bob = 3 * math.sin(phase); tilt = math.sin(phase) * (2 if mood in ['Working','Waiting','Error'] else .8)
    chunks = []
    for part in parts:
        bone = part['bone']; transform = ''
        eyes = 'Arcs' if mood == 'Done' else 'Lines' if mood == 'Doze' else 'Wink' if mood == 'Wink' else 'Open'
        if 'Iris' in bone or 'Lid' in bone:
            if eyes in ['Arcs','Lines'] or (eyes == 'Wink' and '(0)' in bone): continue
            if index == 7 and eyes == 'Open':
                x,y = part['pivot']; transform += f'translate({x} {y}) scale(1 .15) translate({-x} {-y}) '
        if 'Arc' in bone and not (eyes == 'Arcs' or (eyes == 'Wink' and '(0)' in bone)): continue
        if 'Lines' in bone and eyes != 'Lines': continue
        if 'Brows' in bone and mood not in ['Waiting','Error']: continue
        mouth = 'Grin' if mood == 'Done' else 'Frown' if mood == 'Error' else 'Flat' if mood == 'Doze' else 'Smile'
        if 'Mouth' in bone and ('Mouth::' + mouth) not in bone: continue
        if 'Mark' in bone:
            permitted = ('Bang' in bone and mood == 'Waiting') or ('Drop' in bone and mood == 'Error') or ('Sparkle' in bone and mood == 'Done') or ('Heart' in bone and mood == 'Wink') or ('Z' in bone and mood == 'Doze')
            if not permitted: continue
        if not any(x in bone for x in ['Body','Hand']): transform = f'translate(0 {-bob:.3f}) rotate({tilt:.3f} 256 236) ' + transform
        if 'Hand' in bone and mood == 'Working': transform = f'translate(0 {abs(math.sin(phase)) * -3:.3f})'
        color = f'#{part["color"]:06x}'
        shapes = part['shapes'].replace('"black"', f'"{color}"')
        chunks.append(f'<g fill="{color}" opacity="{part["alpha"]}" transform="{transform}">{shapes}</g>')
    return '<svg xmlns="http://www.w3.org/2000/svg" width="224" height="248" viewBox="110 -36 292 323">' + ''.join(chunks) + '</svg>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'third_party/gui_v2_pet/rig.rs'
    parts = layers(source.read_text(encoding='utf-8'))
    moods = ['Calm','Working','Waiting','Error','Done','Doze','Wink']
    svgs = [frame(parts, mood, index) for mood in moods for index in range(8)]
    html = '<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;background:transparent}main{display:grid;grid-template-columns:repeat(8,224px);width:1792px}svg{display:block}</style></head><body><main>' + ''.join(svgs) + '</main></body></html>'
    (args.out / 'atlas.html').write_text(html, encoding='utf-8')
    (args.out / 'metadata.json').write_text(json.dumps(dict(columns=8,rows=7,frame_width=224,frame_height=248,logical_width=112,logical_height=124,moods=moods,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),layers=len(parts)),indent=2)+'\n',encoding='utf-8')
    print(f'Prepared {len(parts)} attributed layers / {len(svgs)} frames')


if __name__ == '__main__': main()
