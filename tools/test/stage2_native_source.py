"""Read actual Stage2 host modules for source and CPU allocation proofs.

This is test instrumentation only. Production builds compile complete modules.
"""
import re
from pathlib import Path
from test_cuda_p2_structure import mask, balanced

ROOT = Path(__file__).resolve().parents[2]
STAGE2 = ROOT/'src/cuda/stage2'


def units(text):
    clean = mask(text); i = 0
    while i < len(text):
        while i < len(text) and clean[i].isspace(): i += 1
        if i == len(text): break
        start = i
        if clean[i] == '#':
            end = clean.index('\n', i)+1
            while clean[end-2] == '\\': end = clean.index('\n', end)+1
            yield 'pp', start, end, text[start:end]; i = end; continue
        paren = bracket = 0
        while i < len(text):
            c = clean[i]
            if c == '(': paren += 1
            elif c == ')': paren -= 1
            elif c == '[': bracket += 1
            elif c == ']': bracket -= 1
            if not paren and not bracket:
                if c == ';':
                    i += 1; yield 'decl', start, i, text[start:i]; break
                if c == '{':
                    end = balanced(clean, i, '{', '}')+1; lead = clean[start:i]
                    if re.match(r'(?:struct|class|enum)\b', lead):
                        end = clean.index(';', end)+1; kind = 'type'
                    elif re.match(r'namespace\b', lead): kind = 'namespace'
                    elif not re.search(r'\)\s*(?:const\s*)?$', lead):
                        i = end; continue
                    else: kind = 'func'
                    yield kind, start, end, text[start:end]; i = end; break
            i += 1
        else: raise ValueError('unparsed declaration: '+text[start:start+100])


def signature(text):
    clean = mask(text); opening = clean.index('(')
    end = balanced(clean, opening, '(', ')')+1
    name = re.search(r'((?:\w+::)*(?:~?\w+|operator=))\s*$', clean[:opening])[1]
    body = clean.index('{', end)
    return name, opening, end, body


def class_declaration(text):
    """Replace inline method bodies with declarations, retaining fields/defaults."""
    clean = mask(text); start = clean.index('{'); end = balanced(clean, start, '{', '}')
    edits = []; inner = text[start+1:end]
    for kind, a, b, value in units(inner):
        if kind == 'type' and re.match(r'struct\s+\w+', mask(value)):
            edits.append((a, b, class_declaration(value)))
        elif kind == 'func':
            _, _, ep, bp = signature(value); head = value[:bp]
            if ':' in mask(head)[ep:]: head = head[:ep]
            edits.append((a, b, head.rstrip()+';'))
    for a, b, value in sorted(edits, reverse=True): inner = inner[:a]+value+inner[b:]
    return text[:start+1]+inner+text[end:end+1]+';'


def reconstruct_class(declaration, implementations):
    """Reinsert the actual out-of-line methods into one CPU fixture class."""
    cls = re.match(r'struct\s+(\w+)', mask(declaration))[1]
    methods = {}
    for kind, _, _, value in units(implementations):
        if kind != 'func': continue
        name, _, _, bp = signature(value)
        if name.startswith(cls+'::'):
            methods[name.split('::')[-1]] = value[bp:]
    clean = mask(declaration); edits = []
    for name, body in methods.items():
        for match in re.finditer(r'(?<![\w~])'+re.escape(name)+r'\s*\(', clean):
            end = balanced(clean, clean.index('(', match.start()), '(', ')')+1
            end += re.match(r'\s*(?:const\s*)?', clean[end:]).end()
            if clean[end] == ';': edits.append((end, end+1, body)); break
    for a, b, value in sorted(edits, reverse=True): declaration = declaration[:a]+value+declaration[b:]
    return declaration


def module_text(*names):
    return '\n'.join((STAGE2/name).read_text(encoding='utf-8-sig') for name in names)


def native_view():
    """Test-only view of real fields/methods for the existing CPU CUDA doubles.

    Every body comes from a complete current production module. No frozen or
    handwritten allocation implementation substitutes for the current source.
    """
    paths = sorted(p for p in STAGE2.iterdir() if p.suffix in ('.h','.cpp'))
    implementations = {}; free = []
    for path in paths:
        if path.suffix != '.cpp': continue
        for kind, _, _, value in units(path.read_text(encoding='utf-8-sig')):
            if kind == 'func':
                name, _, ep, bp = signature(value)
                if '::' in name: implementations[name] = (value[ep:bp],value[bp:])
                else: free.append(value)
    def restore(value, parent=''):
        clean = mask(value); cls = re.match(r'struct\s+(\w+)', clean)[1]
        prefix = parent+cls+'::'; opened = clean.index('{'); closed = balanced(clean,opened,'{','}')
        inner = value[opened+1:closed]; edits = []
        for kind,a,b,item in units(inner):
            if kind == 'type' and re.match(r'struct\s+\w+',mask(item)):
                edits.append((a,b,restore(item,prefix)))
            elif kind == 'decl' and '(' in mask(item):
                c = mask(item); op = c.index('('); ep = balanced(c,op,'(',')')+1
                found = re.search(r'(~?\w+)\s*$',c[:op])
                if found and prefix+found[1] in implementations and '=' not in c[ep:]:
                    suffix,body = implementations[prefix+found[1]]
                    if '\n' in body:
                        lines = body.splitlines()
                        body = '\n'.join([lines[0]]+['    '+line if line else '' for line in lines[1:]])
                    # Constructor initializers live in the definition; qualifiers
                    # for other methods already live in their declaration.
                    init = suffix if ':' in mask(suffix) else ''
                    edits.append((a,b,item.rstrip().removesuffix(';')+init+body))
        for a,b,item in sorted(edits,reverse=True): inner = inner[:a]+item+inner[b:]
        return value[:opened+1]+inner+value[closed:]
    classes = []
    for path in paths:
        if path.suffix != '.h': continue
        for kind,_,_,value in units(path.read_text(encoding='utf-8-sig')):
            if kind == 'type' and re.match(r'struct\s+\w+',mask(value)): classes.append(restore(value))
    return '\n\n'.join(classes+free), paths
