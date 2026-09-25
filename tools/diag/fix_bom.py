# fix_bom.py -- restore the UTF-8 BOM on files that contain non-ASCII bytes.
# nvcc/cl read a .cu without a BOM as ANSI(GBK); a Chinese comment then swallows the
# following newline and the next line (often a #define) disappears into the comment.
# See docs/ECM_CGBN_OPTIMIZATION.md section 6 item 1.
import io, os, sys

paths = sys.argv[1:]
for p in paths:
    raw = open(p, 'rb').read()
    has_bom = raw.startswith(b'\xef\xbb\xbf')
    non_ascii = any(b > 127 for b in raw)
    if not non_ascii:
        print('ASCII-only, no BOM needed : %s' % p)
        continue
    if has_bom:
        print('already has BOM           : %s' % p)
        continue
    body = raw.decode('utf-8')
    io.open(p, 'w', encoding='utf-8-sig', newline='').write(body)
    print('BOM restored              : %s' % p)
