"""Independent decoder for the Stage1 batch checkpoint used by regression gates."""
def batch(path):
    raw = path.read_bytes()
    body, tail = raw.rsplit(b'CHECKSUM ', 1)
    h = 1469598103934665603
    for byte in body:
        h = ((h ^ byte)*1099511628211) & ((1 << 64)-1)
    assert int(tail) == h
    import io
    stream = io.BytesIO(body)
    assert stream.readline() == b'ECM_STAGE1_BATCH_V1\n'
    identity = stream.readline().decode().strip()
    count = int(stream.readline())
    rows = []
    for _ in range(count):
        sigma, size = map(int, stream.readline().split())
        payload = stream.read(size)
        assert len(payload) == size and stream.read(1) == b'\n'
        rows.append((sigma, payload))
    assert not stream.read()
    return identity, rows


def progressed(rows):
    import re
    for _, payload in rows:
        if payload.startswith(b'MPA-MONT-CKPT '):
            at=re.search(rb'(?m)^BITNUM=(\d+)$',payload)
            if b'STATUS=DONE\n' in payload or (at and int(at[1])>0):return True
        elif payload:
            return True
    return False
