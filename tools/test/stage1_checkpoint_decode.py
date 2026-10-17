"""Independent decoder for the Stage1 batch checkpoint used by regression gates."""
def batch(path):
    raw = path.read_bytes()
    if raw.startswith(b'ECMS1B2\0'):
        import struct
        table=[]
        for i in range(256):
            c=i
            for _ in range(8):c=(c>>1)^(0x82f63b78 if c&1 else 0)
            table.append(c)
        def crc(data):
            c=0xffffffff
            for x in data:c=table[(c^x)&255]^(c>>8)
            return c^0xffffffff
        body=raw[:-4];assert crc(body)==struct.unpack('<I',raw[-4:])[0]
        version,count,size=struct.unpack_from('<IIQ',body,8);assert version==2
        at=24;identity=body[at:at+size].decode();at+=size;rows=[]
        for _ in range(count):
            sigma,size,checksum,flags=struct.unpack_from('<QQII',body,at);at+=24
            assert flags==0 and size<=len(body)-at
            payload=body[at:at+size];at+=size
            assert crc(payload)==checksum
            rows.append((sigma,payload))
        assert at==len(body)
        return identity,rows
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
