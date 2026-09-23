import subprocess, sys, re

FILES = ["src/cpu/ecm_edwards_cpu.cpp", "src/cpu/simd_edwards.cpp", "src/cpu/ecm_edwards_mont.h",
         "src/cpu/simd_mont_ifma.cpp", "src/cpu/ecm_edwards_cpu.h"]

def git_show(path):
    try:
        out = subprocess.run(["git", "show", "HEAD:" + path], capture_output=True, check=True)
        return out.stdout
    except Exception as e:
        return None

def report(tag, raw):
    if raw is None:
        print("  %-12s : (not in HEAD)" % tag)
        return
    try:
        txt = raw.decode("utf-8")
        ok = True
    except UnicodeDecodeError as e:
        txt = raw.decode("utf-8", "replace")
        ok = False
    # mojibake signature: CJK chars typical of utf8-decoded-as-gbk garbage
    sig = len(re.findall(r"[\u9225-\u9fff\u9518\u9526\u951b\u9422\u7f01-\u7f2f]", txt))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", txt))
    # reverse transcode: text.encode(gbk) -> should give the original utf-8 bytes
    rev_ok = False
    rev_cjk = 0
    try:
        rev = txt.encode("gbk", "strict").decode("utf-8", "strict")
        rev_cjk = len(re.findall(r"[\u4e00-\u9fff]", rev))
        rev_ok = True
    except Exception:
        rev_ok = False
    print("  %-12s : utf8_valid=%s  cjk=%d  mojibake_sig=%d  reverse_transcode_ok=%s (cjk=%d)"
          % (tag, ok, cjk, sig, rev_ok, rev_cjk))

for f in FILES:
    print(f)
    with open(f, "rb") as fh:
        disk = fh.read()
    report("disk", disk)
    report("HEAD", git_show(f))
