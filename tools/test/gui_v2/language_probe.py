"""Choose an actual filename from the external XML dropdown in an isolated UI."""
import re,time
from win32_probe import user

def select_language(hwnd,log,filename,accepted=True):
    def latest():
        rows=[line for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('language_sample:')]
        return rows[-1] if rows else ''
    def wait(pattern):
        end=time.monotonic()+12
        while time.monotonic()<end:
            line=latest();match=re.search(pattern,line)
            if match:return match
            time.sleep(.06)
        raise AssertionError((pattern,latest()))
    def click(value):
        x,y=map(int,value.split(','));assert x>0 and y>0,value
        for action in [0,1,2]:user.PostMessageW(hwnd,0x8021,action,(y<<16)|(x&65535));time.sleep(.14)
    click(wait(r'combo=([1-9]\d*,[1-9]\d*)').group(1))
    click(wait(r'item\['+re.escape(filename)+r'\]=([1-9]\d*,[1-9]\d*)').group(1))
    if accepted:wait(r'file='+re.escape(filename)+r' loaded=')
