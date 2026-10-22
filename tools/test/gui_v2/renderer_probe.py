"""Backend identity checks; DX11 runtime success is not Vulkan API validation."""
def renderer_args(backend):
    return ['--renderer',backend]+(['--validation'] if backend=='vulkan' else [])

def renderer_clean(text,backend):
    if backend=='vulkan':
        return 'backend=Vulkan' in text and 'validation_summary: errors=0 warnings=0' in text
    return ('backend=DX11' in text and 'renderer_summary: backend=DX11' in text
            and 'shutdown=complete' in text and '\nerror:' not in text
            and 'instance: loader=' not in text and 'validation_summary:' not in text)
