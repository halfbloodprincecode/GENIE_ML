import re
import importlib

def Import(path, fname=None):
    path = re.sub(r'/+', '/', path).rstrip('/')
    fname = path.split('/')[-1].replace('.py', '')
    try:
        spec = importlib.util.spec_from_file_location(fname, path)
        f = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(f)
        return f
    except Exception as e:
        print(f'ERROR: {e}')