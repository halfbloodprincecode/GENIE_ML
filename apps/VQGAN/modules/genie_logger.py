from utils.ptLogger import GenieLoggerBase
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

class GenieLogger(GenieLoggerBase):
    def __init__(self, save_dir: str, name: Optional[str] = 'GeineLogs', agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None, agg_default_func: Optional[Callable[[Sequence[float]], float]] = None, **kwargs: Any):
        super().__init__(save_dir, name, agg_key_funcs, agg_default_func, **kwargs)