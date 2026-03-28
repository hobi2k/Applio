import os
import sys

# applio/ 서브디렉토리를 sys.path에 추가 (rvc.*, tabs.* 임포트용)
_APPLIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "applio")
if _APPLIO_DIR not in sys.path:
    sys.path.insert(0, _APPLIO_DIR)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
