import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_SESSION = None

def _new_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": "rijks-harvester/0.1 (ex@mple.com)",
        "Connection": "close",
    })
    return s

def get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _new_session()
    return _SESSION

def reset_session() -> None:
    global _SESSION
    try:
        if _SESSION is not None:
            _SESSION.close()
    except Exception:
        pass
    _SESSION = _new_session()