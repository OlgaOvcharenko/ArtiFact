import itertools
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Global counter for rotating sessions
_request_counter = itertools.count()
ROTATE_EVERY = 25

def new_session() -> requests.Session:
    """Create a new resilient requests session with retries."""
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": "artwork-sampling/0.1 (ex@mple.com)",
        "Accept": "application/json",
        "Connection": "close",
    })
    return s

class SessionManager:
    """Manages an HTTP session and automatically rotates it after ROTATE_EVERY requests."""
    def __init__(self, rotate_every: int = ROTATE_EVERY):
        self.session = new_session()
        self.counter = itertools.count()
        self.rotate_every = rotate_every

    def get(self, *args, **kwargs) -> requests.Response:
        i = next(self.counter)
        if i > 0 and i % self.rotate_every == 0:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = new_session()
            
        return self.session.get(*args, **kwargs)
