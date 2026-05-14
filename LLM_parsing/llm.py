import os
import time
import re
import json
import hashlib
from typing import Any, Dict, List, Optional
import concurrent.futures
import litellm
from litellm import batch_completion
from dotenv import load_dotenv

load_dotenv()


MODEL_MAPPING = {
    "g25lite": "gemini/gemini-2.5-flash-lite",
    "g25flash": "gemini/gemini-2.5-flash",
    "g25pro": "gemini/gemini-2.5-pro",
    "v25lite": "vertex_ai/gemini-2.5-flash-lite", # For vertex AI credits
    "v25flash": "vertex_ai/gemini-2.5-flash",
    "v25pro": "vertex_ai/gemini-2.5-pro",
}

def _safe_json_loads(text: Any) -> Dict[str, Any]:
    """
    Strip ```json fences etc. and parse JSON.
    On failure, return {}.
    """
    if not isinstance(text, str):
        return {}

    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {}
    
def _canonical_prompt_string(content: str) -> str:
    """
    Produce a stable canonical representation of the prompt string:
    - Normalize whitespace
    - Strip leading/trailing spaces
    - Normalize newlines
    - Ensure deterministic representation
    """
    if isinstance(content, list):
        try:
            return json.dumps(content, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(content)

    if not isinstance(content, str):
        return ""

    text = content.strip()

    text = re.sub(r"[ \t]+", " ", text)
    
    return text


def _conversation_fingerprint(model: str, conv: List[Dict[str, str]]) -> str:
    """
    Stable cache key for a single example.
    We serialize the entire conversation to ensure uniqueness.
    """
    content = json.dumps(conv, sort_keys=True, ensure_ascii=False)
    
    content = _canonical_prompt_string(content)

    key_obj = {
        "model": model,
        "prompt": content,
    }

    return json.dumps(key_obj, sort_keys=True, ensure_ascii=False)


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: str, key: str) -> str:
    """
    Map a logical key to a file path under cache_dir.
    """
    fname = _hash_key(key) + ".json"
    return os.path.join(cache_dir, fname)

def _disk_cache_put_success(cache_dir: str, key: str, value: Dict[str, Any]) -> None:
    """
    Store a successful result.
    NEVER store failures.
    """
    path = _cache_path(cache_dir, key)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
    except Exception:
        pass

def _disk_cache_get(cache_dir: str, key: str) -> Optional[Dict[str, Any]]:
    """
    Try to load a cached JSON dict from disk.
    Return dict on hit, None on miss/any error.
    """
    path = _cache_path(cache_dir, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _disk_cache_put(cache_dir: str, key: str, value: Dict[str, Any]) -> None:
    """
    Store a JSON-serializable dict on disk.
    Fail silently on errors (we just lose cache for that item).
    """
    path = _cache_path(cache_dir, key)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
    except Exception:
        return

_MEM_CACHE: Dict[str, Dict[str, Any]] = {}


def generate_json_batch(
    message_list: List[List[Dict[str, str]]],
    model: str,
    api_base: Optional[str] = None,
    api_key_env: str = "GEMINI_API_KEY",
    chunk_size: int = 16,
    max_workers: int = 4,
    rpm_limit: int = 100,
    window_seconds: int = 60,
    cache_dir: str = ".llm_cache",
    fallback_model: Optional[str] = "g25lite",
    ignore_cache: bool = False,
) -> List[Dict[str, Any]]:

    n = len(message_list)
    if n == 0:
        return []

    actual_model = MODEL_MAPPING.get(model, model)
    fallback_actual = MODEL_MAPPING.get(fallback_model, fallback_model) if fallback_model else None

    conn_kwargs = {}

    if api_base:
        conn_kwargs["api_base"] = api_base

    if "vertex_ai" in actual_model:
        v_loc = os.environ.get("VERTEX_LOCATION", "us-central1")
        conn_kwargs["vertex_location"] = v_loc
        v_proj = os.environ.get("VERTEX_PROJECT")
        if v_proj:
            conn_kwargs["vertex_project"] = v_proj
        # Vertex uses ADC or GOOGLE_APPLICATION_CREDENTIALS, not GEMINI_API_KEY
        api_key = None 
    else:
        # API Key handling for Google AI Studio
        api_key = os.environ.get(api_key_env)
        if "gemini" in actual_model and not api_key:
             print(f"[llm] Warning: {api_key_env} not found in environment.")


    # look which items can be loaded from successful cache
    final_results: List[Dict[str, Any] | None] = [None] * n
    to_query_indices: List[int] = []
    to_query_messages: List[List[Dict[str, str]]] = []

    for i, conv in enumerate(message_list):
        if not ignore_cache:
            key = _conversation_fingerprint(model, conv)

            if key in _MEM_CACHE:
                final_results[i] = _MEM_CACHE[key]
                continue

            cached = _disk_cache_get(cache_dir, key)
            if cached is not None:
                _MEM_CACHE[key] = cached
                final_results[i] = cached
                continue

        to_query_indices.append(i)
        to_query_messages.append(conv)

    n_uncached = len(to_query_messages)
    if n_uncached == 0:
        return [(r or {}, {}) for r in final_results]

    print(
        f"[llm] Start batch: {n} total, {n_uncached} uncached. "
        f"chunk_size={chunk_size}, rpm_limit={rpm_limit}"
    )

    window_start = None
    used_in_window = 0

    def _maybe_wait_for_window(chunk_len: int):
        nonlocal window_start, used_in_window
        now = time.time()

        if window_start is None:
            window_start = now
            used_in_window = 0

        elapsed = now - window_start

        if elapsed >= window_seconds:
            window_start = now
            used_in_window = 0
            return

        if used_in_window + chunk_len > rpm_limit:
            sleep_t = window_seconds - elapsed
            if sleep_t > 0:
                print(f"[llm] Sleeping {sleep_t:.1f}s for RPM limit ...")
                time.sleep(sleep_t)
            window_start = time.time()
            used_in_window = 0

    start_idx = 0

    while start_idx < n_uncached:
        end_idx = min(start_idx + chunk_size, n_uncached)
        chunk = to_query_messages[start_idx:end_idx]
        chunk_indices = to_query_indices[start_idx:end_idx]
        chunk_len = len(chunk)

        t0 = time.time()
        _maybe_wait_for_window(chunk_len)
        t1 = time.time()

        attempts = 0
        while True:
            attempts += 1
            print(
                f"[llm] Processing uncached {start_idx+1}-{end_idx}/{n_uncached}, "
                f"attempt {attempts}"
            )
            print(f"[llm]   waited_for_window={t1 - t0:.3f}s")

            t_call0 = time.time()
            try:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
                
                futures_to_idx = {}
                for idx, msg_list in enumerate(chunk):
                    f = executor.submit(
                        litellm.completion,
                        model=actual_model,
                        api_key=api_key,
                        messages=msg_list,
                        temperature=0,
                        num_retries=0, 
                        timeout=30,
                        **conn_kwargs
                    )
                    futures_to_idx[f] = idx
                
                resp_list = [None] * len(chunk)
                
                for f in concurrent.futures.as_completed(futures_to_idx.keys(), timeout=90):
                    idx = futures_to_idx[f]
                    try:
                        resp_list[idx] = f.result()
                    except Exception as e:
                        resp_list[idx] = e
                        
                executor.shutdown(wait=False, cancel_futures=True)
                
                for i in range(len(resp_list)):
                    if resp_list[i] is None:
                        resp_list[i] = Exception("Timeout: litellm.completion deadlocked")

            except concurrent.futures.TimeoutError:
                executor.shutdown(wait=False, cancel_futures=True)
                print("[llm][warn] ThreadPool hung. Orphaning threads.")
                for i in range(len(resp_list)):
                    if resp_list[i] is None:
                        resp_list[i] = Exception("Timeout: litellm.completion deadlocked")
            except Exception as e:
                msg = str(e)
                overloaded = (
                    "503" in msg
                    or "overloaded" in msg
                    or "UNAVAILABLE" in msg
                )

                if overloaded and fallback_model:
                    print(f"[llm] Primary {model} overloaded; retrying with {fallback_model}")
                    try:
                        resp_list = batch_completion(
                            model=fallback_actual,
                            api_key=api_key,
                            messages=chunk,
                            temperature=0,
                            max_workers=max_workers,
                            return_exceptions=True,
                            **conn_kwargs
                        )
                    except Exception:
                        # cannot recover
                        for gi in chunk_indices:
                            final_results[gi] = {}
                        break
                else:
                    print(f"[llm][warn] exception: {e}")
                    if attempts < 3:
                        time.sleep(5)
                        continue
                    for gi in chunk_indices:
                        final_results[gi] = {}
                    break
            finally:
                t_call1 = time.time()
                print(f"[llm]   batch_completion took {t_call1 - t_call0:.3f}s")

            if any(
                isinstance(r, Exception) and (
                    "429" in str(r)
                    or "RateLimitError" in str(r)
                    or "RESOURCE_EXHAUSTED" in str(r)
                )
                for r in resp_list
            ):
                print("[llm] Rate limit inside batch.")
                if attempts < 3:
                    print("[llm] Rate limit inside batch. Retrying...")
                    time.sleep(5)
                    continue
                else:
                    print("[llm] Rate limit retries exhausted. Processing partial results...")
                    pass

            for gi, conv, r in zip(chunk_indices, chunk, resp_list):
                key = _conversation_fingerprint(model, conv)

                if isinstance(r, Exception):
                    print(f"[llm][warn] item-level error: {r}")
                    final_results[gi] = {}
                    continue

                try:
                    content = r["choices"][0]["message"]["content"]
                    parsed = _safe_json_loads(content)
                    
                    usage_info = r.get("usage", {})
                except Exception:
                    parsed = {}
                    usage_info = {}

                if parsed:
                    _MEM_CACHE[key] = parsed
                    _disk_cache_put_success(cache_dir, key, parsed)

                final_results[gi] = (parsed, usage_info)

            used_in_window += chunk_len
            break

        start_idx = end_idx

    cleaned_results = []
    for r in final_results:
        if isinstance(r, tuple):
            cleaned_results.append(r)
        elif isinstance(r, dict):
            cleaned_results.append((r, {}))
        else:
            cleaned_results.append(({}, {}))
    
    return cleaned_results