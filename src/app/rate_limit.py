from collections import deque
from math import ceil
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request

_RATE_LIMIT_WINDOWS: dict[tuple[str, str], deque[float]] = {}
_RATE_LIMIT_LOCK = Lock()


def enforce_rate_limit(
    request: Request,
    *,
    scope: str,
    limit: int,
    window_seconds: int = 60,
) -> None:
    """Limit requests per client IP and endpoint scope."""
    if limit < 1 or window_seconds < 1:
        raise ValueError("Limit and window_seconds must be positive.")

    client_ip = request.client.host if request.client else "unknown"
    now = monotonic()
    bucket_key = (scope, client_ip)

    with _RATE_LIMIT_LOCK:
        # Remove inactive buckets to prevent unbounded memory growth.
        stale_keys = [
            key
            for key, timestamps in _RATE_LIMIT_WINDOWS.items()
            if not timestamps
            or now - timestamps[-1] >= window_seconds * 2
        ]

        for key in stale_keys:
            del _RATE_LIMIT_WINDOWS[key]

        timestamps = _RATE_LIMIT_WINDOWS.setdefault(
            bucket_key,
            deque(),
        )

        # Keep only requests within the active time window.
        while timestamps and now - timestamps[0] >= window_seconds:
            timestamps.popleft()

        if len(timestamps) >= limit:
            retry_after = max(
                1,
                ceil(window_seconds - (now - timestamps[0])),
            )

            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded for {scope}. "
                    f"Try again in {retry_after} seconds."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)