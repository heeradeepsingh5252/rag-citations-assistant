import pytest
from fastapi import HTTPException
from src.app import rate_limit
from src.app.rate_limit import enforce_rate_limit
from starlette.requests import Request


@pytest.fixture(autouse=True)
def fresh_rate_limit_store(monkeypatch):
    """Give each test an isolated rate-limit store."""
    monkeypatch.setattr(rate_limit, "_RATE_LIMIT_WINDOWS", {})


def make_request(client_ip: str = "192.0.2.10") -> Request:
    """Create a minimal Starlette request for a client IP."""
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/test",
            "raw_path": b"/test",
            "query_string": b"",
            "headers": [],
            "client": (client_ip, 12345),
            "server": ("testserver", 80),
        }
    )


def test_rate_limit_blocks_excess_requests():
    request = make_request()

    enforce_rate_limit(
        request,
        scope="query",
        limit=1,
        window_seconds=60,
    )

    with pytest.raises(HTTPException) as error:
        enforce_rate_limit(
            request,
            scope="query",
            limit=1,
            window_seconds=60,
        )

    assert error.value.status_code == 429
    assert "Retry-After" in error.value.headers


def test_different_client_ips_have_separate_limits():
    request_a = make_request("192.0.2.10")
    request_b = make_request("192.0.2.11")

    enforce_rate_limit(
        request_a,
        scope="query",
        limit=1,
        window_seconds=60,
    )

    # A different IP should have its own allowance.
    enforce_rate_limit(
        request_b,
        scope="query",
        limit=1,
        window_seconds=60,
    )


def test_different_scopes_have_separate_limits():
    request = make_request()

    enforce_rate_limit(
        request,
        scope="query",
        limit=1,
    )

    # The query request must not consume the upload allowance.
    enforce_rate_limit(
        request,
        scope="upload",
        limit=1,
    )


def test_rate_limit_rejects_invalid_configuration():
    request = make_request()

    with pytest.raises(ValueError):
        enforce_rate_limit(request, scope="query", limit=0)

    with pytest.raises(ValueError):
        enforce_rate_limit(
            request,
            scope="query",
            limit=1,
            window_seconds=0,
        )