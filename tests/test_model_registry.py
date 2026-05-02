"""Branch-coverage tests for nre.utils.model_registry.

The module is mostly HTTP plumbing for NGC. We cover the tractable pieces
that can be exercised without spinning up a real or mocked HTTP server:

  * pure helpers (`partial_api_key`, `log_and_raise`)
  * URL/file validation predicates
  * netrc-based credential resolution (monkeypatching `netrc.netrc`)
  * NGC-specific URL parsing and session-token shortcut for `nvapi-` PATs
  * the `create_model_registry` factory branch

The full download/auth path (`_download_to_file`, `get_model`,
`_request_session_token_from_legacy_api_key`) is left for end-to-end coverage —
those need a real HTTP stack and a valid NGC token, which the unit-test
venv does not have.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from instant_nurec._pkg.utils.model_registry import (
    ModelRegistry,
    ModelRegistryError,
    NgcModelRegistry,
    create_model_registry,
    log_and_raise,
    partial_api_key,
)


# ---------------------------------------------------------------------------
# partial_api_key
# ---------------------------------------------------------------------------


def test_partial_api_key_longer_than_clear_view_is_masked():
    out = partial_api_key("nvapi-secret123", clear_view_length=6)
    # First 6 chars exposed, remaining 9 replaced with stars.
    assert out == "nvapi-" + "*" * 9


def test_partial_api_key_equal_length_returns_unmasked():
    """Edge case: clear_view_length equals key length — no masking applied."""
    out = partial_api_key("abcdef", clear_view_length=6)
    assert out == "abcdef"


def test_partial_api_key_shorter_than_clear_view_returns_unmasked():
    """Edge case: short key doesn't get truncated or padded."""
    out = partial_api_key("abc", clear_view_length=6)
    assert out == "abc"


def test_partial_api_key_zero_clear_view_masks_everything():
    out = partial_api_key("hello", clear_view_length=0)
    assert out == "*" * 5


# ---------------------------------------------------------------------------
# log_and_raise
# ---------------------------------------------------------------------------


def test_log_and_raise_with_format_args(caplog):
    with caplog.at_level(logging.ERROR, logger="instant_nurec._pkg.utils.model_registry"):
        with pytest.raises(ValueError, match="bad: 42"):
            log_and_raise(ValueError, "bad: %s", 42)
    assert "bad: 42" in caplog.text


def test_log_and_raise_without_format_args(caplog):
    with caplog.at_level(logging.ERROR, logger="instant_nurec._pkg.utils.model_registry"):
        with pytest.raises(RuntimeError, match="plain message"):
            log_and_raise(RuntimeError, "plain message")
    assert "plain message" in caplog.text


def test_model_registry_error_inherits_exception():
    err = ModelRegistryError("boom")
    assert isinstance(err, Exception)
    assert str(err) == "boom"


# ---------------------------------------------------------------------------
# Helpers: a minimal concrete subclass for testing the abstract base
# ---------------------------------------------------------------------------


class _StubRegistry(ModelRegistry):
    """Minimal concrete subclass — supplies the abstract methods so we can
    instantiate ``ModelRegistry`` directly for branch coverage of the base."""

    @property
    def domain(self) -> str:
        return "example.com"

    @staticmethod
    def api_key_hint() -> str:
        return "(testing — no API key required)"


# ---------------------------------------------------------------------------
# ModelRegistry.__init__ — invalid URL branch
# ---------------------------------------------------------------------------


def test_init_raises_on_invalid_url(tmp_path):
    with pytest.raises(ModelRegistryError, match="Invalid model URL"):
        _StubRegistry("https://wrong-domain.com/foo", tmp_path)


def test_init_with_valid_url_stores_attributes(tmp_path):
    reg = _StubRegistry("https://example.com/foo", tmp_path)
    assert reg.model_url == "https://example.com/foo"
    assert reg.model_cache_dir == tmp_path
    assert reg.api_key is None
    assert reg.session is None


# ---------------------------------------------------------------------------
# ModelRegistry._validate_url
# ---------------------------------------------------------------------------


def test_validate_url_accepts_matching_domain(tmp_path):
    reg = _StubRegistry("https://example.com/foo", tmp_path)
    assert reg._validate_url("https://example.com/anything") is True


def test_validate_url_rejects_mismatched_domain(tmp_path, caplog):
    reg = _StubRegistry("https://example.com/foo", tmp_path)
    with caplog.at_level(logging.WARNING, logger="instant_nurec._pkg.utils.model_registry"):
        assert reg._validate_url("https://other.com/foo") is False
    assert "Invalid URL domain" in caplog.text


def test_validate_url_returns_false_on_unparseable_url(tmp_path, monkeypatch):
    """Force urlparse to raise; the except branch should swallow and return False."""
    reg = _StubRegistry("https://example.com/foo", tmp_path)

    import instant_nurec._pkg.utils.model_registry as mr

    def _boom(_url):
        raise RuntimeError("synthetic urlparse failure")

    monkeypatch.setattr(mr, "urlparse", _boom)
    assert reg._validate_url("https://example.com/anything") is False


# ---------------------------------------------------------------------------
# ModelRegistry._verify_cached_file / _verify_downloaded_file
# ---------------------------------------------------------------------------


def test_verify_cached_file_true_when_at_or_above_min(tmp_path):
    reg = _StubRegistry("https://example.com/x", tmp_path)
    f = tmp_path / "ok.bin"
    f.write_bytes(b"x" * (reg.MIN_FILE_SIZE + 1))
    assert reg._verify_cached_file(f) is True


def test_verify_cached_file_false_when_below_min(tmp_path):
    reg = _StubRegistry("https://example.com/x", tmp_path)
    f = tmp_path / "small.bin"
    f.write_bytes(b"x")
    assert reg._verify_cached_file(f) is False


def test_verify_cached_file_false_on_oserror(tmp_path):
    """Non-existent file → stat() raises OSError → returns False."""
    reg = _StubRegistry("https://example.com/x", tmp_path)
    assert reg._verify_cached_file(tmp_path / "nope.bin") is False


def test_verify_downloaded_file_true_when_size_matches(tmp_path):
    reg = _StubRegistry("https://example.com/x", tmp_path)
    f = tmp_path / "d.bin"
    f.write_bytes(b"x" * 100)
    assert reg._verify_downloaded_file(f, 100) is True


def test_verify_downloaded_file_false_when_size_mismatches(tmp_path):
    reg = _StubRegistry("https://example.com/x", tmp_path)
    f = tmp_path / "d.bin"
    f.write_bytes(b"x" * 100)
    assert reg._verify_downloaded_file(f, 101) is False


def test_verify_downloaded_file_false_on_oserror(tmp_path):
    reg = _StubRegistry("https://example.com/x", tmp_path)
    assert reg._verify_downloaded_file(tmp_path / "nope.bin", 100) is False


# ---------------------------------------------------------------------------
# ModelRegistry._request_session_token_from_api_key
# ---------------------------------------------------------------------------


def test_request_session_token_default_returns_api_key_unchanged(tmp_path):
    reg = _StubRegistry("https://example.com/x", tmp_path)
    assert reg._request_session_token_from_api_key("hunter2") == "hunter2"


def test_request_session_token_rejects_empty_api_key(tmp_path):
    reg = _StubRegistry("https://example.com/x", tmp_path)
    with pytest.raises(ModelRegistryError, match="API key cannot be empty"):
        reg._request_session_token_from_api_key("")


# ---------------------------------------------------------------------------
# ModelRegistry._get_session + BearerAuth
# ---------------------------------------------------------------------------


def test_get_session_returns_session_with_bearer_auth(tmp_path):
    import requests

    reg = _StubRegistry("https://example.com/x", tmp_path)
    session = reg._get_session("token-xyz")
    assert isinstance(session, requests.Session)

    # Verify the auth handler stamps the Authorization header on a request.
    req = requests.Request("GET", "https://example.com/anything")
    prepared = req.prepare()
    auth_handler = session.auth
    auth_handler(prepared)
    assert prepared.headers["Authorization"] == "Bearer token-xyz"


def test_get_session_bearer_auth_rejects_empty_token(tmp_path):
    reg = _StubRegistry("https://example.com/x", tmp_path)
    with pytest.raises(ValueError, match="Token cannot be empty"):
        reg._get_session("")


def test_get_session_bearer_auth_rejects_whitespace_only_token(tmp_path):
    reg = _StubRegistry("https://example.com/x", tmp_path)
    with pytest.raises(ValueError, match="Token cannot be empty"):
        reg._get_session("   ")


# ---------------------------------------------------------------------------
# ModelRegistry._get_api_key — empty / netrc paths
# ---------------------------------------------------------------------------


def test_get_api_key_rejects_empty_string(tmp_path):
    reg = _StubRegistry("https://example.com/x", tmp_path)
    with pytest.raises(ModelRegistryError, match="API key cannot be empty"):
        reg._get_api_key("   ")


def test_get_api_key_returns_provided_key_unchanged(tmp_path):
    reg = _StubRegistry("https://example.com/x", tmp_path)
    assert reg._get_api_key("provided-key") == "provided-key"


class _FakeNetrc:
    """Drop-in replacement for netrc.netrc(). Authenticators returns the
    configured (login, account, password) tuple keyed by domain."""

    def __init__(self, table):
        self._table = table

    def authenticators(self, host):
        return self._table.get(host)


def test_get_api_key_falls_back_to_netrc_credential(tmp_path, monkeypatch):
    import netrc

    reg = _StubRegistry("https://example.com/x", tmp_path)
    monkeypatch.setattr(
        netrc, "netrc", lambda: _FakeNetrc({"example.com": ("user", "acct", "netrc-key")})
    )
    assert reg._get_api_key(None) == "netrc-key"


def test_get_api_key_raises_when_netrc_returns_none(tmp_path, monkeypatch):
    import netrc

    reg = _StubRegistry("https://example.com/x", tmp_path)
    monkeypatch.setattr(netrc, "netrc", lambda: _FakeNetrc({}))  # no entry
    with pytest.raises(ModelRegistryError, match="API key must be provided"):
        reg._get_api_key(None)


def test_get_api_key_wraps_netrc_filenotfound_error(tmp_path, monkeypatch):
    """A missing ~/.netrc raises FileNotFoundError; the helper must wrap it
    in a ModelRegistryError, not propagate the raw exception."""
    import netrc

    def _raise_missing():
        raise FileNotFoundError("~/.netrc not found")

    reg = _StubRegistry("https://example.com/x", tmp_path)
    monkeypatch.setattr(netrc, "netrc", _raise_missing)
    with pytest.raises(ModelRegistryError, match="Failed to get API key"):
        reg._get_api_key(None)


# ---------------------------------------------------------------------------
# ModelRegistry._get_credential_from_netrc — direct calls
# ---------------------------------------------------------------------------


def test_get_credential_from_netrc_returns_password(monkeypatch):
    import netrc

    monkeypatch.setattr(
        netrc, "netrc", lambda: _FakeNetrc({"example.com": ("u", "a", "secret")})
    )
    assert ModelRegistry._get_credential_from_netrc("example.com") == "secret"


def test_get_credential_from_netrc_returns_none_when_no_authenticator(monkeypatch):
    import netrc

    monkeypatch.setattr(netrc, "netrc", lambda: _FakeNetrc({}))
    assert ModelRegistry._get_credential_from_netrc("example.com") is None


def test_get_credential_from_netrc_returns_none_when_password_empty(monkeypatch):
    import netrc

    monkeypatch.setattr(
        netrc, "netrc", lambda: _FakeNetrc({"example.com": ("u", "a", "")})
    )
    assert ModelRegistry._get_credential_from_netrc("example.com") is None


def test_get_credential_from_netrc_filenotfound_wraps_to_registry_error(monkeypatch):
    import netrc

    def _raise():
        raise FileNotFoundError(".netrc gone")

    monkeypatch.setattr(netrc, "netrc", _raise)
    with pytest.raises(ModelRegistryError, match="Failed to read .netrc"):
        ModelRegistry._get_credential_from_netrc("example.com")


# ---------------------------------------------------------------------------
# NgcModelRegistry — domain, api_key_hint, _get_api_key, _extract_org_team
# ---------------------------------------------------------------------------


def _ngc_url(org="nvstaging", team="nre", model="kelvin", version="1.0"):
    return f"https://api.ngc.nvidia.com/v2/org/{org}/team/{team}/models/{model}/versions/{version}/files/{model}.ckpt"


def test_ngc_domain_property(tmp_path):
    reg = NgcModelRegistry(_ngc_url(), tmp_path, api_key="nvapi-test")
    assert reg.domain == "api.ngc.nvidia.com"


def test_ngc_api_key_hint_returns_class_constant():
    hint = NgcModelRegistry.api_key_hint()
    assert "Personal Key" in hint
    assert "ngc.nvidia.com" in hint


def test_ngc_get_api_key_uses_env_var_when_none_provided(tmp_path, monkeypatch):
    monkeypatch.setenv("NGC_API_KEY", "nvapi-from-env")
    reg = NgcModelRegistry(_ngc_url(), tmp_path, api_key="nvapi-init")
    # Reset the api_key to None so _get_api_key takes the env-var branch.
    reg.api_key = None
    assert reg._get_api_key(None) == "nvapi-from-env"


def test_ngc_get_api_key_rejects_empty_string(tmp_path):
    reg = NgcModelRegistry(_ngc_url(), tmp_path, api_key="nvapi-init")
    with pytest.raises(ModelRegistryError, match="API key cannot be empty"):
        reg._get_api_key("   ")


def test_ngc_get_api_key_warns_on_legacy_prefix(tmp_path, monkeypatch, caplog):
    """Non-`nvapi-` prefix → warn that this looks like a legacy key."""
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    import netrc

    monkeypatch.setattr(netrc, "netrc", lambda: _FakeNetrc({}))
    reg = NgcModelRegistry(_ngc_url(), tmp_path, api_key="nvapi-init")
    with caplog.at_level(logging.WARNING, logger="instant_nurec._pkg.utils.model_registry"):
        out = reg._get_api_key("legacy-token-no-prefix")
    assert out == "legacy-token-no-prefix"
    assert "does not appear to be a valid NGC Personal Key" in caplog.text


def test_ngc_get_api_key_silent_on_nvapi_prefix(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    reg = NgcModelRegistry(_ngc_url(), tmp_path, api_key="nvapi-init")
    with caplog.at_level(logging.WARNING, logger="instant_nurec._pkg.utils.model_registry"):
        out = reg._get_api_key("nvapi-good")
    assert out == "nvapi-good"
    assert "does not appear" not in caplog.text


def test_ngc_extract_org_team_parses_url_components(tmp_path):
    reg = NgcModelRegistry(_ngc_url(org="nvidia", team="rtx"), tmp_path, api_key="nvapi-x")
    org, team = reg._extract_org_team_from_model_url()
    assert org == "nvidia"
    assert team == "rtx"


def test_ngc_extract_org_team_raises_on_unmatched_url(tmp_path):
    """Malformed (but domain-valid) URL → ModelRegistryError, not raw RE error."""
    reg = NgcModelRegistry(_ngc_url(), tmp_path, api_key="nvapi-x")
    # Replace url with a domain-matching but pattern-mismatching URL.
    reg.model_url = "https://api.ngc.nvidia.com/v2/wrong/path"
    with pytest.raises(ModelRegistryError, match="Invalid NGC URL format"):
        reg._extract_org_team_from_model_url()


# ---------------------------------------------------------------------------
# NgcModelRegistry._request_session_token_from_api_key — branch on prefix
# ---------------------------------------------------------------------------


def test_ngc_session_token_passes_through_nvapi_prefixed_key(tmp_path):
    """PAT keys (`nvapi-...`) act as the session token directly — no HTTP call."""
    reg = NgcModelRegistry(_ngc_url(), tmp_path, api_key="nvapi-init")
    assert reg._request_session_token_from_api_key("nvapi-personal") == "nvapi-personal"


def test_ngc_session_token_legacy_branch_calls_exchange(tmp_path, monkeypatch):
    """Non-`nvapi-` keys go through the legacy-token exchange path."""
    reg = NgcModelRegistry(_ngc_url(), tmp_path, api_key="nvapi-init")

    captured = {}

    def _fake_exchange(self, legacy_api_key, org, team):
        captured["api_key"] = legacy_api_key
        captured["org"] = org
        captured["team"] = team
        return "session-from-exchange"

    monkeypatch.setattr(
        NgcModelRegistry, "_request_session_token_from_legacy_api_key", _fake_exchange
    )
    out = reg._request_session_token_from_api_key("legacy-token")
    assert out == "session-from-exchange"
    assert captured["api_key"] == "legacy-token"
    assert captured["org"] == "nvstaging"
    assert captured["team"] == "nre"


# ---------------------------------------------------------------------------
# create_model_registry factory
# ---------------------------------------------------------------------------


def test_create_model_registry_returns_ngc_for_ngc_domain(tmp_path):
    reg = create_model_registry(_ngc_url(), tmp_path, api_key="nvapi-x")
    assert isinstance(reg, NgcModelRegistry)


def test_create_model_registry_rejects_unknown_domain(tmp_path):
    with pytest.raises(ModelRegistryError, match="Unsupported model registry domain"):
        create_model_registry("https://other-host.io/foo", tmp_path)


# ---------------------------------------------------------------------------
# Abstract method bodies (`pass`) — exercised via super() calls.
# ABC normally forces subclasses to override, but the body is reachable when
# a concrete subclass calls super().<method>(). This is the cheapest way to
# get coverage of the literal `pass` lines in the abstract declarations.
# ---------------------------------------------------------------------------


def test_abstract_domain_body_runs_via_property_fget(tmp_path):
    """The abstract `domain` property descriptor's `fget` is the literal
    function with body ``pass``. Call it directly to cover the body — going
    through ``super().domain`` from a sub-subclass would reach the StubRegistry
    override instead of the abstract definition."""
    reg = _StubRegistry("https://example.com/x", tmp_path)
    # The property is shadowed on the subclass; fetch the abstract one off the
    # base directly via __dict__ to bypass the override.
    abstract_prop = ModelRegistry.__dict__["domain"]
    assert abstract_prop.fget(reg) is None  # `pass` returns None implicitly


def test_abstract_api_key_hint_body_runs_via_super_call():
    class _Sub(NgcModelRegistry):
        @staticmethod
        def api_key_hint() -> str:
            ModelRegistry.api_key_hint()  # walks into the abstract pass
            return "stub"

    assert _Sub.api_key_hint() == "stub"


# ---------------------------------------------------------------------------
# get_model — cache-hit / cache-stale branches (no network)
# ---------------------------------------------------------------------------


def test_get_model_returns_cached_path_when_file_is_valid(tmp_path):
    reg = _StubRegistry("https://example.com/foo.bin", tmp_path)
    cached = tmp_path / "foo.bin"
    cached.write_bytes(b"x" * (reg.MIN_FILE_SIZE + 1))
    assert reg.get_model() == str(cached)


def test_get_model_redownloads_when_cached_file_is_too_small(tmp_path, monkeypatch):
    """Stale cache (file present but below MIN_FILE_SIZE) → unlink + redownload."""
    reg = _StubRegistry("https://example.com/foo.bin", tmp_path)
    stale = tmp_path / "foo.bin"
    stale.write_bytes(b"x")  # < MIN_FILE_SIZE → invalid

    download_target = tmp_path / "foo.bin.fresh"
    download_target.write_bytes(b"x" * (reg.MIN_FILE_SIZE + 1))

    def _fake_download(self, filename):
        assert filename == "foo.bin"
        # Replicate redownload-after-unlink: the stale file should already be gone.
        assert not stale.exists()
        return str(download_target)

    monkeypatch.setattr(_StubRegistry, "_download_to_file", _fake_download)
    out = reg.get_model()
    assert out == str(download_target)


def test_get_model_calls_download_when_no_cache_present(tmp_path, monkeypatch):
    reg = _StubRegistry("https://example.com/foo.bin", tmp_path)
    sentinel = str(tmp_path / "downloaded-fresh.bin")

    monkeypatch.setattr(
        _StubRegistry, "_download_to_file", lambda self, filename: sentinel
    )
    assert reg.get_model() == sentinel


# ---------------------------------------------------------------------------
# NgcModelRegistry._request_session_token_from_legacy_api_key — HTTP path
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, *, status: int = 200, payload=None, raise_on_get_json=False):
        self.status_code = status
        self._payload = payload or {}
        self._raise_on_get_json = raise_on_get_json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self):
        if self._raise_on_get_json:
            raise ValueError("not json")
        return self._payload


def test_legacy_api_key_exchange_returns_session_token_on_success(tmp_path, monkeypatch):
    import instant_nurec._pkg.utils.model_registry as mr

    reg = NgcModelRegistry(_ngc_url(), tmp_path, api_key="nvapi-init")

    captured = {}

    def _fake_get(url, *, auth, headers, timeout):
        captured["url"] = url
        captured["auth"] = auth
        return _FakeResponse(payload={"token": "session-abc"})

    monkeypatch.setattr(mr.requests, "get", _fake_get)
    out = reg._request_session_token_from_legacy_api_key("legacy-tok", "org-x", "team-y")
    assert out == "session-abc"
    # Auth-URL must encode org+team in the scope.
    assert "scope=group/ngc:org-x" in captured["url"]
    assert "team-y" in captured["url"]
    assert captured["auth"] == ("$oauthtoken", "legacy-tok")


def test_legacy_api_key_exchange_raises_when_no_token_in_response(tmp_path, monkeypatch):
    import instant_nurec._pkg.utils.model_registry as mr

    reg = NgcModelRegistry(_ngc_url(), tmp_path, api_key="nvapi-init")
    monkeypatch.setattr(
        mr.requests, "get", lambda *a, **k: _FakeResponse(payload={})  # missing 'token'
    )
    with pytest.raises(ModelRegistryError, match="No token found"):
        reg._request_session_token_from_legacy_api_key("legacy-tok", "o", "t")


def test_legacy_api_key_exchange_wraps_request_exception(tmp_path, monkeypatch):
    import instant_nurec._pkg.utils.model_registry as mr
    import requests

    reg = NgcModelRegistry(_ngc_url(), tmp_path, api_key="nvapi-init")

    def _boom(*a, **k):
        raise requests.exceptions.ConnectionError("offline")

    monkeypatch.setattr(mr.requests, "get", _boom)
    with pytest.raises(ModelRegistryError, match="Failed to exchange legacy API key"):
        reg._request_session_token_from_legacy_api_key("legacy-tok", "o", "t")


def test_legacy_api_key_exchange_wraps_invalid_response_format(tmp_path, monkeypatch):
    import instant_nurec._pkg.utils.model_registry as mr

    reg = NgcModelRegistry(_ngc_url(), tmp_path, api_key="nvapi-init")
    monkeypatch.setattr(
        mr.requests,
        "get",
        lambda *a, **k: _FakeResponse(raise_on_get_json=True),
    )
    with pytest.raises(ModelRegistryError, match="Invalid response format"):
        reg._request_session_token_from_legacy_api_key("legacy-tok", "o", "t")


def test_legacy_api_key_exchange_propagates_http_error(tmp_path, monkeypatch):
    """A 5xx from raise_for_status() flows through the requests-exception branch."""
    import instant_nurec._pkg.utils.model_registry as mr

    reg = NgcModelRegistry(_ngc_url(), tmp_path, api_key="nvapi-init")
    monkeypatch.setattr(
        mr.requests, "get", lambda *a, **k: _FakeResponse(status=503)
    )
    with pytest.raises(ModelRegistryError, match="Failed to exchange legacy API key"):
        reg._request_session_token_from_legacy_api_key("legacy-tok", "o", "t")


# ---------------------------------------------------------------------------
# ModelRegistry._download_to_file — full HTTP-streaming path with mocks
# ---------------------------------------------------------------------------


class _StreamingResponse:
    """Minimal stand-in for ``requests.Response`` used by ``_download_to_file``."""

    def __init__(self, *, chunks, content_length=None, raise_for_status_exc=None):
        self._chunks = list(chunks)
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)
        self._exc = raise_for_status_exc

    def raise_for_status(self):
        if self._exc:
            raise self._exc

    def iter_content(self, chunk_size):
        return iter(self._chunks)


class _FakeSession:
    """Stand-in for ``requests.Session`` exposing only ``.get``."""

    def __init__(self, response):
        self._response = response

    def get(self, url, stream=False):
        return self._response


def test_download_to_file_writes_chunks_and_returns_path(tmp_path, monkeypatch):
    """Happy path: stream a few chunks, write the file, verify size, return path."""
    reg = _StubRegistry("https://example.com/foo.bin", tmp_path, api_key="apikey-x")
    payload = b"x" * (reg.MIN_FILE_SIZE + 1)
    # Split payload into chunks the stream code will accumulate.
    chunks = [payload[i : i + 1024] for i in range(0, len(payload), 1024)]
    response = _StreamingResponse(chunks=chunks, content_length=len(payload))
    reg.session = _FakeSession(response)

    out = reg._download_to_file("foo.bin")
    assert Path(out).exists()
    assert Path(out).read_bytes() == payload


def test_download_to_file_lazy_inits_api_key_and_session(tmp_path, monkeypatch):
    """If api_key=None and session=None, _download_to_file initializes both."""
    import netrc

    monkeypatch.setattr(
        netrc, "netrc", lambda: _FakeNetrc({"example.com": ("u", "a", "lazy-key")})
    )
    reg = _StubRegistry("https://example.com/foo.bin", tmp_path, api_key=None)
    payload = b"x" * (reg.MIN_FILE_SIZE + 5)
    response = _StreamingResponse(chunks=[payload], content_length=len(payload))

    # Patch _get_session to install the fake session instead of real requests.
    monkeypatch.setattr(
        _StubRegistry, "_get_session", lambda self, _tok: _FakeSession(response)
    )

    out = reg._download_to_file("foo.bin")
    assert reg.api_key == "lazy-key"
    assert reg.session is not None
    assert Path(out).exists()


def test_download_to_file_raises_on_empty_api_key_after_lazy_init(tmp_path, monkeypatch):
    """If lazy-init returns an empty api_key, the empty-key guard fires."""
    reg = _StubRegistry("https://example.com/foo.bin", tmp_path, api_key=None)
    monkeypatch.setattr(_StubRegistry, "_get_api_key", lambda self, ak=None: "")

    with pytest.raises(ModelRegistryError, match="API key cannot be empty"):
        reg._download_to_file("foo.bin")


def test_download_to_file_raises_when_no_content_length_header(tmp_path):
    """Missing Content-Length header → ModelRegistryError."""
    reg = _StubRegistry("https://example.com/foo.bin", tmp_path, api_key="k")
    response = _StreamingResponse(chunks=[b"x"])  # no content-length set
    reg.session = _FakeSession(response)

    with pytest.raises(ModelRegistryError, match="Server did not provide Content-Length"):
        reg._download_to_file("foo.bin")


def test_download_to_file_raises_on_size_mismatch(tmp_path):
    """Downloaded file size != expected size → ModelRegistryError."""
    reg = _StubRegistry("https://example.com/foo.bin", tmp_path, api_key="k")
    # Claim 999 bytes but only stream 1.
    response = _StreamingResponse(chunks=[b"x"], content_length=999)
    reg.session = _FakeSession(response)

    with pytest.raises(ModelRegistryError, match="Downloaded file is invalid"):
        reg._download_to_file("foo.bin")


def test_download_to_file_wraps_request_exception_with_api_key_hint(tmp_path, monkeypatch):
    """A RequestException → ModelRegistryError that mentions the api-key hint."""
    import requests

    reg = _StubRegistry("https://example.com/foo.bin", tmp_path, api_key="abcdefghij")
    response = _StreamingResponse(
        chunks=[],
        raise_for_status_exc=requests.exceptions.HTTPError("boom"),
    )
    reg.session = _FakeSession(response)

    with pytest.raises(ModelRegistryError) as exc_info:
        reg._download_to_file("foo.bin")
    msg = str(exc_info.value)
    assert "Failed to download model" in msg
    # The hint should be partially-masked api key (first 8 chars + stars).
    assert "abcdefgh" in msg
    assert "no API key required" in msg  # _StubRegistry's api_key_hint()


def test_download_to_file_wraps_unexpected_error(tmp_path):
    """Non-RequestException → generic 'Unexpected error' wrap."""
    reg = _StubRegistry("https://example.com/foo.bin", tmp_path, api_key="k")

    class _BoomSession:
        def get(self, *a, **kw):
            raise RuntimeError("synthetic failure")

    reg.session = _BoomSession()

    with pytest.raises(ModelRegistryError, match="Unexpected error"):
        reg._download_to_file("foo.bin")


def test_download_to_file_skips_keep_alive_chunks(tmp_path):
    """Empty chunks (from `iter_content`) should be skipped, not written."""
    reg = _StubRegistry("https://example.com/foo.bin", tmp_path, api_key="k")
    payload = b"y" * (reg.MIN_FILE_SIZE + 1)
    # Interleave keep-alive (empty) chunks with payload.
    chunks = [b"", payload[: len(payload) // 2], b"", payload[len(payload) // 2 :], b""]
    response = _StreamingResponse(chunks=chunks, content_length=len(payload))
    reg.session = _FakeSession(response)

    out = reg._download_to_file("foo.bin")
    assert Path(out).read_bytes() == payload
