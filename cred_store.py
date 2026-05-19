"""
Credential store — OS keyring backed.

Reconstructed module. The original upload's `cred_store.py` actually
contained the Angel One client (now correctly at core/angel_client.py);
the real credential store was missing, so app.py's `cs.load_credentials`,
`cs.save_credentials`, `cs.is_available`, `cs.clear_credentials` would
have failed to import. This restores them.

Security model:
  - Credentials are stored via the `keyring` library, which uses the
    OS-native secure store (Windows Credential Manager / macOS Keychain
    / Secret Service on Linux). They are NOT written to any file in the
    repo or working dir.
  - On Streamlit Community Cloud there is usually NO usable keyring
    backend — is_available() returns False there and the app correctly
    falls back to secrets.toml / manual entry. "Remember me" is a
    local-machine convenience only.
  - We store the four fields as one JSON blob under a single keyring
    entry to keep it atomic.
"""

import json

_SERVICE = "nifty_breadth_dashboard"
_ENTRY = "angel_one_credentials"

try:
    import keyring
    from keyring.errors import KeyringError
    _KEYRING_IMPORTED = True
except Exception:                       # keyring not installed at all
    keyring = None
    KeyringError = Exception
    _KEYRING_IMPORTED = False


def is_available() -> bool:
    """True only if a real, writable keyring backend exists.

    The `keyring` package always imports even with no backend; it falls
    back to a 'fail' backend that raises on use. We detect that so the
    UI can hide 'Remember me' instead of offering a feature that errors.
    """
    if not _KEYRING_IMPORTED:
        return False
    try:
        kr = keyring.get_keyring()
        cls = kr.__class__.__name__.lower()
        # The null/fail backends are not real secure stores.
        if "fail" in cls or "null" in cls:
            return False
        return True
    except Exception:
        return False


def save_credentials(api_key: str, client_id: str,
                     password: str, totp_key: str) -> tuple[bool, str]:
    """Persist credentials in the OS keyring.
    Returns (ok, message)."""
    if not is_available():
        return False, "No OS keyring backend available on this system."
    blob = json.dumps({
        "api_key":   api_key,
        "client_id": client_id,
        "password":  password,
        "totp_key":  totp_key,
    })
    try:
        keyring.set_password(_SERVICE, _ENTRY, blob)
        return True, "Credentials saved to OS keyring."
    except KeyringError as e:
        return False, f"Keyring write failed: {e}"
    except Exception as e:
        return False, f"Unexpected keyring error: {e}"


def load_credentials() -> dict | None:
    """Return the saved credential dict, or None if nothing stored / no
    backend. Never raises — callers treat None as 'not saved'."""
    if not is_available():
        return None
    try:
        blob = keyring.get_password(_SERVICE, _ENTRY)
    except Exception:
        return None
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except (ValueError, TypeError):
        return None
    # Only return if all four fields are present and non-empty.
    needed = ("api_key", "client_id", "password", "totp_key")
    if all(data.get(k) for k in needed):
        return {k: data[k] for k in needed}
    return None


def clear_credentials() -> tuple[bool, str]:
    """Delete the stored credential entry. Returns (ok, message).
    Treats 'nothing was stored' as success."""
    if not is_available():
        return False, "No OS keyring backend available on this system."
    try:
        keyring.delete_password(_SERVICE, _ENTRY)
        return True, "Saved credentials cleared."
    except keyring.errors.PasswordDeleteError:
        return True, "No saved credentials to clear."
    except Exception as e:
        return False, f"Could not clear credentials: {e}"
