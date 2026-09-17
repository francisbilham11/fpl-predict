from __future__ import annotations
import os, re, pathlib
import requests
from typing import Optional
from dotenv import load_dotenv
from ..utils.logging import get_logger

log = get_logger(__name__)

# override=True: this module's whole job is reading credentials this project's own commands
# (set_token_env, set_cookie_env, pw_login) just wrote to .env — a stale value already sitting
# in the shell environment (.envrc/direnv, or a leftover manual export) must not shadow it.
load_dotenv(override=True)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15"
)
API_ROOT = "https://fantasy.premierleague.com/api"
MYTEAM_URL_TMPL = API_ROOT + "/my-team/{entry_id}/"


def _env(k: str) -> str:
    return os.getenv(k, "").strip()


def _upsert_env_var(path: str | os.PathLike, key: str, value: str) -> None:
    p = pathlib.Path(path)
    lines = p.read_text().splitlines() if p.exists() else []
    wrote = False
    out = []
    for line in lines:
        if re.match(rf"^\s*{re.escape(key)}\s*=", line):
            out.append(f"{key}={value}")
            wrote = True
        else:
            out.append(line)
    if not wrote:
        out.append(f"{key}={value}")
    p.write_text("\n".join(out) + "\n")


def set_token_env(token: str, save_env_path: Optional[str] = ".env") -> None:
    """Save an x-api-authorization Bearer token to .env as FPL_AUTH_TOKEN."""
    token = token.strip()
    if token.startswith("Bearer "):
        token = token[len("Bearer "):].strip()
    os.environ["FPL_AUTH_TOKEN"] = token
    if save_env_path:
        _upsert_env_var(save_env_path, "FPL_AUTH_TOKEN", token)
    log.info("Saved FPL_AUTH_TOKEN%s", f" to {save_env_path}" if save_env_path else "")


def set_cookie_env(cookie: str, save_env_path: Optional[str] = ".env") -> None:
    """Save a full Cookie header string to .env as FPL_SESSION (accepts raw 'pl_profile=…; pl_session=…')."""
    cookie = cookie.strip()
    os.environ["FPL_SESSION"] = cookie
    if save_env_path:
        _upsert_env_var(save_env_path, "FPL_SESSION", cookie)
    log.info("Saved FPL_SESSION%s", f" to {save_env_path}" if save_env_path else "")


def get_auth_header_candidates() -> list[dict[str, str]]:
    """Every configured auth header set, token first then cookie.

    Presence of a token doesn't mean it's still valid — FPL bearer tokens expire in
    around an hour (see README) — so a caller making a real request should try each
    candidate in order and fall back to the next on a 401/403 rather than trusting
    the first one blindly.
    """
    # Referer/X-Api-Language mirror what the site's own frontend actually sends alongside
    # this header (checked directly against a real browser request) — some APIs gate on
    # Referer/Origin as a lightweight CSRF/bot check, which a bare Bearer header wouldn't
    # satisfy on its own.
    common = {
        "User-Agent": UA,
        "accept-language": "en",
        "Referer": "https://fantasy.premierleague.com/",
        "X-Api-Language": "en",
    }

    candidates = []
    token = _env("FPL_AUTH_TOKEN")
    if token:
        candidates.append({**common, "x-api-authorization": f"Bearer {token}"})

    cookie = _env("FPL_SESSION")
    if cookie:
        candidates.append({**common, "Cookie": cookie})

    if not candidates:
        raise RuntimeError(
            "No auth configured. Set FPL_AUTH_TOKEN (preferred) or FPL_SESSION in your environment or .env.\n"
            "Tip: Copy the whole 'Cookie' OR the 'x-api-authorization: Bearer …' value from DevTools and run:\n"
            "  fpl auth set-token  (for Bearer)\n"
            "  or\n"
            "  fpl auth set-cookie (for Cookie)"
        )
    return candidates


def get_auth_headers() -> dict[str, str]:
    """The single preferred auth header set (token if set, else cookie).

    For a real request that can retry on failure, prefer get_auth_header_candidates()
    and fall back through the list on a 401/403 instead of trusting this blindly.
    """
    return get_auth_header_candidates()[0]


def pw_login(email: str, password: str, save_env_path: Optional[str] = ".env") -> str:
    """
    Login to FPL using email/password and return the session cookie.
    Saves to FPL_SESSION if save_env_path is provided.
    """
    login_url = "https://users.premierleague.com/accounts/login/"
    
    # Start a session to maintain cookies
    session = requests.Session()
    session.headers.update({
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://fantasy.premierleague.com",
        "Referer": "https://fantasy.premierleague.com/"
    })
    
    # Login payload
    payload = {
        "login": email,
        "password": password,
        "app": "plfpl-web",
        "redirect_uri": "https://fantasy.premierleague.com/a/login"
    }
    
    try:
        # Perform login - follow redirects to get all cookies
        response = session.post(login_url, data=payload, allow_redirects=True)
        
        # Check response status
        if response.status_code not in [200, 302]:
            log.error(f"Login returned status {response.status_code}")
            log.debug(f"Response text: {response.text[:500]}")
        
        # Get all cookies from the session (across all domains)
        all_cookies = []
        for cookie in session.cookies:
            all_cookies.append(f"{cookie.name}={cookie.value}")
            log.debug(f"Cookie: {cookie.name}={cookie.value[:20]}... (domain: {cookie.domain})")
        
        # Look for the important cookies
        important_cookies = ['pl_profile', 'sessionid', 'csrftoken']
        found_cookies = {name: False for name in important_cookies}
        
        for cookie in session.cookies:
            if cookie.name in important_cookies:
                found_cookies[cookie.name] = True
        
        # A failed login (wrong password) typically still returns HTTP 200 with the
        # login form re-rendered, and still sets a csrftoken cookie for that page —
        # so "we got some cookies" is not proof of a successful login. Only an actual
        # authenticated-session cookie (pl_profile or sessionid) proves it worked.
        authenticated = found_cookies["pl_profile"] or found_cookies["sessionid"]

        # Build cookie string
        if all_cookies and authenticated:
            cookie_str = "; ".join(all_cookies)
            log.info(f"Received {len(all_cookies)} cookies")
            log.info(f"Important cookies found: {found_cookies}")
            
            # Save to environment
            os.environ["FPL_SESSION"] = cookie_str
            if save_env_path:
                _upsert_env_var(save_env_path, "FPL_SESSION", cookie_str)
                log.info(f"Saved FPL_SESSION to {save_env_path}")
            
            # Also try to get user's entry ID from the me endpoint
            try:
                me_response = session.get(
                    "https://fantasy.premierleague.com/api/me/",
                    headers={"Cookie": cookie_str}
                )
                if me_response.status_code == 200:
                    me_data = me_response.json()
                    entry_id = me_data.get("player", {}).get("entry")
                    if entry_id:
                        log.info(f"Found FPL entry ID: {entry_id}")
                        if save_env_path:
                            _upsert_env_var(save_env_path, "FPL_ENTRY_ID", str(entry_id))
                            log.info(f"Saved FPL_ENTRY_ID to {save_env_path}")
            except Exception as e:
                log.debug(f"Could not fetch entry ID: {e}")
            
            return cookie_str
        elif not all_cookies:
            raise RuntimeError("Login failed - no cookies received. Check email/password.")
        else:
            log.error(f"Login did not authenticate (status={response.status_code}, cookies found: {found_cookies})")
            raise RuntimeError(
                "Login failed - received cookies but no authenticated session (pl_profile/sessionid). "
                "Check email/password."
            )
        
    except requests.RequestException as e:
        log.error(f"Network error during login: {e}")
        raise RuntimeError(f"Failed to connect to FPL: {e}")
    except Exception as e:
        log.error(f"Login failed: {e}")
        raise RuntimeError(f"Failed to login to FPL: {e}")


def test_auth(entry_id: int, timeout: int = 20) -> int:
    """Ping /api/my-team/{entry_id}/ with whatever auth is set. Returns HTTP status."""
    hdrs = get_auth_headers()
    url = MYTEAM_URL_TMPL.format(entry_id=int(entry_id))
    r = requests.get(url, headers=hdrs, timeout=timeout)
    return r.status_code