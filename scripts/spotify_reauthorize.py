#!/usr/bin/env python3
"""Reauthorize Spotify and replace the repository refresh-token secret."""

import argparse
import base64
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen


AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8888/callback"
SCOPES = "user-read-recently-played"


def load_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE entries without requiring python-dotenv."""
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()

            key, separator, value = line.partition("=")
            key = key.strip()
            if not separator or not key:
                continue

            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    result: Dict[str, str] = {}
    expected_state = ""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/callback":
            self.send_error(404)
            return

        query = parse_qs(parsed.query)
        state = query.get("state", [""])[0]
        if not secrets.compare_digest(state, self.expected_state):
            type(self).result = {"error": "state_mismatch"}
            self._respond(400, "Spotify authorization failed: state mismatch.")
            return

        if "error" in query:
            type(self).result = {"error": query["error"][0]}
            self._respond(400, "Spotify authorization was not completed.")
            return

        code = query.get("code", [""])[0]
        if not code:
            type(self).result = {"error": "missing_code"}
            self._respond(400, "Spotify authorization failed: no code returned.")
            return

        type(self).result = {"code": code}
        self._respond(
            200,
            "Spotify authorization succeeded. You can close this tab and return "
            "to the terminal.",
        )

    def _respond(self, status: int, message: str) -> None:
        body = (
            "<!doctype html><meta charset='utf-8'>"
            f"<title>Spotify Brain</title><p>{message}</p>"
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def wait_for_authorization(
    client_id: str, redirect_uri: str, timeout_seconds: int
) -> str:
    parsed_redirect = urlparse(redirect_uri)
    if (
        parsed_redirect.scheme != "http"
        or parsed_redirect.hostname != "127.0.0.1"
        or parsed_redirect.path != "/callback"
        or parsed_redirect.port is None
    ):
        raise ValueError(
            "SPOTIFY_REDIRECT_URI must use the loopback form "
            "http://127.0.0.1:<port>/callback."
        )

    state = secrets.token_urlsafe(32)
    OAuthCallbackHandler.expected_state = state
    OAuthCallbackHandler.result = {}

    params = {
        "response_type": "code",
        "client_id": client_id,
        "scope": SCOPES,
        "redirect_uri": redirect_uri,
        "state": state,
        "show_dialog": "true",
    }
    authorization_url = f"{AUTHORIZE_URL}?{urlencode(params)}"

    server = HTTPServer(("127.0.0.1", parsed_redirect.port), OAuthCallbackHandler)
    server.timeout = 1
    print("Opening Spotify sign-in in your browser.")
    print(f"If it does not open automatically, visit:\n{authorization_url}\n")
    browser_opened = webbrowser.open_new_tab(authorization_url)
    if browser_opened:
        print("Waiting for you to finish Spotify sign-in in the browser...")
    else:
        print("The browser did not open; copy the URL above into your browser.")
        print("Keep this terminal running while you complete sign-in...")

    deadline = time.monotonic() + timeout_seconds
    try:
        while not OAuthCallbackHandler.result and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()

    result = OAuthCallbackHandler.result
    if not result:
        raise TimeoutError("Timed out waiting for Spotify authorization.")
    if "error" in result:
        raise RuntimeError(f"Spotify authorization failed: {result['error']}")
    return result["code"]


def exchange_code(
    client_id: str, client_secret: str, code: str, redirect_uri: str
) -> str:
    request_body = urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        }
    ).encode("utf-8")
    basic_credentials = base64.b64encode(
        f"{client_id}:{client_secret}".encode("utf-8")
    ).decode("ascii")
    request = Request(
        TOKEN_URL,
        data=request_body,
        headers={
            "Authorization": f"Basic {basic_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            status_code = response.status
            response_body = response.read()
    except HTTPError as exc:
        status_code = exc.code
        response_body = exc.read()

    try:
        token_data = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Spotify token exchange failed with HTTP {status_code}."
        ) from exc

    if not 200 <= status_code < 300:
        description = token_data.get("error_description", token_data.get("error"))
        raise RuntimeError(f"Spotify token exchange failed: {description}")

    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        raise RuntimeError("Spotify did not return a refresh token.")
    return refresh_token


def set_github_secret(refresh_token: str) -> None:
    if not shutil.which("gh"):
        raise RuntimeError(
            "GitHub CLI (`gh`) is not installed. Run without --github-secret and "
            "copy the returned token into the repository secret manually."
        )

    subprocess.run(
        ["gh", "secret", "set", "SPOTIFY_REFRESH_TOKEN"],
        input=refresh_token,
        text=True,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Obtain a fresh Spotify refresh token via Authorization Code flow."
    )
    parser.add_argument(
        "--github-secret",
        action="store_true",
        help="store the token as the current repository's GitHub Actions secret",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="seconds to wait for browser authorization (default: 300)",
    )
    return parser.parse_args()


def main() -> int:
    load_env_file()
    args = parse_args()
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", DEFAULT_REDIRECT_URI)

    if not client_id or not client_secret:
        print(
            "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env or "
            "the environment.",
            file=sys.stderr,
        )
        return 2

    try:
        code = wait_for_authorization(client_id, redirect_uri, args.timeout)
        refresh_token = exchange_code(
            client_id, client_secret, code, redirect_uri
        )
        if args.github_secret:
            set_github_secret(refresh_token)
            print("Updated GitHub Actions secret SPOTIFY_REFRESH_TOKEN.")
        else:
            print("\nNew SPOTIFY_REFRESH_TOKEN (do not commit this value):")
            print(refresh_token)
            print(
                "\nReplace the SPOTIFY_REFRESH_TOKEN repository secret, then "
                "rerun the failed workflow."
            )
    except (
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "\nSpotify authorization canceled. Run the command again and keep "
            "the terminal open until browser sign-in finishes.",
            file=sys.stderr,
        )
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
