import os
import sys
import unittest
from unittest.mock import Mock, patch

try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pandas"] = Mock()

try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    sys.modules["requests"] = Mock()

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    sys.modules["dotenv"] = Mock()

from src.ingestion.spotify_ingest import (
    SpotifyIngester,
    SpotifyReauthorizationRequired,
)


CREDENTIALS = {
    "SPOTIFY_CLIENT_ID": "client-id",
    "SPOTIFY_CLIENT_SECRET": "client-secret",
    "SPOTIFY_REFRESH_TOKEN": "refresh-token",
}


class SpotifyAuthenticationTests(unittest.TestCase):
    @patch.dict(os.environ, CREDENTIALS, clear=False)
    @patch("src.ingestion.spotify_ingest.requests.post")
    def test_invalid_grant_requires_reauthorization(self, post: Mock) -> None:
        response = Mock(status_code=400)
        response.json.return_value = {
            "error": "invalid_grant",
            "error_description": "Refresh token expired",
        }
        post.return_value = response
        ingester = SpotifyIngester()

        with self.assertRaisesRegex(
            SpotifyReauthorizationRequired, "must not be retried"
        ):
            ingester._refresh_access_token()

        self.assertIsNone(ingester.refresh_token)
        self.assertIsNone(ingester.access_token)
        post.assert_called_once_with(
            "https://accounts.spotify.com/api/token",
            auth=("client-id", "client-secret"),
            data={
                "grant_type": "refresh_token",
                "refresh_token": "refresh-token",
            },
            timeout=30,
        )

    @patch.dict(os.environ, CREDENTIALS, clear=False)
    @patch("src.ingestion.spotify_ingest.requests.post")
    def test_refresh_uses_rotated_refresh_token(self, post: Mock) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {
            "access_token": "access-token",
            "refresh_token": "rotated-refresh-token",
            "expires_in": 3600,
        }
        post.return_value = response
        ingester = SpotifyIngester()

        self.assertEqual(ingester._refresh_access_token(), "access-token")
        self.assertEqual(ingester.refresh_token, "rotated-refresh-token")


if __name__ == "__main__":
    unittest.main()
