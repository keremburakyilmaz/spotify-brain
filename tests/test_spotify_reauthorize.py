import base64
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs

from scripts.spotify_reauthorize import TOKEN_URL, exchange_code


class SpotifyReauthorizationHelperTests(unittest.TestCase):
    @patch("scripts.spotify_reauthorize.urlopen")
    def test_exchange_code_uses_basic_auth_and_returns_refresh_token(
        self, urlopen: Mock
    ) -> None:
        response = Mock(status=200)
        response.read.return_value = b'{"refresh_token":"fresh-token"}'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        urlopen.return_value = response

        token = exchange_code(
            "client-id",
            "client-secret",
            "authorization-code",
            "http://127.0.0.1:8888/callback",
        )

        self.assertEqual(token, "fresh-token")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, TOKEN_URL)
        self.assertEqual(request.get_method(), "POST")
        expected_credentials = base64.b64encode(
            b"client-id:client-secret"
        ).decode("ascii")
        self.assertEqual(
            request.get_header("Authorization"),
            f"Basic {expected_credentials}",
        )
        self.assertEqual(
            parse_qs(request.data.decode("utf-8")),
            {
                "grant_type": ["authorization_code"],
                "code": ["authorization-code"],
                "redirect_uri": ["http://127.0.0.1:8888/callback"],
            },
        )


if __name__ == "__main__":
    unittest.main()
