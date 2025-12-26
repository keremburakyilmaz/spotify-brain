import os
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

def _chunked(iterable: Iterable, size: int) -> List[List]:
    lst = list(iterable)
    return [lst[i : i + size] for i in range(0, len(lst), size)]

class SpotifyIngester:

    TOKEN_BUFFER_SECONDS = 60

    def __init__(self) -> None:
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.refresh_token = os.getenv("SPOTIFY_REFRESH_TOKEN")
        self.access_token: Optional[str] = None
        self.token_expires_at: float = 0.0

        if not self.client_id:
            raise ValueError(
                "SPOTIFY_CLIENT_ID not found in environment variables or .env file"
            )
        if not self.client_secret:
            raise ValueError(
                "SPOTIFY_CLIENT_SECRET not found in environment variables or .env file"
            )
        if not self.refresh_token:
            raise ValueError(
                "SPOTIFY_REFRESH_TOKEN not found in environment variables or .env file"
            )


    def _refresh_access_token(self) -> str:
        url = "https://accounts.spotify.com/api/token"
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        response = requests.post(url, data=data)
        if response.status_code != 200:
            error_msg = "Unknown error"
            try:
                error_data = response.json()
                error_msg = error_data.get(
                    "error_description",
                    error_data.get("error", str(response.status_code)),
                )
            except Exception:
                error_msg = response.text or f"HTTP {response.status_code}"

            raise ValueError(
                "Failed to refresh access token: "
                f"{error_msg}\n"
                "Check SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, and "
                "SPOTIFY_REFRESH_TOKEN in your environment/.env."
            )

        token_data = response.json()
        self.access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self.token_expires_at = (
            time.time() + expires_in - self.TOKEN_BUFFER_SECONDS
        )
        return self.access_token

    def _get_access_token(self) -> str:
        if not self.access_token or time.time() >= self.token_expires_at:
            self._refresh_access_token()
        return self.access_token

    def _spotify_get(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Dict:
        headers = {"Authorization": f"Bearer {self._get_access_token()}"}
        
        for attempt in range(max_retries):
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 401:
                self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self.access_token}"
                response = requests.get(url, headers=headers, params=params)

            # Retry on server errors (5xx) and rate limits (429)
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    if response.status_code == 429:
                        # For rate limits, check Retry-After header if available
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = int(retry_after)
                            except ValueError:
                                pass
                    print(f"Spotify API returned {response.status_code}. Retrying in {wait_time}s. (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", error_msg)
                    except Exception:
                        pass
                    raise requests.exceptions.HTTPError(
                        f"Spotify API error after {max_retries} attempts: {error_msg}",
                        response=response
                    )

            response.raise_for_status()
            return response.json()


    def fetch_recently_played(
        self, limit: int = 50, after: Optional[int] = None
    ) -> List[Dict]:
        url = "https://api.spotify.com/v1/me/player/recently-played"
        params = {"limit": min(limit, 50)}
        if after:
            params["after"] = after

        data = self._spotify_get(url, params)
        return data.get("items", [])

    def fetch_track_metadata(self, track_ids: List[str]) -> Dict[str, Dict]:
        metadata: Dict[str, Dict] = {}
        unique_ids = list(dict.fromkeys(track_ids))

        # Fetch basic track info (names)
        for batch in _chunked(unique_ids, 50): 
            ids_param = ",".join(batch)
            url = f"https://api.spotify.com/v1/tracks?ids={ids_param}"
            data = self._spotify_get(url)
            for track in data.get("tracks", []):
                if not track:
                    continue
                tid = track["id"]
                artists = track.get("artists", []) or []
                
                # Extract album artwork image URL
                # Images are sorted by size (largest first), use medium (300x300) or largest if not available
                album = track.get("album", {})
                images = album.get("images", [])
                image_url = None
                if images:
                    # Prefer medium size (300x300), fallback to largest
                    for img in images:
                        if img.get("width") == 300:
                            image_url = img.get("url")
                            break
                    if not image_url and images:
                        image_url = images[0].get("url")  # Use largest if no medium
                
                metadata[tid] = {
                    "track_name": track.get("name", ""),
                    "artist_name": ", ".join(a["name"] for a in artists),
                    "image_url": image_url,
                }

        return metadata


    @staticmethod
    def fetch_audio_features(track_ids: List[str], id_to_label: Optional[Dict[str, str]] = None) -> Dict[str, Dict]:
        base_url = "https://api.reccobeats.com/v1/audio-features"
        features_by_spotify_id: Dict[str, Dict] = {}

        unique_ids = list(dict.fromkeys([tid for tid in track_ids if tid]))

        for batch in _chunked(unique_ids, 40):
            try:
                params = [("ids", tid) for tid in batch]
                resp = requests.get(base_url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                if isinstance(data, dict) and "content" in data:
                    items = data["content"]
                elif isinstance(data, dict) and "audioFeatures" in data:
                    items = data["audioFeatures"]
                elif isinstance(data, dict) and "data" in data:
                    items = data["data"]
                elif isinstance(data, list):
                    items = data
                else:
                    items = []

                returned_spotify_ids = set()
                for item in items or []:
                    if not isinstance(item, dict):
                        continue

                    sid = None
                    href = item.get("href", "")
                    if href and "open.spotify.com/track/" in href:
                        try:
                            sid = href.split("/track/")[1].split("?")[0].split("/")[0]
                        except Exception:
                            pass
                    
                    if not sid:
                        sid = (
                            item.get("spotifyId")
                            or item.get("spotify_id")
                            or item.get("trackId")
                            or item.get("track_id")
                        )
                    
                    if sid and sid in batch:
                        features_by_spotify_id[sid] = item
                        returned_spotify_ids.add(sid)

                missing = [tid for tid in batch if tid not in returned_spotify_ids]
                for tid in missing:
                    label = (id_to_label or {}).get(tid, tid)
                    print(f"[ReccoBeats] Audio features not found for {label}")

                time.sleep(0.05)

            except requests.RequestException as exc:
                for tid in batch:
                    label = (id_to_label or {}).get(tid, tid)
                    print(f"[ReccoBeats] Error for {label}: {exc}")
                continue

        return features_by_spotify_id


def get_latest_timestamp(history_path: str) -> Optional[datetime]:
    if not os.path.exists(history_path):
        return None

    df = pd.read_parquet(history_path)
    if df.empty or "played_at" not in df.columns:
        return None

    if not pd.api.types.is_datetime64_any_dtype(df["played_at"]):
        df["played_at"] = pd.to_datetime(df["played_at"])

    return df["played_at"].max()


def assign_session_ids(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    if df.empty:
        df["session_id"] = []
        return df

    df = df.sort_values("played_at").reset_index(drop=True)
    df["session_id"] = 0

    current_session = 0
    df.loc[0, "session_id"] = current_session

    time_diffs = df["played_at"].diff()
    gap_threshold = pd.Timedelta(minutes=gap_minutes)

    for i in range(1, len(df)):
        if time_diffs.iloc[i] > gap_threshold:
            current_session += 1
        df.loc[i, "session_id"] = current_session

    return df

def ingest(history_path: str = "data/history.parquet", max_tracks: int = 50) -> pd.DataFrame:
    ingester = SpotifyIngester()

    if os.path.exists(history_path):
        existing_df = pd.read_parquet(history_path)
        if existing_df.empty:
            print("Existing history file is empty.")
            latest_timestamp = None
        else:
            latest_timestamp = get_latest_timestamp(history_path)
    else:
        existing_df = pd.DataFrame()
        latest_timestamp = None

    existing_pairs = set()
    if not existing_df.empty and {"track_id", "played_at"}.issubset(existing_df.columns):
        if not pd.api.types.is_datetime64_any_dtype(existing_df["played_at"]):
            existing_df["played_at"] = pd.to_datetime(existing_df["played_at"])
        existing_pairs = set(
            zip(
                existing_df["track_id"].astype(str),
                existing_df["played_at"].astype("int64"),  # ns since epoch
            )
        )

    after_timestamp_ms: Optional[int]
    if latest_timestamp:
        after_timestamp_ms = int(latest_timestamp.timestamp() * 1000)
    else:
        after_timestamp_ms = None

    print(f"Fetching recently played tracks (after: {after_timestamp_ms})")
    try:
        recent_items = ingester.fetch_recently_played(
            limit=max_tracks, after=after_timestamp_ms
        )
    except requests.exceptions.HTTPError as exc:
        print(f"Failed to fetch recently played tracks from Spotify: {exc}")
        print("Returning existing history without new tracks.")
        return existing_df if not existing_df.empty else pd.DataFrame()

    if not recent_items:
        print("No new tracks found.")
        return existing_df if not existing_df.empty else pd.DataFrame()

    # Filter by timestamp if history is not empty
    # If history is empty, get all tracks (don't filter by timestamp)
    new_records: List[Dict] = []

    for item in recent_items:
        played_at_str = item.get("played_at")
        if not played_at_str:
            continue
        played_at = pd.to_datetime(played_at_str)

        if latest_timestamp is not None:
            if played_at <= latest_timestamp:
                continue

        track = item.get("track") or {}
        track_id = track.get("id")
        if not track_id:
            continue

        # Check for duplicates using both track_id and played_at timestamp
        played_at_ns = pd.Timestamp(played_at).value  
        key = (str(track_id), played_at_ns)
        if key in existing_pairs:
            continue

        artist_names = ", ".join(a["name"] for a in track.get("artists", []) or [])

        new_records.append(
            {
                "track_id": track_id,
                "played_at": played_at,
                "track_name": track.get("name", ""),
                "artist_name": artist_names,
            }
        )

    if not new_records:
        print("No new tracks after filtering by timestamp.")
        return existing_df if not existing_df.empty else pd.DataFrame()

    print(f"Processing {len(new_records)} new tracks")
    new_df = pd.DataFrame(new_records)

    track_ids = new_df["track_id"].dropna().tolist()

    print("Fetching audio features from ReccoBeats")
    id_to_label = {
        r["track_id"]: f'{r.get("track_name","").strip()} — {r.get("artist_name","").strip()}'
        for _, r in new_df.iterrows()
        if pd.notna(r.get("track_id"))
    }

    audio_features = ingester.fetch_audio_features(track_ids, id_to_label=id_to_label)

    print("Fetching track metadata from Spotify")
    metadata = ingester.fetch_track_metadata(track_ids)

    audio_feature_cols = [
        "valence",
        "energy",
        "danceability",
        "acousticness",
        "instrumentalness",
        "tempo",
    ]

    for idx, row in new_df.iterrows():
        tid = row["track_id"]
        af = audio_features.get(tid)
        if af:
            for col in audio_feature_cols:
                new_df.loc[idx, col] = af.get(col)
            tempo = af.get("tempo", 0.0) or 0.0
            new_df.loc[idx, "tempo_norm"] = max(0.0, min(tempo / 250.0, 1.0))
        else:
            for col in audio_feature_cols:
                new_df.loc[idx, col] = None
            new_df.loc[idx, "tempo_norm"] = None

        meta = metadata.get(tid)
        if meta:
            new_df.loc[idx, "track_name"] = meta.get(
                "track_name", row.get("track_name", "")
            )
            new_df.loc[idx, "artist_name"] = meta.get(
                "artist_name", row.get("artist_name", "")
            )
            new_df.loc[idx, "image_url"] = meta.get("image_url")

    new_df["played_at"] = pd.to_datetime(new_df["played_at"])
    new_df = new_df.dropna(subset=["track_id", "played_at"])
    new_df = new_df.drop_duplicates(subset=["track_id", "played_at"], keep="first")
    new_df = new_df.dropna()
    
    # Save ingestion file (will be updated with session_ids later)
    ingestion_file = None
    if not new_df.empty:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ingestion_dir = os.path.join(os.path.dirname(history_path), "ingestions")
        os.makedirs(ingestion_dir, exist_ok=True)
        ingestion_file = os.path.join(ingestion_dir, f"ingestion_{timestamp_str}.parquet")
        new_df.to_parquet(ingestion_file, index=False)
        print(f"Saved {len(new_df)} truly new tracks to {ingestion_file}")

    if not existing_df.empty:
        all_cols = sorted(set(existing_df.columns) | set(new_df.columns))
        existing_df = existing_df.reindex(columns=all_cols)
        new_df = new_df.reindex(columns=all_cols)
        if not new_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = existing_df
    else:
        combined_df = new_df

    print("Assigning session IDs")
    combined_df = assign_session_ids(combined_df, gap_minutes=30)
    combined_df = combined_df.sort_values("played_at").reset_index(drop=True)
    
    # Update ingestion file with session_ids by extracting new tracks from combined_df
    new_tracks_with_sessions = None
    if ingestion_file and not new_df.empty:
        # Extract new tracks from combined_df using merge on (track_id, played_at)
        new_tracks_with_sessions = combined_df.merge(
            new_df[["track_id", "played_at"]],
            on=["track_id", "played_at"],
            how="inner"
        )
        
        if not new_tracks_with_sessions.empty:
            new_tracks_with_sessions.to_parquet(ingestion_file, index=False)
            print(f"Updated ingestion file with session IDs")
    print(
        f"Ingestion complete. New tracks this run: {len(new_df)}. "
        f"History will be updated at the end of the pipeline."
    )
    
    # Return the new tracks DataFrame with session IDs
    return new_tracks_with_sessions if new_tracks_with_sessions is not None and not new_tracks_with_sessions.empty else new_df

def update_history_from_ingestion(ingestion_file_path: str,
                                  history_path: str = "data/history.parquet") -> pd.DataFrame:
    # Load ingestion file
    if not os.path.exists(ingestion_file_path):
        raise FileNotFoundError(f"Ingestion file not found: {ingestion_file_path}")
    
    new_df = pd.read_parquet(ingestion_file_path)
    
    if new_df.empty:
        print("Ingestion file is empty, nothing to update")
        if os.path.exists(history_path):
            return pd.read_parquet(history_path)
        return pd.DataFrame()
    
    # Load existing history
    if os.path.exists(history_path):
        existing_df = pd.read_parquet(history_path)
    else:
        existing_df = pd.DataFrame()
    
    # Combine existing and new tracks
    if not existing_df.empty:
        all_cols = sorted(set(existing_df.columns) | set(new_df.columns))
        for col in all_cols:
            if col not in existing_df.columns:
                existing_df[col] = None
            if col not in new_df.columns:
                new_df[col] = None
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=["track_id", "played_at"], keep="last")
    else:
        combined_df = new_df.copy()
    
    # Drop rows with null values
    rows_before = len(combined_df)
    combined_df = combined_df.dropna()
    rows_dropped = rows_before - len(combined_df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows with null values")
    
    # Save updated history
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    combined_df.to_parquet(history_path, index=False)
    
    print(
        f"Updated history.parquet: Total tracks: {len(combined_df)}, "
        f"New tracks added: {len(new_df)}"
    )
    
    return combined_df


if __name__ == "__main__":
    df_history = ingest()
    print(f"\nIngested {len(df_history)} tracks total")
