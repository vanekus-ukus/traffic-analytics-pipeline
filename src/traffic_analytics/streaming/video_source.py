from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import subprocess
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from sqlalchemy import text

from traffic_analytics.config.settings import Settings

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
YTDLP_BIN = PROJECT_ROOT / ".venv" / "bin" / "yt-dlp"
MEDIA_SUFFIXES = {".mp4", ".m3u8", ".mov", ".avi", ".mkv", ".webm"}


@dataclass(slots=True)
class VideoResolution:
    requested_uri: str
    resolved_uri: str
    local_path: Path
    source_type: str
    status: str
    message: str
    duration_sec: float | None
    fps: float | None
    width: int | None
    height: int | None


@dataclass(slots=True)
class MediaRequest:
    url: str
    headers: dict[str, str]
    format_id: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    protocol: str | None = None
    source_type: str = "remote"
    validation_error: str | None = None


def _headers_arg(headers: dict[str, str] | None) -> list[str]:
    if not headers:
        return []
    header_text = "".join(f"{key}: {value}\r\n" for key, value in headers.items())
    return ["-headers", header_text]


def _probe_video(source: str, headers: dict[str, str] | None = None) -> dict[str, object] | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        *_headers_arg(headers),
        source,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception:
        return None


def _parse_probe_info(probe: dict[str, object] | None) -> tuple[float | None, float | None, int | None, int | None]:
    if not probe:
        return None, None, None, None
    streams = probe.get("streams") or []
    stream = streams[0] if streams else {}
    width = stream.get("width")
    height = stream.get("height")
    fps = None
    rate = stream.get("r_frame_rate")
    if isinstance(rate, str) and "/" in rate:
        num, den = rate.split("/", 1)
        if den != "0":
            fps = float(num) / float(den)
    duration = probe.get("format", {}).get("duration")
    return float(duration) if duration else None, fps, width, height


def _run_yt_dlp(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(YTDLP_BIN), *args], capture_output=True, text=True, check=True)


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _looks_like_direct_media_url(value: str) -> bool:
    suffix = Path(urlparse(value).path).suffix.lower()
    return suffix in MEDIA_SUFFIXES


def _resolve_direct_url(url: str) -> str | None:
    probe = _probe_video(url)
    return url if probe else None


def _resolve_page_url_with_ytdlp(url: str) -> str | None:
    if not YTDLP_BIN.exists():
        return None
    try:
        result = _run_yt_dlp(["-f", "best[height<=720]/best", "--get-url", url])
        resolved = result.stdout.strip().splitlines()[0].strip()
        return resolved or None
    except Exception as exc:
        LOGGER.warning("Failed to resolve direct media URL via yt-dlp: %s", exc)
        return None


def _fetch_text(url: str) -> str | None:
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=15) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        LOGGER.warning("Failed to fetch %s: %s", url, exc)
        return None


def _resolve_page_url_from_player_config(url: str) -> MediaRequest | None:
    html = _fetch_text(url)
    if not html:
        return None
    match = re.search(r'<script[^>]+src="([^"]*config\.js)"', html, flags=re.IGNORECASE)
    if not match:
        return None
    config_url = urljoin(url, match.group(1))
    config_text = _fetch_text(config_url)
    if not config_text:
        return None
    json_match = re.search(r"playerConfig\s*=\s*(\{.*\})\s*;?\s*$", config_text.strip(), flags=re.DOTALL)
    if not json_match:
        return None
    try:
        payload = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return None
    source_path = payload.get("source")
    if not isinstance(source_path, str) or not source_path:
        return None
    media_url = urljoin(url, source_path)
    return MediaRequest(url=media_url, headers={}, source_type="remote_page_direct")


def resolve_media_url(url: str) -> str | None:
    if _is_url(url) and _looks_like_direct_media_url(url):
        return _resolve_direct_url(url)
    if _is_url(url):
        return _resolve_page_url_with_ytdlp(url)
    path = Path(url)
    return str(path) if path.exists() else None


def resolve_media_request(url: str) -> MediaRequest | None:
    if not _is_url(url):
        path = Path(url)
        return MediaRequest(url=str(path), headers={}, source_type="local") if path.exists() else None
    if _looks_like_direct_media_url(url):
        resolved = _resolve_direct_url(url)
        return MediaRequest(url=resolved, headers={}) if resolved else None
    if not YTDLP_BIN.exists():
        return None
    try:
        result = _run_yt_dlp(["--dump-single-json", url])
        payload = json.loads(result.stdout)
        requested = (payload.get("requested_downloads") or [{}])[0]
        media_url = requested.get("url") or payload.get("url")
        headers = requested.get("http_headers") or payload.get("http_headers") or {}
        if media_url:
            return MediaRequest(url=media_url, headers=headers, source_type="remote_page_direct")
    except Exception as exc:
        LOGGER.warning("Failed to resolve media request via yt-dlp metadata: %s", exc)
    config_request = _resolve_page_url_from_player_config(url)
    if config_request:
        return config_request
    resolved = _resolve_page_url_with_ytdlp(url)
    return MediaRequest(url=resolved, headers={}, source_type="remote_page_direct") if resolved else None


def _build_media_request_from_format(fmt: dict[str, object], default_headers: dict[str, str]) -> MediaRequest | None:
    media_url = fmt.get("url")
    if not isinstance(media_url, str) or not media_url:
        return None
    headers = fmt.get("http_headers")
    if not isinstance(headers, dict):
        headers = default_headers
    fps = fmt.get("fps")
    return MediaRequest(
        url=media_url,
        headers=headers,
        format_id=fmt.get("format_id") if isinstance(fmt.get("format_id"), str) else None,
        width=int(fmt["width"]) if fmt.get("width") else None,
        height=int(fmt["height"]) if fmt.get("height") else None,
        fps=float(fps) if fps else None,
        protocol=fmt.get("protocol") if isinstance(fmt.get("protocol"), str) else None,
        source_type="remote_candidate",
    )


def resolve_media_candidates(url: str) -> list[MediaRequest]:
    if not _is_url(url):
        path = Path(url)
        return [MediaRequest(url=str(path), headers={}, source_type="local")] if path.exists() else []
    if _looks_like_direct_media_url(url):
        resolved = _resolve_direct_url(url)
        return [MediaRequest(url=resolved, headers={}, source_type="remote_direct")] if resolved else []
    if not YTDLP_BIN.exists():
        return []
    candidates: list[MediaRequest] = []
    try:
        result = _run_yt_dlp(["--dump-single-json", url])
        payload = json.loads(result.stdout)
        default_headers = payload.get("http_headers") if isinstance(payload.get("http_headers"), dict) else {}
        requested = payload.get("requested_downloads") or []
        for item in requested:
            if isinstance(item, dict):
                request = _build_media_request_from_format(item, default_headers)
                if request:
                    candidates.append(request)
        formats = payload.get("formats") or []
        for fmt in formats:
            if not isinstance(fmt, dict):
                continue
            request = _build_media_request_from_format(fmt, default_headers)
            if request:
                candidates.append(request)
    except Exception as exc:
        LOGGER.warning("Failed to resolve media candidates via yt-dlp metadata: %s", exc)
        config_request = _resolve_page_url_from_player_config(url)
        if config_request:
            return [config_request]
        fallback = resolve_media_request(url)
        return [fallback] if fallback else []

    deduped: list[MediaRequest] = []
    seen: set[tuple[str, str | None]] = set()
    for candidate in candidates:
        key = (candidate.url, candidate.format_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    deduped.sort(
        key=lambda item: (
            item.height or 0,
            item.width or 0,
            item.fps or 0.0,
            1 if item.protocol and "m3u8" in item.protocol else 0,
        ),
        reverse=True,
    )
    return deduped


def validate_media_request(request: MediaRequest) -> MediaRequest | None:
    probe = _probe_video(request.url, request.headers)
    duration, fps, width, height = _parse_probe_info(probe)
    if width is None or height is None:
        request.validation_error = "ffprobe_returned_no_video_stream"
        return None
    request.width = width
    request.height = height
    request.fps = fps or request.fps
    request.validation_error = None
    return request


def resolve_validated_media_request(url: str) -> MediaRequest | None:
    candidates = resolve_media_candidates(url)
    for candidate in candidates:
        validated = validate_media_request(candidate)
        if validated:
            LOGGER.info(
                "Selected media candidate format=%s resolution=%sx%s protocol=%s",
                validated.format_id,
                validated.width,
                validated.height,
                validated.protocol,
            )
            return validated
    return None


def _cache_preview_clip(url: str, settings: Settings) -> Path | None:
    cache_path = settings.vk_cache_path
    part_path = cache_path.with_suffix(".mp4.part")
    if cache_path.exists() and cache_path.stat().st_size > 1024:
        return cache_path
    if part_path.exists() and part_path.stat().st_size > 1024:
        return part_path
    if not YTDLP_BIN.exists():
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_template = str(cache_path.with_suffix(".%(ext)s"))
        _run_yt_dlp(
            [
                "-f",
                "best[height<=720]/best",
                "--download-sections",
                f"*00:00:00-00:{settings.stream_max_seconds // 60:02d}:{settings.stream_max_seconds % 60:02d}",
                "--force-keyframes-at-cuts",
                "--merge-output-format",
                "mp4",
                "-o",
                output_template,
                url,
            ]
        )
        generated = list(cache_path.parent.glob(f"{cache_path.stem}.*"))
        for candidate in generated:
            if candidate.suffix == ".part":
                continue
            if candidate.exists() and candidate.stat().st_size > 1024:
                if candidate != cache_path:
                    candidate.replace(cache_path)
                return cache_path
        if part_path.exists() and part_path.stat().st_size > 1024:
            return part_path
    except Exception as exc:
        LOGGER.warning("Failed to cache preview clip: %s", exc)
        if part_path.exists() and part_path.stat().st_size > 1024:
            return part_path
    return None


def _resolved_video_resolution(
    requested_uri: str,
    resolved_uri: str,
    local_path: Path,
    source_type: str,
    status: str,
    message: str,
) -> VideoResolution:
    probe = _probe_video(resolved_uri if source_type.endswith("direct") else str(local_path))
    duration, fps, width, height = _parse_probe_info(probe)
    return VideoResolution(
        requested_uri=requested_uri,
        resolved_uri=resolved_uri,
        local_path=local_path,
        source_type=source_type,
        status=status,
        message=message,
        duration_sec=duration,
        fps=fps,
        width=width,
        height=height,
    )


def _iter_requested_sources(settings: Settings) -> list[str]:
    sources: list[str] = []
    for candidate in [settings.stream_source, *settings.stream_source_pool, settings.vk_url]:
        if not candidate:
            continue
        if candidate not in sources:
            sources.append(candidate)
    return sources


def resolve_video_source(settings: Settings) -> VideoResolution:
    fallback = settings.video_fallback or settings.vk_cache_path
    requested_sources = _iter_requested_sources(settings)

    for requested in requested_sources:
        requested_path = Path(requested)
        if requested and requested_path.exists():
            return _resolved_video_resolution(
                requested_uri=requested,
                resolved_uri=str(requested_path),
                local_path=requested_path,
                source_type="local_direct",
                status="resolved_local_direct",
                message="Using local video file provided as stream source.",
            )

        if not (requested and _is_url(requested) and settings.enable_network):
            continue

        if _looks_like_direct_media_url(requested):
            resolved = _resolve_direct_url(requested)
            if resolved:
                return _resolved_video_resolution(
                    requested_uri=requested,
                    resolved_uri=resolved,
                    local_path=fallback,
                    source_type="remote_direct",
                    status="resolved_remote_direct",
                    message="Using direct remote media URL.",
                )

        media_request = resolve_validated_media_request(requested)
        if media_request:
            direct_probe = _probe_video(media_request.url, media_request.headers)
            if direct_probe:
                duration, fps, width, height = _parse_probe_info(direct_probe)
                return VideoResolution(
                    requested_uri=requested,
                    resolved_uri=media_request.url,
                    local_path=fallback,
                    source_type="remote_page_direct",
                    status="resolved_remote_page_direct",
                    message=f"Using validated media candidate {media_request.format_id or 'unknown'} from remote page.",
                    duration_sec=duration,
                    fps=fps,
                    width=width,
                    height=height,
                )

        cached = _cache_preview_clip(requested, settings)
        if cached:
            return _resolved_video_resolution(
                requested_uri=requested,
                resolved_uri=str(cached),
                local_path=cached,
                source_type="remote_cached_preview" if cached.suffix != ".part" else "remote_partial_cache",
                status="resolved_remote_cached_preview" if cached.suffix != ".part" else "resolved_remote_partial_cache",
                message="Using locally cached preview clip from remote source."
                if cached.suffix != ".part"
                else "Using partially downloaded preview clip from remote source.",
            )

    return _resolved_video_resolution(
        requested_uri=requested_sources[0] if requested_sources else settings.stream_source,
        resolved_uri=str(fallback),
        local_path=fallback,
        source_type="local_fallback",
        status="resolved_local_fallback",
        message="Remote source unavailable or unreadable, using local fallback video.",
    )


def register_video_source(engine, resolution: VideoResolution) -> int:
    with engine.begin() as conn:
        video_source_id = conn.execute(
            text(
                """
                INSERT INTO raw.video_sources (
                    source_type, requested_uri, resolved_uri, local_path, status, message,
                    duration_sec, fps, width, height
                )
                VALUES (
                    :source_type, :requested_uri, :resolved_uri, :local_path, :status, :message,
                    :duration_sec, :fps, :width, :height
                )
                RETURNING video_source_id
                """
            ),
            {
                "source_type": resolution.source_type,
                "requested_uri": resolution.requested_uri,
                "resolved_uri": resolution.resolved_uri,
                "local_path": str(resolution.local_path),
                "status": resolution.status,
                "message": resolution.message,
                "duration_sec": resolution.duration_sec,
                "fps": resolution.fps,
                "width": resolution.width,
                "height": resolution.height,
            },
        ).scalar_one()
    return int(video_source_id)
