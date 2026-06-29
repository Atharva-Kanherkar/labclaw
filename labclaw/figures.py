"""Fetch remote figure URLs and store them locally as PNG/JPEG.

The reader sends figures to Cerebras as base64 data URIs and does
NOT support external image URLs -- it silently drops any http(s) figure path.
So whoever discovers a figure must download it and store a local PNG/JPEG.
arXiv figures are often vector (SVG) or live inside a PDF and have to be
transcoded to a raster format before they are usable. This module owns that
fetch + transcode step.

No hard third-party dependency: PNG/JPEG pass straight through with the stdlib.
Other raster formats are transcoded with Pillow if it is installed; SVG/PDF use
optional converters. When no converter is available the figure is skipped and
the reason is reported, never crashing the scout.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# A fetcher returns the raw bytes at a URL. Injectable so tests stay offline.
BytesFetcher = Callable[[str], bytes]


@dataclass
class Figure:
    """A figure stored locally as PNG/JPEG, ready for the reader."""

    figure_id: str
    path: str  # local filesystem path, always .png/.jpg/.jpeg
    source_url: str
    alt_text: str = ""
    caption: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "figure_id": self.figure_id,
            "path": self.path,
            "source_url": self.source_url,
            "alt_text": self.alt_text,
            "caption": self.caption,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Figure":
        return cls(
            figure_id=d["figure_id"],
            path=d["path"],
            source_url=d.get("source_url", ""),
            alt_text=d.get("alt_text", ""),
            caption=d.get("caption", ""),
            metadata=d.get("metadata", {}),
        )


# Magic-byte signatures -> logical type.
def detect_image_type(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:2] == b"BM":
        return "bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if data[:5] == b"%PDF-":
        return "pdf"
    head = data[:512].lstrip()
    if head[:5].lower() == b"<?xml" or head[:4].lower() == b"<svg":
        return "svg"
    return "unknown"


RASTER_TRANSCODE = {"gif", "bmp", "webp", "tiff"}


def transcode_to_png(data: bytes, src_type: str) -> Optional[bytes]:
    """Best-effort conversion of a non-PNG/JPEG image to PNG bytes.

    Returns None (rather than raising) when no converter is available, so the
    caller can skip the figure and keep going.
    """
    if src_type in RASTER_TRANSCODE:
        try:
            import io

            from PIL import Image  # optional dependency
        except ImportError:
            return None
        try:
            with Image.open(io.BytesIO(data)) as img:
                buf = io.BytesIO()
                img.convert("RGBA" if img.mode in ("P", "LA") else "RGB").save(
                    buf, format="PNG"
                )
                return buf.getvalue()
        except Exception:
            return None
    if src_type == "svg":
        try:
            import cairosvg  # optional

            return cairosvg.svg2png(bytestring=data)
        except Exception:
            return None
    if src_type == "pdf":
        try:
            import io

            import pypdfium2 as pdfium  # optional

            pdf = pdfium.PdfDocument(data)
            page = pdf[0]
            pil = page.render(scale=2).to_pil()
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None
    return None


class FigureStore:
    """Downloads figure URLs and writes them to dest_dir as PNG/JPEG."""

    def __init__(self, dest_dir, fetcher: BytesFetcher) -> None:
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.fetcher = fetcher

    def store(
        self,
        url: str,
        *,
        figure_id: Optional[str] = None,
        alt_text: str = "",
        caption: str = "",
    ) -> Optional[Figure]:
        """Fetch one figure; return a local Figure or None if unusable.

        Skips (returns None) on download failure or an un-transcodable format,
        rather than raising, so a single bad figure never sinks a scout run.
        """
        figure_id = figure_id or hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        try:
            data = self.fetcher(url)
        except Exception:
            return None
        if not data:
            return None

        kind = detect_image_type(data)
        if kind in ("png", "jpeg"):
            ext = "png" if kind == "png" else "jpg"
            out = self.dest_dir / f"{figure_id}.{ext}"
            out.write_bytes(data)
            return Figure(figure_id, str(out), url, alt_text, caption)

        png = transcode_to_png(data, kind)
        if png is None:
            # Known gap: vector/PDF without a converter installed, or a corrupt
            # image. Reported via return None; caller records it in metadata.
            return None
        out = self.dest_dir / f"{figure_id}.png"
        out.write_bytes(png)
        return Figure(
            figure_id,
            str(out),
            url,
            alt_text,
            caption,
            metadata={"transcoded_from": kind},
        )
