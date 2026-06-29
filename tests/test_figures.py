import base64

import pytest

from labclaw.figures import FigureStore, detect_image_type, transcode_to_png

# 1x1 transparent PNG and 1x1 GIF, as recorded bytes (no network).
PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
GIF_1x1 = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
SVG = b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>'
PDF = b"%PDF-1.4\n..."


def test_detect_image_type():
    assert detect_image_type(PNG_1x1) == "png"
    assert detect_image_type(GIF_1x1) == "gif"
    assert detect_image_type(b"\xff\xd8\xff\xe0junk") == "jpeg"
    assert detect_image_type(SVG) == "svg"
    assert detect_image_type(PDF) == "pdf"
    assert detect_image_type(b"not an image") == "unknown"


def test_store_png_passthrough(tmp_path):
    store = FigureStore(tmp_path, lambda url: PNG_1x1)
    fig = store.store("https://x/img.png", alt_text="a plot")
    assert fig is not None
    assert fig.path.endswith(".png")
    assert fig.alt_text == "a plot"
    from pathlib import Path

    assert Path(fig.path).read_bytes() == PNG_1x1  # stored verbatim


def test_store_unknown_format_skips(tmp_path):
    store = FigureStore(tmp_path, lambda url: b"garbage bytes")
    assert store.store("https://x/bad") is None


def test_store_download_failure_skips(tmp_path):
    def boom(url):
        raise OSError("connection reset")

    store = FigureStore(tmp_path, boom)
    assert store.store("https://x/img.png") is None


def test_store_gif_transcodes_to_png(tmp_path):
    pytest.importorskip("PIL")  # transcode needs Pillow
    store = FigureStore(tmp_path, lambda url: GIF_1x1)
    fig = store.store("https://x/anim.gif")
    assert fig is not None
    assert fig.path.endswith(".png")
    assert fig.metadata.get("transcoded_from") == "gif"
    from pathlib import Path

    assert detect_image_type(Path(fig.path).read_bytes()) == "png"


def test_transcode_returns_none_without_converter():
    # PDF has no stdlib converter; without the optional dep it must skip cleanly.
    if transcode_to_png(PDF, "pdf") is None:
        assert True
    else:
        # If a converter happens to be installed, it should yield PNG bytes.
        assert detect_image_type(transcode_to_png(PDF, "pdf")) == "png"
