import pytest

from labclaw.telegram import (
    CommandRouter,
    LabClawBot,
    TelegramClient,
    TelegramConfig,
    TelegramError,
    parse_command,
)


class FakeTransport:
    """Records calls and returns queued responses, no network involved."""

    def __init__(self, responses=None):
        self.calls = []
        self._responses = list(responses or [])

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        if self._responses:
            return self._responses.pop(0)
        return {"ok": True, "result": {"message_id": 1}}


def make_client(responses=None, default_chat_id="555"):
    config = TelegramConfig(token="TOKEN", default_chat_id=default_chat_id)
    transport = FakeTransport(responses)
    return TelegramClient(config, transport=transport), transport


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_config_from_env_reads_token_and_chat():
    config = TelegramConfig.from_env(
        {"TELEGRAM_BOT_TOKEN": "abc", "TELEGRAM_CHAT_ID": "42"}
    )
    assert config.token == "abc"
    assert config.default_chat_id == "42"


def test_config_from_env_requires_token():
    with pytest.raises(RuntimeError):
        TelegramConfig.from_env({})


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


def test_send_message_builds_request_and_uses_default_chat():
    client, transport = make_client()
    client.send_message("hello")
    url, payload = transport.calls[0]
    assert url.endswith("/botTOKEN/sendMessage")
    assert payload == {"chat_id": "555", "text": "hello"}


def test_send_message_explicit_chat_overrides_default():
    client, transport = make_client()
    client.send_message("hi", chat_id="999", parse_mode="Markdown")
    _, payload = transport.calls[0]
    assert payload["chat_id"] == "999"
    assert payload["parse_mode"] == "Markdown"


def test_send_message_without_any_chat_raises():
    client, _ = make_client(default_chat_id=None)
    with pytest.raises(RuntimeError):
        client.send_message("hi")


def test_get_updates_passes_offset_and_timeout():
    client, transport = make_client([{"ok": True, "result": []}])
    client.get_updates(offset=7, timeout=0)
    _, payload = transport.calls[0]
    assert payload == {"timeout": 0, "offset": 7}


def test_api_error_raises():
    client, _ = make_client([{"ok": False, "description": "boom"}])
    with pytest.raises(TelegramError, match="boom"):
        client.get_me()


# --------------------------------------------------------------------------- #
# Command parsing + routing
# --------------------------------------------------------------------------- #


def test_parse_command_basic():
    assert parse_command("/read samples/a.md") == ("read", "samples/a.md")


def test_parse_command_strips_bot_mention():
    assert parse_command("/ping@LabClawBot") == ("ping", "")


def test_parse_command_ignores_plain_text():
    assert parse_command("just chatting") is None


def test_router_dispatch_and_unknown():
    router = CommandRouter()
    router.add("ping", lambda args, msg: "pong")
    assert router.dispatch({"text": "/ping"}) == "pong"
    assert "Unknown command" in router.dispatch({"text": "/nope"})


# --------------------------------------------------------------------------- #
# Bot behaviour
# --------------------------------------------------------------------------- #


def make_update(update_id, text, chat_id=100):
    return {
        "update_id": update_id,
        "message": {"text": text, "chat": {"id": chat_id}},
    }


def test_bot_ping_replies_and_tracks_offset():
    client, transport = make_client()
    bot = LabClawBot(client)
    bot.handle_update(make_update(10, "/ping", chat_id=321))

    _, payload = transport.calls[-1]
    assert payload == {"chat_id": "321", "text": "pong"}
    assert bot._offset == 11  # offset advances past the handled update


def test_bot_whoami_reports_chat_id():
    client, transport = make_client()
    bot = LabClawBot(client)
    bot.handle_update(make_update(1, "/whoami", chat_id=777))
    _, payload = transport.calls[-1]
    assert "777" in payload["text"]


def test_bot_plain_message_sends_no_reply():
    client, transport = make_client()
    bot = LabClawBot(client)
    reply = bot.handle_update(make_update(2, "hello there"))
    assert reply is None
    assert transport.calls == []  # nothing sent


def test_bot_read_missing_file_message():
    client, transport = make_client()
    bot = LabClawBot(client)
    bot.handle_update(make_update(3, "/read does/not/exist.md"))
    _, payload = transport.calls[-1]
    assert "File not found" in payload["text"]


def test_parse_read_args_flag_anywhere():
    assert LabClawBot._parse_read_args("--local samples/a.md") == (True, "samples/a.md")
    assert LabClawBot._parse_read_args("samples/a.md --local") == (True, "samples/a.md")
    assert LabClawBot._parse_read_args("samples/a.md") == (False, "samples/a.md")
    assert LabClawBot._parse_read_args("-l samples/a.md") == (True, "samples/a.md")



def test_bot_read_local_runs_offline(tmp_path, monkeypatch):
    # Key is set, but --local must force the offline parser (no SDK call).
    monkeypatch.setenv("CEREBRAS_API_KEY", "x")
    source = tmp_path / "claim.md"
    source.write_text("# Demo\nThe claim is a 2x speedup.\n", encoding="utf-8")
    client, transport = make_client()
    bot = LabClawBot(client)
    bot.handle_update(make_update(4, f"/read --local {source}"))
    _, payload = transport.calls[-1]
    assert "Demo" in payload["text"]
    assert "Read failed" not in payload["text"]


def test_poll_once_handles_batch_and_advances_offset():
    updates = [make_update(5, "/ping"), make_update(6, "/help")]
    responses = [
        {"ok": True, "result": updates},
        {"ok": True, "result": {"message_id": 1}},
        {"ok": True, "result": {"message_id": 2}},
    ]
    client, _ = make_client(responses)
    bot = LabClawBot(client)
    handled = bot.poll_once()
    assert handled == 2
    assert bot._offset == 7


# --------------------------------------------------------------------------- #
# Resilience: network errors, token redaction, allowed_updates
# --------------------------------------------------------------------------- #

import urllib.error

from labclaw.telegram import DEFAULT_ALLOWED_UPDATES, redact_token


def test_poll_once_sets_allowed_updates():
    client, transport = make_client([{"ok": True, "result": []}])
    bot = LabClawBot(client)
    bot.poll_once()
    _, payload = transport.calls[0]
    assert payload["allowed_updates"] == DEFAULT_ALLOWED_UPDATES


def test_poll_once_propagates_network_error():
    # Connection-level failures are not swallowed by the client; they surface
    # so run_polling can retry them.
    def transport(url, payload):
        raise urllib.error.URLError("name resolution failed")

    config = TelegramConfig(token="TOK", default_chat_id="1")
    client = TelegramClient(config, transport=transport)
    bot = LabClawBot(client)
    with pytest.raises(urllib.error.URLError):
        bot.poll_once()


def test_run_polling_survives_network_error_and_retries():
    calls = {"updates": 0}

    def transport(url, payload):
        if url.endswith("/getMe"):
            return {"ok": True, "result": {"username": "bot"}}
        calls["updates"] += 1
        if calls["updates"] == 1:
            raise urllib.error.URLError("dropped connection")  # first poll blows up
        return {"ok": True, "result": []}  # then recovers

    config = TelegramConfig(token="TOK", default_chat_id="1")
    client = TelegramClient(config, transport=transport)
    bot = LabClawBot(client)
    # Must not raise; must retry the poll after the error.
    bot.run_polling(idle_sleep=0, error_sleep=0, max_cycles=2)
    assert calls["updates"] >= 2


def test_run_polling_survives_oserror():
    def transport(url, payload):
        if url.endswith("/getMe"):
            return {"ok": True, "result": {"username": "bot"}}
        raise OSError("socket timeout")

    config = TelegramConfig(token="TOK", default_chat_id="1")
    client = TelegramClient(config, transport=transport)
    bot = LabClawBot(client)
    bot.run_polling(idle_sleep=0, error_sleep=0, max_cycles=3)  # should not crash


def test_redact_token_scrubs_token_from_error_text():
    leaked = "URLError at https://api.telegram.org/botSECRET123/getUpdates"
    cleaned = redact_token(leaked, "SECRET123")
    assert "SECRET123" not in cleaned
    assert "<token>" in cleaned


def test_run_polling_startup_survives_network_error(capsys):
    # get_me failing at startup must not crash the loop.
    def transport(url, payload):
        if url.endswith("/getMe"):
            raise urllib.error.URLError("boom")
        return {"ok": True, "result": []}

    config = TelegramConfig(token="TOK", default_chat_id="1")
    client = TelegramClient(config, transport=transport)
    bot = LabClawBot(client)
    bot.run_polling(idle_sleep=0, error_sleep=0, max_cycles=1)  # should not crash
