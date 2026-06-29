"""Two-way Telegram bot for LabClaw using the raw Bot API over HTTP.

No third-party dependencies: HTTP is done with the standard library so the
bot works anywhere Python runs. The transport is injectable, which keeps the
client fully unit-testable without touching the network.

Quick start::

    export TELEGRAM_BOT_TOKEN=123456:abcdef
    export TELEGRAM_CHAT_ID=987654321        # optional default ping target
    python -m labclaw.telegram                # start the two-way bot
    python -m labclaw.telegram --send "hi"    # one-off ping, then exit

Talk to the bot in Telegram with /start, /help, /ping or /read <path>.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

API_ROOT = "https://api.telegram.org"
DEFAULT_TIMEOUT = 30
DEFAULT_POLL_TIMEOUT = 25
# Only the update types we actually handle, so Telegram doesn't ship the rest.
DEFAULT_ALLOWED_UPDATES = ["message", "edited_message"]

# A transport takes (url, payload_dict) and returns the decoded JSON response.
Transport = Callable[[str, dict], dict]

# A command handler takes (args_string, message_dict) and returns reply text.
CommandHandler = Callable[[str, dict], Optional[str]]


def redact_token(text: str, token: Optional[str]) -> str:
    """Strip the bot token from any string before it is logged or surfaced.

    urllib embeds the request URL (which contains /bot<TOKEN>/) in its error
    messages and tracebacks, so a raw network error would otherwise leak the
    token into logs.
    """
    if not token:
        return text
    return text.replace(token, "<token>")


@dataclass
class TelegramConfig:
    """Connection settings, normally sourced from the environment."""

    token: str
    default_chat_id: Optional[str] = None
    api_root: str = API_ROOT
    timeout: int = DEFAULT_TIMEOUT
    poll_timeout: int = DEFAULT_POLL_TIMEOUT

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "TelegramConfig":
        env = env if env is not None else os.environ
        token = env.get("TELEGRAM_BOT_TOKEN")
        if not token:
            raise RuntimeError(
                "Set TELEGRAM_BOT_TOKEN (get one from @BotFather on Telegram)."
            )
        return cls(
            token=token,
            default_chat_id=env.get("TELEGRAM_CHAT_ID"),
            api_root=env.get("TELEGRAM_API_ROOT", API_ROOT),
        )


def _urllib_transport(url: str, payload: dict, *, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # Telegram returns JSON on errors too.
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"ok": False, "description": f"HTTP {exc.code}: {body}"}
    # NOTE: connection-level failures (URLError, socket timeout, OSError) are
    # intentionally NOT caught here -- they propagate so run_polling can apply
    # its retry/backoff. Callers that surface these must redact the token.


class TelegramError(RuntimeError):
    """Raised when the Telegram API returns ok=false."""


class TelegramClient:
    """Thin wrapper over the Telegram Bot API methods LabClaw needs."""

    def __init__(self, config: TelegramConfig, transport: Optional[Transport] = None) -> None:
        self.config = config
        self._transport = transport

    def _call(self, method: str, params: dict) -> dict:
        url = f"{self.config.api_root}/bot{self.config.token}/{method}"
        if self._transport is not None:
            response = self._transport(url, params)
        else:
            response = _urllib_transport(url, params, timeout=self.config.timeout)
        if not response.get("ok", False):
            raise TelegramError(response.get("description", "unknown Telegram error"))
        return response["result"]

    def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        *,
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
    ) -> dict:
        """Send (ping) a message to a chat. Falls back to the default chat id."""
        target = chat_id or self.config.default_chat_id
        if not target:
            raise RuntimeError(
                "No chat_id given and TELEGRAM_CHAT_ID is unset; cannot send."
            )
        params: dict[str, Any] = {"chat_id": target, "text": text}
        if parse_mode:
            params["parse_mode"] = parse_mode
        if disable_notification:
            params["disable_notification"] = True
        return self._call("sendMessage", params)

    def get_updates(
        self,
        offset: Optional[int] = None,
        timeout: Optional[int] = None,
        allowed_updates: Optional[list] = None,
    ) -> list:
        """Long-poll for incoming updates (messages, commands)."""
        params: dict[str, Any] = {
            "timeout": self.config.poll_timeout if timeout is None else timeout
        }
        if offset is not None:
            params["offset"] = offset
        if allowed_updates is not None:
            params["allowed_updates"] = allowed_updates
        return self._call("getUpdates", params)

    def get_me(self) -> dict:
        return self._call("getMe", {})


# --------------------------------------------------------------------------- #
# Command routing
# --------------------------------------------------------------------------- #


def parse_command(text: str) -> Optional[tuple[str, str]]:
    """Return (command, args) for a '/cmd args' message, else None.

    Handles the '/cmd@BotName args' form Telegram uses in group chats.
    """
    if not text or not text.startswith("/"):
        return None
    head, _, args = text.partition(" ")
    command = head[1:].split("@", 1)[0].lower()
    if not command:
        return None
    return command, args.strip()


@dataclass
class CommandRouter:
    """Maps command names to handlers and dispatches incoming messages."""

    handlers: dict[str, CommandHandler] = field(default_factory=dict)
    fallback: Optional[CommandHandler] = None

    def register(self, name: str) -> Callable[[CommandHandler], CommandHandler]:
        def decorator(func: CommandHandler) -> CommandHandler:
            self.handlers[name.lower()] = func
            return func

        return decorator

    def add(self, name: str, handler: CommandHandler) -> None:
        self.handlers[name.lower()] = handler

    def dispatch(self, message: dict) -> Optional[str]:
        text = message.get("text", "")
        parsed = parse_command(text)
        if parsed is None:
            return self.fallback("", message) if self.fallback else None
        command, args = parsed
        handler = self.handlers.get(command)
        if handler is None:
            if self.fallback:
                return self.fallback(text, message)
            known = ", ".join(f"/{name}" for name in sorted(self.handlers))
            return f"Unknown command. Try: {known}"
        return handler(args, message)


# --------------------------------------------------------------------------- #
# LabClaw bot
# --------------------------------------------------------------------------- #


class LabClawBot:
    """Wires LabClaw capabilities to Telegram commands and runs the poll loop."""

    def __init__(self, client: TelegramClient, router: Optional[CommandRouter] = None) -> None:
        self.client = client
        self.router = router or CommandRouter()
        self._offset: Optional[int] = None
        if not self.router.handlers:
            self._register_default_commands()

    def _register_default_commands(self) -> None:
        router = self.router

        @router.register("start")
        def _start(args: str, message: dict) -> str:
            return (
                "LabClaw bot online. I fact-check ML/code claims.\n"
                "Commands: /help /ping /read <path>"
            )

        @router.register("help")
        def _help(args: str, message: dict) -> str:
            return (
                "Commands:\n"
                "/ping - check I'm alive\n"
                "/read [--local] <path> - extract claim cards from a source file\n"
                "/whoami - show your chat id\n"
                "/help - this message"
            )

        @router.register("ping")
        def _ping(args: str, message: dict) -> str:
            return "pong"

        @router.register("whoami")
        def _whoami(args: str, message: dict) -> str:
            chat = message.get("chat", {})
            return f"chat_id: {chat.get('id', 'unknown')}"

        @router.register("read")
        def _read(args: str, message: dict) -> str:
            return self._handle_read(args)

    def _handle_read(self, args: str) -> str:
        force_local, path_str = self._parse_read_args(args)
        if not path_str:
            return "Usage: /read [--local] <path-to-source-file>"
        source = Path(path_str)
        if not source.exists():
            return f"File not found: {path_str}"
        try:
            # Imported lazily so the bot loads without the reader's deps.
            from labclaw.multimodal_reader import format_human_result, read_source

            # KNOWN LIMITATION: this runs synchronously inside the poll loop, so
            # a live Gemma extraction blocks all other updates and pauses polling
            # until it returns. Acceptable for a single-user MVP; move to a
            # worker/queue if concurrency is needed.
            use_gemma = not force_local and bool(os.environ.get("CEREBRAS_API_KEY"))
            try:
                result = read_source(source, use_gemma=use_gemma)
            except RuntimeError as exc:
                if not use_gemma:
                    raise
                # Cerebras SDK/key missing -- fall back so /read still works.
                result = read_source(source, use_gemma=False)
                note = f"(live reader unavailable: {exc}; used offline parser)\n"
                return note + format_human_result(result)
            return format_human_result(result)
        except Exception as exc:  # Surface reader errors to the chat, don't crash.
            return f"Read failed: {exc}"

    @staticmethod
    def _parse_read_args(args: str) -> tuple[bool, str]:
        """Split /read args into (force_local, path). Accepts --local anywhere."""
        force_local = False
        parts = []
        for token in args.split():
            if token in ("--local", "-l"):
                force_local = True
            else:
                parts.append(token)
        return force_local, " ".join(parts).strip()

    def handle_update(self, update: dict) -> Optional[str]:
        """Process one update; send a reply if a handler produced one.

        Delivery is intentionally AT MOST ONCE: the offset advances as soon as
        we see the update, before the reply is sent. If send_message fails the
        reply is lost rather than redelivered. This is deliberate -- advancing
        only after a successful send would let a permanently-failing update
        (poison pill) wedge the bot in an infinite reprocess loop.
        """
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            self._offset = update_id + 1
        message = update.get("message") or update.get("edited_message")
        if not message:
            return None
        reply = self.router.dispatch(message)
        if reply:
            chat_id = str(message.get("chat", {}).get("id"))
            self.client.send_message(reply, chat_id=chat_id)
        return reply

    def poll_once(self) -> int:
        """Fetch and handle one batch of updates. Returns the count handled.

        Network errors propagate to the caller (run_polling), which retries.
        """
        updates = self.client.get_updates(
            offset=self._offset, allowed_updates=DEFAULT_ALLOWED_UPDATES
        )
        for update in updates:
            self.handle_update(update)
        return len(updates)

    def run_polling(
        self,
        *,
        idle_sleep: float = 1.0,
        error_sleep: float = 5.0,
        max_cycles: Optional[int] = None,
    ) -> None:
        """Block (almost) forever, long-polling for updates. Ctrl+C to stop.

        Survives transient failures: Telegram API errors AND connection-level
        failures (dropped connection, DNS hiccup, socket timeout) are caught and
        retried after error_sleep, so an always-on bot doesn't die on the first
        network blip. max_cycles bounds the loop for testing.
        """
        token = self.client.config.token
        try:
            me = self.client.get_me()
            print(f"LabClaw bot running as @{me.get('username', '?')}. Ctrl+C to stop.")
        except (TelegramError, urllib.error.URLError, OSError) as exc:
            print(f"Startup warning: {redact_token(str(exc), token)}", file=sys.stderr)

        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            cycles += 1
            try:
                handled = self.poll_once()
                if handled == 0:
                    time.sleep(idle_sleep)
            except (TelegramError, urllib.error.URLError, OSError) as exc:
                print(f"Polling error: {redact_token(str(exc), token)}", file=sys.stderr)
                time.sleep(error_sleep)
            except KeyboardInterrupt:
                print("\nStopped.")
                return


def notify(text: str, chat_id: Optional[str] = None) -> dict:
    """One-shot helper: ping a chat using env config. Returns the sent message."""
    client = TelegramClient(TelegramConfig.from_env())
    return client.send_message(text, chat_id=chat_id)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="LabClaw Telegram bot.")
    parser.add_argument("--send", metavar="TEXT", help="Send one message and exit.")
    parser.add_argument("--chat-id", help="Override target chat id for --send.")
    parser.add_argument(
        "--once", action="store_true", help="Poll a single batch of updates and exit."
    )
    args = parser.parse_args(argv)

    try:
        config = TelegramConfig.from_env()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    client = TelegramClient(config)

    if args.send is not None:
        message = client.send_message(args.send, chat_id=args.chat_id)
        print(f"Sent message id {message.get('message_id')}.")
        return

    bot = LabClawBot(client)
    if args.once:
        handled = bot.poll_once()
        print(f"Handled {handled} update(s).")
        return
    bot.run_polling()


if __name__ == "__main__":
    main()
