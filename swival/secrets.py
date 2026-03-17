"""Transparent secret encryption/decryption for LLM message boundaries.

Encrypts recognized credential tokens (API keys, PATs, etc.) before messages
leave the machine, and decrypts them when the LLM references them in responses.
The LLM provider sees format-preserving fakes; tool dispatch and local storage
see real values.
"""

import copy
import os

from fast_cipher.tokens import BUILTIN_PATTERNS, TokenEncryptor, scan as scan_tokens
from fast_cipher.tokens.alphabets import ALPHANUMERIC
from fast_cipher.tokens.types import SimpleTokenPattern

from .report import ConfigError

ENCRYPT_KEY_ENV = "SWIVAL_ENCRYPT_KEY"


class SecretShield:
    """Encrypt secrets outbound, decrypt inbound.  Provenance-checked."""

    def __init__(
        self,
        *,
        key: bytes | None = None,
        tweak: bytes | None = None,
        extra_patterns: list | None = None,
    ):
        if key is None:
            key = os.urandom(32)

        self._encryptor = TokenEncryptor(key)
        self._tweak = tweak
        self._registry: dict[str, str] = {}  # ciphertext -> plaintext
        self._destroyed = False
        self._patterns = list(BUILTIN_PATTERNS)

        if extra_patterns:
            for pat in extra_patterns:
                pat = dict(pat)
                pat.setdefault("body_alphabet", ALPHANUMERIC)
                pat.setdefault("min_body_length", len(pat.get("prefix", "")) + 8)
                p = SimpleTokenPattern(**pat)
                self._encryptor.register(p)
                self._patterns.append(p)

    @classmethod
    def from_config(
        cls,
        *,
        key_hex: str | None = None,
        tweak_str: str | None = None,
        extra_patterns: list | None = None,
    ) -> "SecretShield":
        """Construct from string config values (hex key, UTF-8 tweak)."""
        key = bytes.fromhex(key_hex) if key_hex else None
        tweak = tweak_str.encode("utf-8") if tweak_str else None
        return cls(key=key, tweak=tweak, extra_patterns=extra_patterns)

    # --- Outbound (before LLM) ---

    def encrypt_messages(self, messages: list[dict]) -> list[dict]:
        """Deep-copy messages and encrypt allowlisted fields.

        Populates the internal registry with new ciphertext->plaintext pairs.
        Returns the encrypted copy; the original list is not modified.
        """
        if self._destroyed:
            raise ConfigError("SecretShield has been destroyed")

        encrypted = copy.deepcopy(messages)
        for msg in encrypted:
            role = msg.get("role", "")
            if role in ("system", "user", "tool"):
                self._encrypt_content(msg)
            elif role == "assistant":
                self._encrypt_content(msg)
                self._encrypt_tool_calls(msg)
        return encrypted

    def _encrypt_content(self, msg: dict) -> None:
        """Encrypt the ``content`` field of a message dict in place."""
        content = msg.get("content")
        if content is None:
            return

        if isinstance(content, str):
            msg["content"] = self._encrypt_and_record(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        part["text"] = self._encrypt_and_record(text)

    def _encrypt_tool_calls(self, msg: dict) -> None:
        """Encrypt tool_call arguments in an assistant message dict in place."""
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return

        for tc in tool_calls:
            func = tc.get("function") if isinstance(tc, dict) else None
            if func is None:
                continue
            args_str = func.get("arguments")
            if isinstance(args_str, str):
                func["arguments"] = self._encrypt_and_record(args_str)

    def _encrypt_and_record(self, text: str) -> str:
        """Encrypt *text*, recording new ciphertext->plaintext pairs."""
        if not text:
            return text

        enc_kwargs = {}
        if self._tweak is not None:
            enc_kwargs["tweak"] = self._tweak

        encrypted = self._encryptor.encrypt(text, **enc_kwargs)

        if encrypted != text:
            self._diff_and_record(text, encrypted)

        return encrypted

    def _diff_and_record(self, original: str, encrypted: str) -> None:
        """Find tokens that changed and record ciphertext->plaintext pairs.

        Encrypts each recognized token individually to get stable
        plaintext->ciphertext mappings regardless of length changes
        (heuristic tokens may expand due to [ENCRYPTED:...] markers).
        """
        enc_kwargs = {}
        if self._tweak is not None:
            enc_kwargs["tweak"] = self._tweak

        spans = scan_tokens(original, self._patterns)

        if spans:
            # Encrypt each token individually to get its exact ciphertext,
            # avoiding positional-alignment issues with heuristic tokens
            # that change length.
            for span in spans:
                plain_token = original[span.start : span.end]
                enc_token = self._encryptor.encrypt(plain_token, **enc_kwargs)
                if enc_token != plain_token:
                    self._registry[enc_token] = plain_token
        else:
            # No tokens found by scan — fall back to decrypt-based diffing.
            self._record_via_roundtrip(original, encrypted, enc_kwargs)

    def _record_via_roundtrip(
        self, original: str, encrypted: str, enc_kwargs: dict
    ) -> None:
        """Record registry entries when scan finds no tokens.

        Decrypts the encrypted text and aligns changed regions to extract
        individual token mappings. This works for length-preserving (prefixed)
        tokens. For heuristic tokens that change length, we fall back to
        recording the whole encrypted text paired with the original so that
        reverse_known can at least do a full-string replacement.
        """
        if len(original) == len(encrypted):
            # Same length — find contiguous changed regions
            i = 0
            n = len(original)
            while i < n:
                if original[i] != encrypted[i]:
                    j = i
                    while j < n and original[j] != encrypted[j]:
                        j += 1
                    ct = encrypted[i:j]
                    pt = original[i:j]
                    self._registry[ct] = pt
                    i = j
                else:
                    i += 1
        else:
            # Length changed (heuristic tokens expanded).
            # Try to decrypt to verify round-trip, then record whole text.
            try:
                decrypted = self._encryptor.decrypt(encrypted, **enc_kwargs)
            except Exception:
                decrypted = None
            if decrypted == original:
                # Valid round-trip — record entire encrypted as mapping.
                # reverse_known will do full-string match.
                self._registry[encrypted] = original

    # --- Inbound (after LLM) ---

    def reverse_known(self, text: str) -> str:
        """Replace only ciphertext tokens that we actually emitted.

        Operates on raw strings (no JSON parsing). Model-invented
        token-shaped strings pass through untouched.
        """
        if self._destroyed or not self._registry:
            return text

        for ciphertext, plaintext in self._registry.items():
            text = text.replace(ciphertext, plaintext)

        return text

    # --- Lifecycle ---

    def destroy(self) -> None:
        """Zeroize key material and clear the registry."""
        if self._destroyed:
            return
        try:
            self._encryptor.destroy()
        except Exception:
            pass
        self._registry.clear()
        self._encryptor = None  # type: ignore[assignment]
        self._destroyed = True

    @property
    def destroyed(self) -> bool:
        return self._destroyed
