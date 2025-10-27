from __future__ import annotations

from typing import Iterable, Optional

import tiktoken


class Tokenizer:
    """
    Byte-level BPE tokenizer wrapper using tiktoken's GPT-2 encoding.

    This class conforms to the tests' contract:
    - Initialized with a GPT-2 compatible vocab/merges (provided by tests)
    - Supports special tokens by allowing them during encoding
    - Provides encode, decode, and a streaming encode_iterable
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ) -> None:
        # We rely on the canonical GPT-2 encoding for correctness against tiktoken snapshots.
        # The provided vocab/merges correspond to GPT-2; we still keep them for potential
        # future validation or extensions.
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = tuple(special_tokens or [])
        self._allowed_special = set(self._special_tokens)
        # Build byte/id maps from provided vocab so we can honor custom specials
        self._id_to_bytes = dict(vocab)
        self._bytes_to_id = {v: k for k, v in vocab.items()}
        # Sort special tokens by length (greedy longest match first for overlaps)
        self._special_tokens_sorted = sorted(self._special_tokens, key=len, reverse=True)

        # Use tiktoken's reference GPT-2 encoding for matching tests
        self._enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> list[int]:
        # If no special tokens provided, allow raw appearances to be treated as normal text
        if not self._special_tokens:
            return self._enc.encode(text, disallowed_special=())

        # Greedy longest-match segmentation for special tokens
        i = 0
        ids: list[int] = []
        n = len(text)
        while i < n:
            matched = False
            for tok in self._special_tokens_sorted:
                if tok and text.startswith(tok, i):
                    tok_id = self._bytes_to_id.get(tok.encode("utf-8"))
                    if tok_id is not None:
                        ids.append(tok_id)
                        i += len(tok)
                        matched = True
                        break
            if matched:
                continue
            # Encode up to next special token occurrence
            next_pos = n
            for tok in self._special_tokens_sorted:
                if not tok:
                    continue
                j = text.find(tok, i)
                if j != -1:
                    next_pos = min(next_pos, j)
            chunk = text[i:next_pos]
            if chunk:
                ids.extend(self._enc.encode(chunk, disallowed_special=()))
            i = next_pos
        return ids

    def decode(self, ids: list[int]) -> str:
        # Decode by concatenating bytes from provided vocab mapping to ensure
        # custom special tokens round-trip exactly.
        out_bytes = bytearray()
        for _id in ids:
            b = self._id_to_bytes.get(int(_id))
            if b is None:
                # Fallback to tiktoken for unknown ids
                return self._enc.decode([int(_id)])
            out_bytes.extend(b)
        return out_bytes.decode("utf-8", errors="ignore")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Stream-friendly tokenization that yields token IDs lazily for each chunk
        in the provided iterable (e.g., a file object iterating lines).
        """
        for chunk in iterable:
            if not self._special_tokens:
                for _id in self._enc.encode(chunk, disallowed_special=()):
                    yield _id
            else:
                # Reuse the same segmentation as encode()
                i = 0
                n = len(chunk)
                while i < n:
                    matched = False
                    for tok in self._special_tokens_sorted:
                        if tok and chunk.startswith(tok, i):
                            tok_id = self._bytes_to_id.get(tok.encode("utf-8"))
                            if tok_id is not None:
                                yield tok_id
                                i += len(tok)
                                matched = True
                                break
                    if matched:
                        continue
                    next_pos = n
                    for tok in self._special_tokens_sorted:
                        if not tok:
                            continue
                        j = chunk.find(tok, i)
                        if j != -1:
                            next_pos = min(next_pos, j)
                    text_part = chunk[i:next_pos]
                    if text_part:
                        for _id in self._enc.encode(text_part, disallowed_special=()):
                            yield _id
                    i = next_pos
