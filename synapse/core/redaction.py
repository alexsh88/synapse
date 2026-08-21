"""Secret redaction — the write-path credential gate (research 2026-07-25 §5.1, Wave 0).

`WritePipeline.remember()` is the single chokepoint every write reaches (MCP, API, session
capture, seed, replay), and it accepts free-form text from any agent in any connected project.
Nothing scanned it before this module.

**Why this is load-bearing for Synapse specifically.** The `UserPromptSubmit` recall hook
auto-injects `global`-scope facts into *every* connected project. A credential captured once —
from a pasted `.env`, a debug session, a transcript read by session capture — would be stored
in Neo4j, embedded into the vector index, and then replayed into the prompts of all eleven
projects. That turns a storage mistake into a distribution channel, so redaction runs BEFORE
the content is embedded and before it is sent to any extraction/triage LLM.

**Policy: redact-and-flag, never reject.** A secret in the text does not make the surrounding
knowledge worthless ("rotate the deploy key quarterly" is a real lesson). We replace only the
credential with ``[REDACTED:<kind>]``, keep the prose, and report which kinds were found so the
caller can log/surface it. The placeholder names the *kind* only — never any part of the value.

**Design constraints:**

- Pure functions, stdlib only → unit-testable with no services (CLAUDE.md §4).
- **Idempotent.** Re-redacting already-redacted text is a no-op and reports nothing, so replay,
  capture-then-remember, and the corpus backfill can all run repeatedly. Every value pattern
  refuses to match an existing placeholder.
- **False positives corrupt knowledge**, so the rules are deliberately conservative. Notably we
  do NOT treat a bare ``token=`` as a secret: real Synapse knowledge contains ``max_tokens=400``.
  Only qualified forms (``access_token``, ``auth_token``, …) count. The entropy backstop
  (§ _entropy_findings) skips hex and UUIDs because git SHAs and content hashes legitimately
  appear in stored facts.
"""

from __future__ import annotations

import math
import re
from collections import Counter

PLACEHOLDER_PREFIX = "[REDACTED:"

# Refuses to re-match an existing placeholder — this is what makes redaction idempotent.
_NOT_PLACEHOLDER = r"(?!\[REDACTED:)"

# Key names that mark the following value as a credential. Deliberately EXCLUDES bare
# "token"/"tokens": `max_tokens=400` and `tokens=8192` are legitimate stored knowledge.
_SECRET_KEY = (
    r"(?:password|passwd|pwd|secret|api[_-]?key|apikey|access[_-]?key|secret[_-]?key"
    r"|access[_-]?token|auth[_-]?token|api[_-]?token|refresh[_-]?token|bearer[_-]?token"
    r"|client[_-]?secret|private[_-]?key|credentials?)"
)

# A credential value: no whitespace, and stops at quote/comma/semicolon/brace so we never eat
# trailing prose. Minimum 6 chars keeps `password=x` placeholders out of the results.
_VALUE = r"[^\s\"',;}\)]{6,}"

# (kind, compiled pattern, replacement). Order matters: specific provider formats first, so the
# generic assignment/entropy rules never see an already-replaced secret.
_RULES: list[tuple[str, re.Pattern[str], str]] = [
    # --- key material blocks ---
    (
        "pem_private_key",
        re.compile(
            r"-----BEGIN[A-Z ]*PRIVATE KEY-----.*?-----END[A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        f"{PLACEHOLDER_PREFIX}pem_private_key]",
    ),
    # --- provider-specific key formats (distinctive prefixes → near-zero false positives) ---
    ("anthropic_api_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}"), f"{PLACEHOLDER_PREFIX}anthropic_api_key]"),
    ("openai_api_key", re.compile(r"sk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{16,}"), f"{PLACEHOLDER_PREFIX}openai_api_key]"),
    ("github_token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), f"{PLACEHOLDER_PREFIX}github_token]"),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), f"{PLACEHOLDER_PREFIX}github_token]"),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), f"{PLACEHOLDER_PREFIX}aws_access_key_id]"),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), f"{PLACEHOLDER_PREFIX}google_api_key]"),
    ("slack_token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}"), f"{PLACEHOLDER_PREFIX}slack_token]"),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}"), f"{PLACEHOLDER_PREFIX}gitlab_token]"),
    ("stripe_key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}"), f"{PLACEHOLDER_PREFIX}stripe_key]"),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
        f"{PLACEHOLDER_PREFIX}jwt]",
    ),
    # --- brokerage account identifiers ----------------------------------------------------------
    # Not a credential — you cannot authenticate with it — but it identifies a real account, and a
    # cluster of related projects discussing a shared gateway will repeat it constantly, so it
    # spreads. This rule was added AFTER a corpus scan (2026-07-26) found one paper account id in
    # 37 places across facts, entity names, entity summaries and episodes, in `global` scope —
    # i.e. auto-injected into every project's prompts.
    #
    # The shape covered is the common one across retail brokers: a one- or two-letter class prefix
    # followed by 7-8 digits, where the prefix distinguishes paper from live from advisor accounts.
    # MEASURED over 7,922 corpus text fields before adding: the paper form matched 37 times, all of
    # them the one real account and nothing else; the live and advisor forms matched ZERO times. So
    # those cost nothing today while covering the more sensitive case. `\bU` cannot match inside
    # `DU…` (both are word chars, so there is no boundary), hence one combined rule.
    (
        "brokerage_account_id",
        re.compile(r"\b(?:DU|U|F)\d{7,8}\b"),
        f"{PLACEHOLDER_PREFIX}brokerage_account_id]",
    ),
    # --- credentials embedded in a connection URI: keep scheme + host (operationally useful) ---
    (
        "credentialed_uri",
        re.compile(rf"([a-zA-Z][a-zA-Z0-9+.\-]*://){_NOT_PLACEHOLDER}[^\s:/@]{{1,64}}:[^\s/@]{{1,128}}@"),
        rf"\g<1>{PLACEHOLDER_PREFIX}credentialed_uri]@",
    ),
    # --- Authorization: Bearer <value> ---
    (
        "bearer_token",
        re.compile(rf"\b(bearer\s+){_NOT_PLACEHOLDER}[A-Za-z0-9_\-.=]{{16,}}", re.IGNORECASE),
        rf"\g<1>{PLACEHOLDER_PREFIX}bearer_token]",
    ),
    # --- KEY=value / KEY: value. Keeps the key name (knowing WHICH secret exists is useful). ---
    (
        "assigned_secret",
        re.compile(
            rf"([A-Za-z0-9_.\-]{{0,32}}{_SECRET_KEY}[A-Za-z0-9_.\-]{{0,32}}\s*[:=]\s*[\"']?)"
            rf"{_NOT_PLACEHOLDER}{_VALUE}",
            re.IGNORECASE,
        ),
        rf"\g<1>{PLACEHOLDER_PREFIX}assigned_secret]",
    ),
]

# --- entropy backstop tuning ---------------------------------------------------------------
# Catches unknown-format credentials with no recognizable prefix. Conservative on purpose:
# a 40+ char unbroken high-entropy token in prose is almost never legitimate knowledge, but
# hex digests and UUIDs are — so those are excluded outright rather than by entropy.
_ENTROPY_MIN_LEN = 40
_ENTROPY_MIN_BITS = 4.2
# Minimum UNBROKEN alphanumeric run. This is what separates a credential from a long
# identifier, and it was calibrated against real false positives found by the first corpus
# scan (2026-07-25): `research_sector_etf_2026_07_08`, `bug_<slug>_2026-12`, and an
# acronym_scaffold_step name all cleared 40 chars and 4.2 bits, because `_`/`-`-separated
# identifiers accumulate entropy across many short, meaningful segments. A real credential is
# a long *unbroken* random run; a filename or slug is not.
_ENTROPY_MIN_RUN = 25
_CANDIDATE = re.compile(rf"[A-Za-z0-9+/=_\-]{{{_ENTROPY_MIN_LEN},}}")
_ALNUM_RUN = re.compile(r"[A-Za-z0-9]+")
_HEX_ONLY = re.compile(r"^[0-9a-fA-F]+$")
_UUID_LIKE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _longest_alnum_run(token: str) -> int:
    return max((len(m.group(0)) for m in _ALNUM_RUN.finditer(token)), default=0)


def _shannon_bits(value: str) -> float:
    """Shannon entropy in bits per character."""
    if not value:
        return 0.0
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in Counter(value).values())


def _is_probably_secret(token: str) -> bool:
    if "REDACTED" in token:
        return False
    if _HEX_ONLY.match(token) or _UUID_LIKE.match(token):
        return False  # git SHAs / sha256 content hashes / uuids appear legitimately in facts
    if _longest_alnum_run(token) < _ENTROPY_MIN_RUN:
        return False  # separator-rich identifier (filename, slug, bug id), not a credential
    # Require a mix of cases or digits — an all-lowercase run is more likely prose/an identifier.
    has_upper = any(c.isupper() for c in token)
    has_lower = any(c.islower() for c in token)
    has_digit = any(c.isdigit() for c in token)
    if not ((has_upper and has_lower) or (has_digit and (has_upper or has_lower))):
        return False
    return _shannon_bits(token) >= _ENTROPY_MIN_BITS


def redact(text: str) -> tuple[str, list[str]]:
    """Strip credentials from *text*.

    Returns ``(redacted_text, kinds)`` where ``kinds`` is the sorted, de-duplicated list of
    credential kinds found (empty when the text is clean). The text is returned unchanged when
    nothing matched, and re-running on already-redacted text yields ``(text, [])``.

    Never raises, never logs the value, and never drops surrounding prose.
    """
    if not text or not text.strip():
        return text, []

    found: set[str] = set()
    out = text

    for kind, pattern, replacement in _RULES:
        out, n = pattern.subn(replacement, out)
        if n:
            found.add(kind)

    # Entropy backstop, last: the rules above have already removed known formats, so anything
    # still long-and-random here is an unrecognized credential shape.
    for token in {m.group(0) for m in _CANDIDATE.finditer(out)}:
        if _is_probably_secret(token):
            out = out.replace(token, f"{PLACEHOLDER_PREFIX}high_entropy_string]")
            found.add("high_entropy_string")

    return out, sorted(found)


def has_secrets(text: str) -> bool:
    """True when *text* contains anything :func:`redact` would strip. Read-only probe."""
    return bool(redact(text)[1])
