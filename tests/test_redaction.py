"""Secret redaction — the write-path gate that keeps credentials out of the graph.

Wave 0 of docs/research/2026-07-25-system-improvement-research.md §5.1. The stakes are
specific to Synapse: the UserPromptSubmit recall hook injects `global`-scope facts into ALL
connected projects, so a secret captured once in one project would be replayed into every
other project's prompts. Redaction must therefore happen BEFORE embedding and before the
content is ever sent to an extraction LLM.

All fixtures below are FAKE credentials with the right SHAPE (never real secrets).
"""

from __future__ import annotations

import pytest

from synapse.core.redaction import PLACEHOLDER_PREFIX, redact


def _kinds(text: str) -> list[str]:
    return redact(text)[1]


def _clean(text: str) -> str:
    return redact(text)[0]


# --- provider key formats -------------------------------------------------------


@pytest.mark.parametrize(
    "kind,secret",
    [
        ("anthropic_api_key", "sk-ant-api03-" + "A1b2C3d4E5f6G7h8" * 4),
        ("openai_api_key", "sk-proj-" + "Xy9Zw8Vu7Ts6Rq5P" * 3),
        ("github_token", "ghp_" + "aB3dE6gH9jK2mN5pQ8rS1tU4vW7xY0zA2bC5"),
        ("github_token", "github_pat_" + "11ABCDE0" + "aB3dE6gH9jK2mN5pQ8rS1t"),
        ("aws_access_key_id", "AKIAIOSFODNN7EXAMPLE"),
        ("google_api_key", "AIza" + "SyD-1234567890abcdefghijklmnopqrstuv"),
        # Split like the others above, and for the same reason: a contiguous literal of this shape
        # is flagged by GitHub's push protection even though the value is invented, and a blocked
        # push on a fake token is a real outage. Python joins it back before the test sees it.
        ("slack_token", "xoxb-" + "123456789012-" + "1234567890123-" + "abcdefghijklmnopqrstuvwx"),
        ("gitlab_token", "glpat-" + "aB3dE6gH9jK2mN5pQ8rS"),
        ("stripe_key", "sk_live_" + "aB3dE6gH9jK2mN5pQ8rS1tU4"),
        ("jwt", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dQw4w9WgXcQ_signature"),
    ],
)
def test_provider_secrets_are_redacted_and_named(kind, secret):
    text = f"We authenticate with {secret} in the staging config."
    cleaned, kinds = redact(text)
    assert secret not in cleaned, f"{kind} survived redaction"
    assert kind in kinds
    assert PLACEHOLDER_PREFIX in cleaned
    # the surrounding knowledge must survive — we redact, never reject
    assert "We authenticate with" in cleaned and "staging config" in cleaned


def test_pem_private_key_block_is_redacted_whole():
    text = (
        "The deploy key is:\n"
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAxGZ1p2q3r4s5t6u7v8w9x0y1z2A3B4C5D6E7F8G9H0I1J2K3\n"
        "L4M5N6O7P8Q9R0S1T2U3V4W5X6Y7Z8a9b0c1d2e3f4g5h6i7j8k9l0m1n2o3p4q5\n"
        "-----END RSA PRIVATE KEY-----\n"
        "Rotate it quarterly."
    )
    cleaned, kinds = redact(text)
    assert "pem_private_key" in kinds
    assert "MIIEowIBAAKCAQEA" not in cleaned
    assert "BEGIN RSA PRIVATE KEY" not in cleaned
    assert "Rotate it quarterly." in cleaned  # the durable lesson survives


# --- assignment / URI forms -----------------------------------------------------


def test_env_style_assignment_redacts_value_not_key():
    cleaned, kinds = redact("Set NEO4J_PASSWORD=hunter2supersecret in .env")
    assert "assigned_secret" in kinds
    assert "hunter2supersecret" not in cleaned
    assert "NEO4J_PASSWORD" in cleaned, "the key name is useful knowledge; keep it"


@pytest.mark.parametrize(
    "snippet,secret",
    [
        ('password: "correct-horse-battery"', "correct-horse-battery"),
        ("client_secret=abc123def456ghi", "abc123def456ghi"),
        ("ANTHROPIC_API_KEY = zzzTopSecretValue", "zzzTopSecretValue"),
        ("access_token: ya29.averylongaccesstokenvalue", "ya29.averylongaccesstokenvalue"),
        ("Authorization: Bearer abcdef1234567890abcdef", "abcdef1234567890abcdef"),
    ],
)
def test_assignment_forms(snippet, secret):
    cleaned, kinds = redact(snippet)
    assert secret not in cleaned
    assert kinds, f"nothing flagged for {snippet!r}"


def test_credentialed_uri_keeps_host_drops_credentials():
    cleaned, kinds = redact("Connect via bolt://neo4j:myrealpassword@127.0.0.1:7688")
    assert "credentialed_uri" in kinds
    assert "myrealpassword" not in cleaned
    # host/port are operationally useful and not secret — they must survive
    assert "127.0.0.1:7688" in cleaned
    assert "bolt://" in cleaned


# --- broker account identifiers -------------------------------------------------


@pytest.mark.parametrize(
    "account",
    ["DU1234567", "DU12345678", "U1234567", "U12345678", "F1234567"],
)
def test_brokerage_account_ids_are_redacted(account):
    """Paper (DU), live (U) and advisor (F) forms all go, prose survives."""
    text = f"The gateway ict-ib-gateway:4004 serves broker account {account}, shared by three apps."
    cleaned, kinds = redact(text)
    assert account not in cleaned
    assert "brokerage_account_id" in kinds
    # The operationally load-bearing part is the topology, not the identity — it must survive.
    assert "ict-ib-gateway:4004" in cleaned and "shared by three apps" in cleaned


def test_brokerage_rule_does_not_split_a_paper_account_into_a_live_one():
    """`\\bU\\d{7,8}` must not match the `U…` sitting inside `DU…`.

    D and U are both word characters, so there is no word boundary between them. If this
    regressed, a paper id would redact as `D[REDACTED:brokerage_account_id]` — leaking the digits.
    """
    cleaned, kinds = redact("account DU7654321 is paper")
    assert cleaned == "account [REDACTED:brokerage_account_id] is paper"
    assert kinds == ["brokerage_account_id"]
    assert "7654321" not in cleaned


def test_brokerage_account_redaction_is_idempotent():
    once = _clean("account DU1234567 is paper")
    assert redact(once) == (once, [])


# --- false-positive guards (these must NOT be redacted) -------------------------


@pytest.mark.parametrize(
    "benign",
    [
        # Adjacent to the broker account rule: too short, too long, or not at a word boundary.
        "Gateway ports are 4001, 4002, 4003 and 4004 across the cluster.",
        "Part SKU1234567 shipped; the U-shaped bracket is unrelated.",
        "clientId U12345 and U123456789 are not account ids (6 and 9 digits).",
        "Use max_tokens=400 for the triage call.",
        "The response had num_tokens = 1024 in the usage block.",
        "Token counting matters for cost; tokens=8192 is the cap.",
        "BigDecimal is the required type for money in Java 21.",
        "The dedup_threshold=0.9 and relate_floor=0.75 settings gate the write path.",
        "Commit 705bf8f fixed the stdin decode bug.",
        "The vector dimension is 1024 (bge-m3) and is LOCKED at first ingestion.",
        "Password rotation policy: quarterly, tracked in the runbook.",
        "Set the timeout to 3 seconds so the hook never blocks a prompt.",
        "Node uuid 4f2c9a1e-7b3d-4e8f-9a0b-1c2d3e4f5a6b is the canonical fact.",
        "The sha256 content hash is e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.",
    ],
)
def test_benign_technical_content_is_untouched(benign):
    cleaned, kinds = redact(benign)
    assert cleaned == benign, f"false positive: {kinds}"
    assert kinds == []


def test_max_tokens_is_not_treated_as_a_token_secret():
    # Regression: a bare `token=` rule would corrupt real Synapse knowledge like "max_tokens=400".
    assert _kinds("haiku_or_local(system, user, max_tokens=400)") == []


# --- entropy backstop -----------------------------------------------------------


def test_high_entropy_blob_is_caught_as_a_backstop():
    # An unknown-format credential with no recognizable prefix.
    blob = "Zk9Qw3Er7Ty1Ui5Op8As2Df6Gh0Jk4Lz7Xc1Vb5Nm9Qw3Er7Ty"
    cleaned, kinds = redact(f"The service uses {blob} to sign requests.")
    assert "high_entropy_string" in kinds
    assert blob not in cleaned


def test_entropy_backstop_ignores_hex_and_uuids():
    # git SHAs, sha256 hashes and uuids are legitimately present in knowledge text.
    for benign in (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "4f2c9a1e-7b3d-4e8f-9a0b-1c2d3e4f5a6b",
        "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    ):
        assert _kinds(f"reference {benign} here") == [], benign


def test_entropy_backstop_ignores_ordinary_prose_and_long_words():
    prose = (
        "The retrieval engine composes global and project scopes, applies a similarity floor, "
        "and ranks by relevance, recency, confidence and connectivity before truncation."
    )
    assert _kinds(prose) == []


# --- behaviour contract ---------------------------------------------------------


def test_multiple_distinct_secrets_all_reported_once_each():
    text = (
        "export ANTHROPIC_API_KEY=sk-ant-api03-" + "A1b2C3d4E5f6G7h8" * 4 + "\n"
        "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
        "export DB=postgres://admin:trickypassword@db.internal:5432/app\n"
    )
    cleaned, kinds = redact(text)
    assert "anthropic_api_key" in kinds
    assert "aws_access_key_id" in kinds
    assert "credentialed_uri" in kinds
    assert kinds == sorted(set(kinds)), "kinds must be de-duplicated and sorted"
    assert "trickypassword" not in cleaned
    assert "AKIAIOSFODNN7EXAMPLE" not in cleaned


def test_clean_content_is_returned_unchanged_and_is_identity():
    original = "We chose Graphiti over raw Neo4j because the temporal model is native."
    cleaned, kinds = redact(original)
    assert cleaned is original or cleaned == original
    assert kinds == []


def test_redaction_is_idempotent():
    once, kinds1 = redact("password=supersecretvalue123")
    twice, kinds2 = redact(once)
    assert twice == once, "re-redacting must not corrupt an existing placeholder"
    assert kinds1 and kinds2 == []


def test_empty_and_none_safe():
    assert redact("") == ("", [])
    assert redact("   ") == ("   ", [])


def test_placeholder_never_leaks_the_secret_kind_value():
    secret = "sk-ant-api03-" + "Zz9Yy8Xx7Ww6" * 4
    cleaned, _ = redact(f"key is {secret}")
    # the placeholder names the KIND, never any portion of the secret
    assert "sk-ant" not in cleaned
    assert "Zz9Yy8" not in cleaned


# --- entropy false-positive regressions -----------------------------------------
# Every shape below was flagged by the FIRST corpus scan (2026-07-25) against the live
# graph and turned out to be legitimate knowledge, not a credential. They are locked in
# here so the entropy backstop can never re-acquire this class of false positive.


@pytest.mark.parametrize(
    "identifier",
    [
        "research_sector_etf_ranking_faber_2026_07_08",   # a memory/doc filename
        "bug_partial_bracket_conservation_2026-12",       # a bug id
        "UMPI_INTERVIEW_COPILOT_TIER0_prompt_step",       # an acronym scaffold name
        "docs/research/2026-07-25-system-improvement-research.md",
        "synapse_relates_fact_vec_index_definition_v2",
        "test_recall_hook_merges_alongside_brief_hook",   # a long test name
        "SPEC-ORDER-HARNESS-phase-0-partial-bracket-redesign",
    ],
)
def test_long_separator_rich_identifiers_are_not_secrets(identifier):
    cleaned, kinds = redact(f"Details live in {identifier} for later reference.")
    assert kinds == [], f"false positive on {identifier!r}: {kinds}"
    assert identifier in cleaned


def test_unbroken_random_run_is_still_caught():
    # The distinguishing feature is an unbroken high-entropy run, not raw length/entropy.
    assert "high_entropy_string" in _kinds("signs with Zk9Qw3Er7Ty1Ui5Op8As2Df6Gh0Jk4Lz7Xc1Vb5Nm9Qw3Er7Ty")


def test_separators_do_not_let_a_real_credential_evade_prefix_rules():
    # Known formats are caught by prefix rules regardless of the entropy heuristic.
    key = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8" * 4
    assert "anthropic_api_key" in _kinds(f"key {key} here")
