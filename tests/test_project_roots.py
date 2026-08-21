"""Where a project folder is allowed to live.

One root was enough while every connected project happened to sit in the same parent directory.
A project kept in the home directory instead failed to connect with

    project folder not found: /projects/<name>

for a folder that existed and was readable — because resolution keeps only the folder NAME (the
container reaches host directories through bind mounts, so a host path has to be re-rooted at its
mount point) and then re-rooted that name at ONE hardcoded root. The path the user typed was
discarded, and the error named a CONTAINER path they had never seen. Roots are plural now.
"""

from __future__ import annotations

import pytest

from synapse.config import settings
from synapse.core import registry


@pytest.fixture
def roots(monkeypatch, tmp_path):
    """A primary root plus two extras, configured the way the env var carries them."""
    primary, extra_a, extra_b = (tmp_path / n for n in ("primary", "a", "b"))
    for p in (primary, extra_a, extra_b):
        p.mkdir()
    monkeypatch.setattr(settings, "projects_root", str(primary))
    monkeypatch.setattr(settings, "extra_project_roots", f"{extra_a}, ,{extra_b}")
    return primary, extra_a, extra_b


def test_the_primary_root_is_searched_before_the_extras(roots):
    """Order is the tie-breaker when the same folder name exists under two roots, and the primary
    is also the one that owns the connected-projects overlay — so it has to come first."""
    primary, extra_a, extra_b = roots
    assert registry.project_roots() == [primary, extra_a, extra_b]


def test_a_blank_entry_is_not_a_root(roots):
    """An empty segment (a stray comma in the env var) must not become Path('.'), which would
    silently resolve every project against the process working directory."""
    assert all(str(r) != "." for r in registry.project_roots())


def test_a_project_under_an_extra_root_resolves_to_that_root(roots):
    _, extra_a, _ = roots
    (extra_a / "story-app").mkdir()
    assert registry.project_folder(r"C:\Users\dev\story-app") == extra_a / "story-app"


def test_a_project_that_exists_nowhere_falls_back_to_the_primary_root(roots):
    """The project LIST asks for a folder in order to report `exists: false` about it, so an
    unresolvable project needs a path back, not an exception."""
    primary, _, _ = roots
    assert registry.project_folder("/gone/ghost-app") == primary / "ghost-app"


def test_one_configured_root_still_behaves_exactly_as_before(monkeypatch, tmp_path):
    """The overwhelmingly common case: no extras configured, resolution unchanged."""
    monkeypatch.setattr(settings, "projects_root", str(tmp_path))
    monkeypatch.setattr(settings, "extra_project_roots", "")
    assert registry.project_roots() == [tmp_path]
    assert registry.project_folder("/anywhere/acme-api") == tmp_path / "acme-api"
