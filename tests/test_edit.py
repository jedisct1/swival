"""Tests for edit.py — the string replacement engine behind edit_file."""

import pytest

from swival.edit import replace


# =========================================================================
# Exact match
# =========================================================================


class TestExactMatch:
    """Basic exact-match replacements."""

    def test_simple_replacement(self):
        content = "hello world"
        result = replace(content, "hello", "goodbye")
        assert result == "goodbye world"

    def test_multiline_replacement(self):
        content = "aaa\nbbb\nccc\n"
        result = replace(content, "bbb\nccc", "BBB\nCCC")
        assert result == "aaa\nBBB\nCCC\n"

    def test_replacement_at_start(self):
        content = "first\nsecond\nthird\n"
        result = replace(content, "first", "FIRST")
        assert result == "FIRST\nsecond\nthird\n"

    def test_replacement_at_end(self):
        content = "first\nsecond\nthird"
        result = replace(content, "third", "THIRD")
        assert result == "first\nsecond\nTHIRD"

    def test_replacement_preserves_surrounding(self):
        content = "before\ntarget line\nafter\n"
        result = replace(content, "target line", "new line")
        assert result == "before\nnew line\nafter\n"


# =========================================================================
# Replace all
# =========================================================================


class TestReplaceAll:
    """replace_all=True replaces every occurrence."""

    def test_replaces_all_occurrences(self):
        content = "foo bar foo baz foo"
        result = replace(content, "foo", "qux", replace_all=True)
        assert result == "qux bar qux baz qux"

    def test_replaces_all_multiline(self):
        content = "x = 1\ny = 2\nx = 1\ny = 3\n"
        result = replace(content, "x = 1", "x = 0", replace_all=True)
        assert result == "x = 0\ny = 2\nx = 0\ny = 3\n"

    def test_replace_all_single_occurrence(self):
        content = "only once here"
        result = replace(content, "once", "twice", replace_all=True)
        assert result == "only twice here"


# =========================================================================
# Multiple matches error
# =========================================================================


class TestMultipleMatchesError:
    """Ambiguous matches raise ValueError when replace_all=False."""

    def test_duplicate_raises(self):
        content = "aaa\nbbb\naaa\n"
        with pytest.raises(ValueError, match="multiple matches"):
            replace(content, "aaa", "ccc")

    def test_adding_context_resolves_ambiguity(self):
        content = "aaa\nbbb\naaa\nccc\n"
        result = replace(content, "aaa\nccc", "AAA\nCCC")
        assert result == "aaa\nbbb\nAAA\nCCC\n"


# =========================================================================
# Not found
# =========================================================================


class TestNotFound:
    """Missing strings raise ValueError."""

    def test_string_not_in_content(self):
        content = "hello world"
        with pytest.raises(ValueError, match="not found"):
            replace(content, "xyz", "abc")

    def test_partial_match_not_found(self):
        content = "abc def ghi"
        with pytest.raises(ValueError, match="not found"):
            replace(content, "abc ghi", "replaced")


# =========================================================================
# No-op (old == new)
# =========================================================================


class TestNoOp:
    """old_string == new_string raises ValueError."""

    def test_identical_strings(self):
        content = "hello world"
        with pytest.raises(ValueError, match="no changes"):
            replace(content, "hello", "hello")

    def test_identical_multiline(self):
        content = "a\nb\nc\n"
        with pytest.raises(ValueError, match="no changes"):
            replace(content, "a\nb", "a\nb")


# =========================================================================
# Validation
# =========================================================================


class TestValidation:
    """Input validation."""

    def test_empty_old_string_raises(self):
        content = "hello"
        with pytest.raises(ValueError, match="empty"):
            replace(content, "", "something")


# =========================================================================
# Fuzzy matching
# =========================================================================


class TestFuzzyMatching:
    """Whitespace tolerance and Unicode normalization."""

    def test_leading_whitespace_tolerance(self):
        content = "  indented line\nafter\n"
        result = replace(content, "indented line", "new line")
        assert "new line" in result

    def test_trailing_whitespace_tolerance(self):
        content = "line with spaces   \nafter\n"
        result = replace(content, "line with spaces", "clean line")
        assert "clean line" in result

    def test_unicode_smart_quotes_normalized(self):
        content = "print(\u201chello\u201d)\n"
        result = replace(content, 'print("hello")', 'print("world")')
        assert 'print("world")' in result

    def test_unicode_em_dash_normalized(self):
        content = "value \u2014 10\n"
        result = replace(content, "value - 10", "value - 20")
        assert "value - 20" in result

    def test_unicode_ellipsis_normalized(self):
        content = "loading\u2026\n"
        result = replace(content, "loading...", "done")
        assert "done" in result

    def test_non_breaking_space_normalized(self):
        content = "hello\u00a0world\n"
        result = replace(content, "hello world", "hi there")
        assert "hi there" in result


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Unusual content shapes."""

    def test_single_line_no_newline(self):
        content = "only line"
        result = replace(content, "only", "first")
        assert result == "first line"

    def test_replacement_with_empty_new_string(self):
        content = "keep\nremove\nkeep\n"
        result = replace(content, "remove\n", "")
        assert result == "keep\nkeep\n"

    def test_multiline_block_replacement(self):
        content = "a\nb\nc\nd\ne\n"
        result = replace(content, "b\nc\nd", "B\nC\nD")
        assert result == "a\nB\nC\nD\ne\n"
