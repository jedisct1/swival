"""Tests for find_skill_prefix()."""

from swival.skills import find_skill_prefix


class TestFindSkillPrefix:
    def test_at_end(self):
        assert find_skill_prefix("$front") == "front"

    def test_mid_sentence(self):
        assert find_skill_prefix("use $fr") == "fr"

    def test_no_boundary_alphanumeric_before_dollar(self):
        assert find_skill_prefix("foo$bar") is None

    def test_bare_dollar(self):
        assert find_skill_prefix("$") == ""

    def test_double_dollar(self):
        # Second $ is preceded by $ (non-alphanumeric), so it IS at a boundary.
        # Runtime extract_skill_mentions also matches from the second $.
        assert find_skill_prefix("$$name") == "name"

    def test_space_before_dollar(self):
        assert find_skill_prefix("hello $sec") == "sec"

    def test_hyphen_in_name(self):
        assert find_skill_prefix("$frontend-design") == "frontend-design"

    def test_uppercase_rejected(self):
        assert find_skill_prefix("$FOO") is None

    def test_starts_with_hyphen_rejected(self):
        assert find_skill_prefix("$-foo") is None

    def test_no_dollar(self):
        assert find_skill_prefix("hello world") is None

    def test_empty_string(self):
        assert find_skill_prefix("") is None

    def test_digit_in_name(self):
        assert find_skill_prefix("$skill2") == "skill2"

    def test_trailing_space_ends_mention(self):
        # Space is not in [a-z0-9-], so not a valid partial name.
        assert find_skill_prefix("$foo ") is None

    def test_dollar_after_newline(self):
        assert find_skill_prefix("hello\n$sk") == "sk"

    def test_multiple_mentions_returns_last(self):
        assert find_skill_prefix("$foo use $ba") == "ba"

    def test_dollar_after_punctuation(self):
        assert find_skill_prefix("($skill") == "skill"

    def test_only_digits(self):
        assert find_skill_prefix("$42") == "42"
