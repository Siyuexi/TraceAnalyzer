"""Pin the file-view command parsing in p2a.core.

Both the raw-text path (`_parse_bash_read_commands`) and the structured tool-call
path (`parse_read_actions_from_tool_calls` -> `_parse_bash_read_commands_from_str`)
share one extractor (`_extract_reads_from_command`); these tests lock the extracted
ranges, the dedup, and the cross-path consistency.
"""

from p2a.core import (
    _parse_bash_read_commands,
    _parse_bash_read_commands_from_str,
    parse_read_actions_from_tool_calls,
)


def test_extracts_cat_head_tail_sed_ranges():
    reads = _parse_bash_read_commands_from_str(
        "cat /testbed/foo.py && head -20 /testbed/bar.py && "
        "tail -5 /testbed/baz.py && sed -n '10,20p' /testbed/qux.py"
    )
    by_file = {r["file_path"]: (r["start_line"], r["end_line"]) for r in reads}
    assert by_file["foo.py"] == (1, 999999)   # cat → whole file
    assert by_file["bar.py"] == (1, 20)       # head -20
    assert by_file["baz.py"] == (1, 999999)   # tail length unknown → full range
    assert by_file["qux.py"] == (10, 20)      # sed -n 'start,endp'


def test_grep_n_yields_single_read_not_duplicate():
    # `grep -n` matches both the grep-n and the generic grep pattern; the shared
    # extractor must dedup it to one read.
    reads = _parse_bash_read_commands_from_str("grep -n needle /testbed/qux.py")
    assert reads == [{"file_path": "qux.py", "start_line": 1, "end_line": 999999}]


def test_text_and_str_paths_agree():
    cmd = "cat /testbed/a.py && grep -n x /testbed/b.py && sed -n '3,7p' /testbed/c.py"
    assert _parse_bash_read_commands(cmd) == _parse_bash_read_commands_from_str(cmd)


def test_structured_execute_bash_tool_call():
    tool_calls = [
        {"function": {"name": "execute_bash", "arguments": {"command": "cat /testbed/foo.py"}}},
        {"function": {"name": "str_replace_editor",
                      "arguments": {"command": "view", "path": "/testbed/bar.py", "view_range": [5, 9]}}},
    ]
    reads = parse_read_actions_from_tool_calls(tool_calls)
    assert {"file_path": "foo.py", "start_line": 1, "end_line": 999999} in reads
    assert {"file_path": "bar.py", "start_line": 5, "end_line": 9} in reads
