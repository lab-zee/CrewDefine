"""YAML dump helpers that match LabZ's on-disk style.

LabZ's hand-written agent YAMLs use:
- block-style lists (`- item` under the key)
- `|` literal style for multiline `system_prompt`
- preserved field order (id, name, role, tools, can_delegate_to, ..., system_prompt)
- no `---` document marker

We do the same so CrewDefine-generated files diff cleanly against existing ones.
"""

from __future__ import annotations

from typing import Any

import yaml


class _LabZDumper(yaml.SafeDumper):
    """SafeDumper with stable settings and a representer for multiline strings."""


def _str_representer(dumper: _LabZDumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LabZDumper.add_representer(str, _str_representer)


def dump_agent_yaml(data: dict[str, Any]) -> str:
    """Serialize a single agent's dict in LabZ's canonical style."""
    return yaml.dump(
        data,
        Dumper=_LabZDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )
