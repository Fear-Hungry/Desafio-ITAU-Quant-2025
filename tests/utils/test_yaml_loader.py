from __future__ import annotations

import textwrap

from itau_quant.utils import yaml_loader


def test_load_yaml_parses_mapping(tmp_path):
    content = textwrap.dedent(
        """
        key: value
        nested:
          number: 10
        """
    )
    path = tmp_path / "config.yaml"
    path.write_text(content)

    loaded = yaml_loader.read_yaml(path)
    assert loaded["key"] == "value"
    assert loaded["nested"]["number"] == 10
