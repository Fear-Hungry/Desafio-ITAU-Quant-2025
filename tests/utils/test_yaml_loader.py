from __future__ import annotations

import textwrap

from itau_quant.utils import yaml_loader


def test_load_yaml_parses_mapping(tmp_path):
    EXPECTED_NUMBER = 10
    content = textwrap.dedent(
        f"""
        key: value
        nested:
          number: {EXPECTED_NUMBER}
        """
    )
    path = tmp_path / "config.yaml"
    path.write_text(content)

    loaded = yaml_loader.read_yaml(path)
    assert loaded["key"] == "value"
    assert loaded["nested"]["number"] == EXPECTED_NUMBER
