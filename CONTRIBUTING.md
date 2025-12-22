# Contributing

## Development setup
```bash
poetry install --sync
```

## Common commands
- Run fast tests: `make test-fast`
- Run full tests: `make test`
- Lint: `make lint`
- Format: `make format`
- Validate configs: `make validate-configs`

## Language policy
This repository is **English-only** for new and updated content.
See `docs/standards/LANGUAGE_POLICY.md`.

## PR guidelines
- Prefer small, focused commits.
- Keep runners as thin wrappers; reusable logic should live under `src/arara_quant/`.
- Avoid committing generated artefacts (reports, figures, outputs).
- When adding a new feature, include a minimal test when feasible.
