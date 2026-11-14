# llminfo — OpenRouter price board

> Built with Python · Shell · CSS (Python, Flask).

A dark-themed price and feature board for selected OpenRouter models: grouped by family, showing per-vendor endpoints, prices chosen with no-train-aware preference, and price-history sparklines.

## Installation

```bash
git clone https://github.com/i-william-hr/llminfo.git
cd llminfo
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Usage

```bash
python3 llminfo.py
```

The llminfo.py entry point runs standalone.

## Tests

```bash
python3 -m unittest discover
```

## License

MIT License — see [LICENSE](LICENSE).
