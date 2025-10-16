"""Enable ``python -m ai_lens_helper`` to forward to the CLI."""

from .cli import app


def main() -> None:  # pragma: no cover - tiny wrapper
    app()


if __name__ == "__main__":  # pragma: no cover - module entry point
    main()
