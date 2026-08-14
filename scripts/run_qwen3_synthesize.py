"""Thin wrapper: delegates to myecho.cli.qwen3_synthesize.main()."""

from myecho.cli.qwen3_synthesize import main

if __name__ == "__main__":
    raise SystemExit(main())
