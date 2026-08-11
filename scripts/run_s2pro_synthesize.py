"""Thin wrapper: delegates to myecho.cli.s2pro_synthesize.main()."""

from myecho.cli.s2pro_synthesize import main

if __name__ == "__main__":
    raise SystemExit(main())

