"""Thin wrapper: delegates to myecho.cli.voxtral_synthesize.main()."""
from myecho.cli.voxtral_synthesize import main
if __name__ == "__main__":
    raise SystemExit(main())
