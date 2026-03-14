import sys

REQ_MAJOR = 3
REQ_MINOR = 13


def main() -> int:
    v = sys.version_info
    if (v.major, v.minor) < (REQ_MAJOR, REQ_MINOR):
        sys.stderr.write(
            f"ERROR: Python {REQ_MAJOR}.{REQ_MINOR}+ required, found {v.major}.{v.minor}.{v.micro}.\n"
        )
        sys.stderr.write("Run: scripts/bootstrap_venv.sh\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
