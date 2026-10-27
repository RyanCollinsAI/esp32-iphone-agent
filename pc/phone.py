"""phone: drive the iPhone through phoned (pc/phoned.py). Coordinates are 0-1000 across / down the screen.

    phone status                 phone screen              # -> PNG paths (screen-grid.png has a 0-1000 grid)
    phone calibrate              phone tap 500 620         phone hold 500 620   # long press
    phone swipe 500 800 500 300  phone scroll 500 500 -5   # lines: negative scrolls down
    phone type "hello"           phone key enter           # key combos: cmd+space, shift+tab, ...
    phone home                   phone back                phone raw "ping"
"""
import json
import sys
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8806/"
ARGS = {"tap": ["x", "y"], "hold": ["x", "y"], "point": ["x", "y"], "carry": ["x", "y"], "swipe": ["x1", "y1", "x2", "y2"], "scroll": ["x", "y", "lines"],
        "type": ["text"], "key": ["combo"], "raw": ["line"]}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    op, rest = sys.argv[1], [a for a in sys.argv[2:] if a != "--fast"]
    keys = ARGS.get(op, [])
    if len(rest) != len(keys):
        sys.exit(f"usage: phone {op} {' '.join(keys).upper()}")
    body = dict(zip(keys, rest))
    if op == "tap" and "--fast" in sys.argv:
        body["aim"] = False
    req = urllib.request.Request(URL + op, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            print(r.read().decode())
    except urllib.error.HTTPError as e:
        sys.exit(e.read().decode())
    except urllib.error.URLError as e:
        sys.exit(f"phoned is not running ({e.reason}): start it with `uv run pc/phoned.py`")


if __name__ == "__main__":
    main()
