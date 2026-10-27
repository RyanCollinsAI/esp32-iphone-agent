# /// script
# requires-python = ">=3.10"
# dependencies = ["bleak>=0.22", "av>=12", "numpy", "pillow", "aiohttp"]
# ///
"""phoned: drive the iPhone from a Windows/Linux/Mac computer through the ESP32 board, no iOS app needed.

The board joins the home Wi-Fi (pc/join_wifi.py) and the phone mirrors to it there. The computer sends HID
commands over BLE (docs/ble-protocol.md) and reads the mirrored H.264 from the board over TCP, then moves the
AssistiveTouch pointer with the same closed loop as the app (Pointer.swift, Vision.swift): relative mouse
moves from a homed corner, a span calibrated by finding the pointer on screen, and a check before each tap.

    uv run pc/phoned.py            # serves http://127.0.0.1:8806 ; drive it with pc/phone.py
"""
import asyncio
import json
import struct
import time
from pathlib import Path

import av
import numpy as np
from aiohttp import web
from bleak import BleakClient, BleakScanner
from PIL import Image, ImageDraw

BASE = "6e400001-a9e8-4b1b-9c39-6d6f8a3c"
CMD, REPLY, STATUS = (BASE + s for s in ("0001", "0002", "0003"))
IDR, CONFIG = 4, 8
TCP_PORT = 7200
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CALIB = HERE / "calibration.json"
PORT = 8806

NUDGE = 12               # mouse units the pointer is wiggled by to find it on screen
STOPS = (150, 350)       # calibration finds the pointer at these distances from the homed corner
REHOME_EVERY = 10        # gestures between re-homing
AIM_TOLERANCE = 0.008    # screen fraction; a bigger miss is corrected before pressing
MAX_MISS = 0.06          # bigger misses are corrected but not learned from
AIM_TRIES = 4            # measure-and-correct rounds before a tap
CLICK_HOLD = 0.12        # s between press and release of a tap
TAP_GAP = 0.15          # s to stay still after a click so iOS sees a finished tap
HOLD = 0.25              # s to rest before lifting at the end of a swipe
SHOW = 0.35              # s from a HID command until the phone's mirrored frame shows it at the board


class Screen:
    """The mirrored screen from the board's TCP stream (port 7200, see video_link.cpp): Windows BLE carries only
    ~120 kbps, too little for video, so BLE does HID and Wi-Fi does video. Keeps the latest frame (gray for
    diffs, RGB for screenshots) and when it reached the board (board clock)."""

    def __init__(self):
        self.codec = av.CodecContext.create("h264", "r")
        self.waiting = True  # no IDR since (re)connecting: P-frames alone decode to smeared garbage
        self.gray: np.ndarray | None = None
        self.rgb: np.ndarray | None = None
        self.at = 0.0  # monotonic time of the latest decoded frame
        self.shown = 0  # board clock (ms) when the latest decoded frame reached the board
        self.offset: float | None = None  # board ms - local ms, set from status heartbeats
        self.frames = 0
        self.changed = asyncio.Event()

    async def run(self, board_ip):
        """Read the board's stream forever; a new connection starts with SPS/PPS and the GOP from its IDR."""
        while True:
            ip = board_ip()
            if not ip:
                await asyncio.sleep(1)
                continue
            try:
                reader, writer = await asyncio.open_connection(ip, TCP_PORT)
                print(f"video: connected to {ip}:{TCP_PORT}", flush=True)
                self.codec, self.waiting = av.CodecContext.create("h264", "r"), True
                while True:  # mirroring sends a few frames a second even on a still screen; silence = dead link
                    n, flags, ts = struct.unpack("<IBI", await asyncio.wait_for(reader.readexactly(9), 20))
                    self._au(await asyncio.wait_for(reader.readexactly(n), 20), flags, ts)
            except (OSError, asyncio.IncompleteReadError, TimeoutError) as e:
                print(f"video: {type(e).__name__} {e}, reconnecting", flush=True)
                await asyncio.sleep(2)

    def _au(self, au: bytes, flags: int, ts: int):
        if self.waiting and not flags & (CONFIG | IDR):
            return
        if flags & IDR:
            self.waiting = False
        try:
            frames = self.codec.decode(av.Packet(au))
        except av.error.InvalidDataError:
            self.waiting = True
            return
        for f in frames:
            self.rgb = f.to_ndarray(format="rgb24")
            r, g, b = (self.rgb[..., i].astype(np.uint16) for i in range(3))
            self.gray = ((r * 299 + g * 587 + b * 114 + 500) // 1000).astype(np.uint8)
            self.at, self.shown = time.monotonic(), ts
            self.frames += 1
            self.changed.set()

    @property
    def live(self) -> bool:
        return self.gray is not None

    async def after(self, t: float, timeout: float = 3.0) -> np.ndarray:
        """The screen as it was SHOW s after local time `t`: the first frame that reached the board by then.
        The video lags ~1 s over BLE, so "the latest frame" right after a command still shows the old screen.
        A still screen sends no frames, so on timeout the latest frame is the answer."""
        assert self.offset is not None, "no status heartbeat yet, board clock unknown"
        want = (t + SHOW) * 1000 + self.offset
        end = time.monotonic() + timeout
        while self.shown < want and time.monotonic() < end:
            self.changed.clear()
            try:
                await asyncio.wait_for(self.changed.wait(), end - time.monotonic())
            except TimeoutError:
                break
        return self.gray

    async def still(self, timeout: float = 4.0) -> np.ndarray:
        """The screen once animations (page slides, app launches) have stopped: two frames ~0.3 s apart that match."""
        end = time.monotonic() + timeout
        prev = await self.after(time.monotonic())
        while time.monotonic() < end:
            await asyncio.sleep(0.3)
            cur = await self.after(time.monotonic())
            if np.count_nonzero(np.abs(cur.astype(np.int16) - prev.astype(np.int16)) > 30) < 400:
                return cur
            prev = cur
        return prev


def find_pointer(a: np.ndarray, b: np.ndarray, near, shift: float, reach=(0.12, 0.08)):
    """Where the pointer is in `a`, given it moved right by `shift` (fraction of width) in `b`. Port of Vision.pointer."""
    if a.shape != b.shape:
        return None
    h, w = a.shape
    left, right = max(0, int((near[0] - reach[0]) * w)), min(w, int((near[0] + reach[0]) * w))
    top, bottom = max(int(0.05 * h), int((near[1] - reach[1]) * h)), min(h, int((near[1] + reach[1]) * h))
    if right <= left or bottom <= top:
        return None
    diff = np.abs(a[top:bottom, left:right].astype(np.int16) - b[top:bottom, left:right].astype(np.int16)) > 30
    ys, xs = np.nonzero(diff)
    print(f"pointer: {len(xs)} changed px near {near}"
          + (f", spread {xs.max() - xs.min()}x{ys.max() - ys.min()}" if len(xs) else ""), flush=True)
    if not 15 <= len(xs) <= diff.size // 4:
        return None  # nothing moved, or more than a pointer
    if xs.max() - xs.min() > 0.15 * w or ys.max() - ys.min() > 0.07 * h:
        return None  # not one compact spot
    return ((xs.mean() + left) / w - shift / 2, (ys.mean() + top) / h)


class Board:
    def __init__(self, screen: Screen):
        self.screen = screen
        self.client: BleakClient | None = None
        self.pending: dict[str, asyncio.Future] = {}
        self.next_id = 0
        self.status: dict = {}
        self.links = 0
        self.lock = asyncio.Lock()

    async def connect(self):
        dev = await BleakScanner.find_device_by_filter(
            lambda d, ad: (d.name or ad.local_name or "").startswith("ESP32 Agent"), timeout=20)
        assert dev, "board not found: is it plugged into the phone (powered) and in range?"
        self.client = BleakClient(dev, timeout=20, winrt={"use_cached_services": False},
                                  disconnected_callback=lambda _: print("board: link lost", flush=True))
        await self.client.connect()
        if self.client.services.get_characteristic(CMD) is None:  # Windows reused a dying link from a previous run
            await self.client.disconnect()
            raise RuntimeError("board services missing on this link, reconnecting")
        await self.client.start_notify(REPLY, self._on_reply)
        await self.client.start_notify(STATUS, self._on_status)
        self.links += 1
        print(f"board: connected to {dev.name}, mtu {self.client.mtu_size}", flush=True)
        await self.send("stream off")  # video comes over Wi-Fi; keep BLE free for commands

    def _on_reply(self, _, data: bytearray):
        rid, _, rest = data.decode(errors="replace").partition(" ")
        fut = self.pending.pop(rid, None)
        if fut and not fut.done():
            fut.set_result(rest)

    def _on_status(self, _, data: bytearray):
        try:
            self.status = json.loads(data)
        except ValueError:
            return
        # the heartbeat left the board before we got it, so the largest up - now is the closest clock offset
        offset = self.status["up"] - time.monotonic() * 1000
        self.screen.offset = offset if self.screen.offset is None else max(self.screen.offset, offset)

    @property
    def connected(self) -> bool:
        return bool(self.client and self.client.is_connected)

    async def send(self, line: str, timeout: float = 10) -> str:
        """One command; raises on a missing or `err` reply. Ids 900000+ (the phone's app uses 100000-800000)."""
        if not self.connected:
            await self.connect()
        self.next_id = (self.next_id + 1) % 99999
        rid = str(900000 + self.next_id)
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        async with self.lock:
            await self.client.write_gatt_char(CMD, f"{rid} {line}".encode(), response=True)
        reply = await asyncio.wait_for(fut, timeout)
        if reply.startswith("err"):
            raise RuntimeError(f"{line}: {reply}")
        return reply


class Pointer:
    """Touch gestures from relative mouse moves. Port of Pointer.swift; coordinates are screen fractions 0..1."""

    def __init__(self, board: Board, screen: Screen):
        self.board, self.screen = board, screen
        self.span: tuple[int, int] | None = None
        self.seen: dict[str, list[tuple[float, float]]] = {"x": [], "y": []}
        self.pos: tuple[int, int] | None = None
        self.moves = 0
        self.link = 0
        if CALIB.exists():
            saved = json.loads(CALIB.read_text())
            self.span = (saved["w"], saved["h"])
            self.seen = {k: [tuple(p) for p in v] for k, v in saved["seen"].items()}

    def _save(self):
        CALIB.write_text(json.dumps({"w": self.span[0], "h": self.span[1], "seen": self.seen}))

    def _remember(self, axis: str, sample: tuple[float, float]):
        self.seen[axis] = (self.seen[axis] + [sample])[-12:]

    def fix(self, t: float, axis: str) -> float:
        """The fraction to aim at so the pointer lands on t, from the line through the observed landings."""
        pts = self.seen[axis]
        if len(pts) < 2:
            return t
        mc, mf = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
        var = sum((p[0] - mc) ** 2 for p in pts)
        a = sum((p[0] - mc) * (p[1] - mf) for p in pts) / var if var > 1e-4 else 1.0
        a = a if 0.85 < a < 1.15 else 1.0
        b = mf - a * mc
        return (t - b) / a if abs(b) < 0.1 else t

    def _units(self, x: float, y: float) -> tuple[int, int]:
        return (round(min(max(self.fix(x, "x"), 0), 1) * self.span[0]), round(min(max(self.fix(y, "y"), 0), 1) * self.span[1]))

    async def _move_to(self, tx: int, ty: int, rehome: bool = True):
        if self.link != self.board.links:  # new link: a dropped gesture may have left the button down
            await self.board.send("release")
            self.pos, self.link = None, self.board.links
        if self.pos is None or (rehome and self.moves >= REHOME_EVERY):
            await self.board.send("home")
            self.pos, self.moves = (0, 0), 0
        x, y = self.pos
        self.pos = None
        await self.board.send(f"move {tx - x} {ty - y}")
        self.pos, self.moves = (tx, ty), self.moves + 1

    async def _wiggle(self, near, reach, nudge: int):
        """Nudge the pointer right and back; returns where it was (fractions) or None."""
        before = await self.screen.still()
        await self.board.send(f"move {nudge} 0")
        after = await self.screen.after(time.monotonic())
        await self.board.send(f"move {-nudge} 0")
        shift = nudge / (self.span[0] if self.span else 400)
        return find_pointer(before, after, near, shift, reach) or find_pointer(before, after, (0.5, 0.5), shift, (0.5, 0.5))

    async def calibrate(self) -> dict:
        assert self.screen.live, "no screen yet: is Screen Mirroring to ESP32 Agent on?"
        self.pos = None
        await self.board.send("home")
        found = []
        for units in STOPS:
            step = units - (found[-1][0] if found else 0)
            await self.board.send(f"move {step} {step}")
            p = await self._wiggle((0.5, 0.5), (0.5, 0.5), NUDGE)
            if p is None or not (0.03 < p[0] < 0.97 and 0.03 < p[1] < 0.97):
                raise RuntimeError(f"calibration: pointer not found on screen after {units} units "
                                   "(is AssistiveTouch on and the pointer visible?)")
            found.append((units, *p))
        (u1, x1, y1), (u2, x2, y2) = found
        self.span = (round((u2 - u1) / (x2 - x1)), round((u2 - u1) / (y2 - y1)))
        self.seen = {"x": [], "y": []}
        for units, x, y in found:
            self._remember("x", (units / self.span[0], x))
            self._remember("y", (units / self.span[1], y))
        self._save()
        self.pos, self.moves = (u2, u2), 0
        return {"span": self.span, "found": found}

    async def point(self, x: float, y: float, aim: bool):
        assert self.span, "not calibrated: run `phone calibrate` first"
        await self._move_to(*self._units(x, y))
        if not aim:
            return None
        misses = []
        for i in range(AIM_TRIES):  # iOS accelerates long moves differently from short ones: correct until close
            found = await self._wiggle((x, y), (0.2, 0.2), NUDGE if x < 0.9 else -NUDGE)
            if found is None:  # an idle pointer fades in on the first wiggle: look once more
                found = await self._wiggle((x, y), (0.2, 0.2), NUDGE if x < 0.9 else -NUDGE)
            if found is None:  # never press blind: the button next door may be "Delete App"
                raise RuntimeError(f"pointer not found after {misses or 'no'} corrections; not pressing (retry)")
            fx, fy = found
            misses.append(f"{fx - x:+.3f},{fy - y:+.3f}")
            if i == 0 and max(abs(fx - x), abs(fy - y)) <= MAX_MISS:
                self._remember("x", (self.fix(x, "x"), fx))
                self._remember("y", (self.fix(y, "y"), fy))
                self._save()
            if abs(fx - x) <= AIM_TOLERANCE and abs(fy - y) <= AIM_TOLERANCE:
                return f"on target, misses {' '.join(misses)}"
            mx, my = round((x - fx) * self.span[0]), round((y - fy) * self.span[1])
            await self.board.send(f"move {mx} {my}")
            if self.pos:
                self.pos = (self.pos[0] + mx, self.pos[1] + my)
        raise RuntimeError(f"still off after {AIM_TRIES} corrections, misses {' '.join(misses)}; not pressing (retry)")

    async def tap(self, x: float, y: float, aim: bool = True):
        note = await self.point(x, y, aim)
        await self.board.send("press")  # the board's instant "click" is ignored by iOS alert buttons
        try:
            await asyncio.sleep(CLICK_HOLD)
        finally:
            await self.board.send("release")
        await asyncio.sleep(TAP_GAP)
        return note

    async def drag(self, points: list[tuple[float, float]], hold: bool = True):
        await self.point(*points[0], aim=False)
        await self.board.send("press")
        try:
            for p in points[1:]:
                await self._move_to(*self._units(*p), rehome=False)  # never home mid-drag
            if hold:
                await asyncio.sleep(HOLD)
        finally:
            await self.board.send("release")

    async def back(self):
        await self.drag([(0.005, 0.45), (0.6, 0.45)], hold=False)
        self.pos = None  # the screen edge may have clamped the pointer


def grid(rgb: np.ndarray) -> Image.Image:
    """The frame with a labelled 0-1000 grid, so coordinates can be read off it."""
    img = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(1, 10):
        x, y = w * i / 10, h * i / 10
        d.line([(x, 0), (x, h)], fill=(255, 0, 255), width=1)
        d.line([(0, y), (w, y)], fill=(255, 0, 255), width=1)
        d.text((x + 2, 2), str(i * 100), fill=(255, 0, 255))
        d.text((2, y + 2), str(i * 100), fill=(255, 0, 255))
    return img


def main():
    OUT.mkdir(exist_ok=True)
    screen = Screen()
    board = Board(screen)
    pointer = Pointer(board, screen)
    shots = {"n": 0}

    def frac(v) -> float:
        f = float(v) / 1000
        assert 0 <= f <= 1, f"coordinates are 0-1000, got {v}"
        return f

    async def handle(req: web.Request):
        op = req.match_info["op"]
        a = await req.json() if req.can_read_body else {}
        try:
            if op == "status":
                return web.json_response({"connected": board.connected, "frames": screen.frames,
                                          "frame_age_s": round(time.monotonic() - screen.at, 1) if screen.live else None,
                                          "stale": screen.waiting,
                                          "calibrated": pointer.span, "board": board.status})
            if op == "screen":
                assert screen.live, "no screen yet: is Screen Mirroring to ESP32 Agent on and the phone unlocked?"
                assert not screen.waiting, "no full frame yet: is the phone mirroring to ESP32 Agent?"
                await screen.still()
                shots["n"] += 1
                plain, gridded = OUT / "screen.png", OUT / "screen-grid.png"
                Image.fromarray(screen.rgb).save(plain)
                grid(screen.rgb).save(gridded)
                return web.json_response({"png": str(plain), "grid": str(gridded), "size": list(screen.rgb.shape[1::-1]),
                                          "age_s": round(time.monotonic() - screen.at, 1)})
            if op == "calibrate":
                return web.json_response(await pointer.calibrate())
            if op == "tap":
                note = await pointer.tap(frac(a["x"]), frac(a["y"]), aim=a.get("aim", True))
                return web.json_response({"ok": True, "aim": note})
            if op == "swipe":
                await pointer.drag([(frac(a["x1"]), frac(a["y1"])), (frac(a["x2"]), frac(a["y2"]))], hold=a.get("hold", True))
            elif op == "hold":  # long press: context menus, home screen edit mode
                note = await pointer.point(frac(a["x"]), frac(a["y"]), aim=True)
                await board.send("press")
                try:
                    await asyncio.sleep(float(a.get("secs", 1.2)))
                finally:
                    await board.send("release")
                return web.json_response({"ok": True, "aim": note})
            elif op == "type":
                await board.send(f"type {a['text']}")
            elif op == "key":
                await board.send(f"key {a['combo']}")
            elif op == "home":
                await board.send("homescreen")
            elif op == "back":
                await pointer.back()
            elif op == "scroll":
                await pointer.point(frac(a["x"]), frac(a["y"]), aim=False)
                await board.send(f"scroll {int(a['lines'])} 0")
            elif op == "raw":
                return web.json_response({"reply": await board.send(a["line"])})
            else:
                return web.json_response({"error": f"unknown op {op}"}, status=404)
            return web.json_response({"ok": True})
        except (AssertionError, RuntimeError, TimeoutError, KeyError, OSError) as e:
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)

    async def keepalive(app):
        async def loop():
            while True:
                try:
                    if not board.connected:
                        await board.connect()
                except Exception as e:  # board off or out of range: say so and keep trying
                    print(f"board: {e}", flush=True)
                await asyncio.sleep(2)
        board_ip = lambda: board.status.get("ip") if board.status.get("net") == "sta" else None  # noqa: E731
        app["keepalive"] = asyncio.create_task(loop())
        app["video"] = asyncio.create_task(screen.run(board_ip))

    app = web.Application()
    app.router.add_route("*", "/{op}", handle)
    app.on_startup.append(keepalive)
    print(f"phoned: http://127.0.0.1:{PORT}  (pc/phone.py talks to it)", flush=True)
    web.run_app(app, host="127.0.0.1", port=PORT, print=None)


if __name__ == "__main__":
    main()
