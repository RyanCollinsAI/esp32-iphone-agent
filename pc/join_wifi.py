# /// script
# requires-python = ">=3.10"
# dependencies = ["bleak>=0.22"]
# ///
"""Put the board on this computer's Wi-Fi (station mode), so the phone mirrors to it over the home network
and phoned reads the video over TCP. Reads the password from Windows' saved profile; never prints it.

    uv run pc/join_wifi.py [SSID]      # default: the Wi-Fi this computer is on now
"""
import asyncio
import json
import re
import subprocess
import sys

from bleak import BleakClient, BleakScanner

NET = "6e400001-a9e8-4b1b-9c39-6d6f8a3c0005"


def wifi(ssid: str | None) -> tuple[str, str]:
    if not ssid:
        out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True).stdout
        ssid = re.search(r"^\s+SSID\s+:\s(.+)$", out, re.M).group(1).strip()
    out = subprocess.run(["netsh", "wlan", "show", "profile", f"name={ssid}", "key=clear"], capture_output=True, text=True).stdout
    m = re.search(r"Key Content\s+:\s(.+)$", out, re.M)
    assert m, f"no saved password for {ssid!r} on this computer"
    return ssid, m.group(1).strip()


async def main():
    ssid, password = wifi(sys.argv[1] if len(sys.argv) > 1 else None)
    dev = await BleakScanner.find_device_by_filter(lambda d, ad: (d.name or ad.local_name or "").startswith("ESP32 Agent"), timeout=20)
    assert dev, "board not found"
    async with BleakClient(dev, timeout=20) as c:
        await c.pair()  # `net` needs an encrypted link (LE Secure Connections, Just Works)
        await c.write_gatt_char(NET, json.dumps({"mode": "sta", "ssid": ssid, "pass": password}).encode(), response=True)
        print(f"board: told to join {ssid!r}")
        for _ in range(30):
            await asyncio.sleep(1)
            net = json.loads(await c.read_gatt_char(NET))
            if net.get("mode") == "sta" and net.get("ip") not in (None, "", "0.0.0.0"):
                print(f"board: on {ssid!r} at {net['ip']}")
                return
        sys.exit(f"board did not get an address on {ssid!r} within 30 s: {json.dumps({k: v for k, v in net.items() if k != 'pass'})}")


asyncio.run(main())
