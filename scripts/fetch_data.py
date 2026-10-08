#!/usr/bin/env python3
"""Download the public footage used in RESULTS.md into data/, and verify it.

    python3 scripts/fetch_data.py            # demo set: three cameras, 135 MB, enough for the quick start
    python3 scripts/fetch_data.py all        # every fixed-camera clip of the benchmark, 25 files, 1.4 GB
    python3 scripts/fetch_data.py movie      # Tears of Steel, 372 MB (the moving-camera limit in RESULTS Part I)
    python3 scripts/fetch_data.py --check    # only ask each server whether the file is still there
    python3 scripts/fetch_data.py all --verify   # re-hash what is already on disk

Nothing here is redistributed by this repository; every file comes from its owner's server and stays
under its owner's terms:

  Xiph    Xiph.org Video Test Media ("derf's collection"), https://media.xiph.org/video/derf/
  CAVIAR  EC funded CAVIAR project, IST 2001 37540, https://homepages.inf.ed.ac.uk/rbf/CAVIAR/
  VIRAT   VIRAT Video Dataset, public release hosted by Kitware, https://viratdata.org/
          (S. Oh et al., "A large-scale benchmark dataset for event recognition in surveillance video", CVPR 2011)
  Blender Tears of Steel, (CC) Blender Foundation, mango.blender.org, CC BY 3.0

Standard library only. Files are written to a .part name first and renamed once size and SHA-256 match.
"""
import argparse, hashlib, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XIPH = "https://media.xiph.org/video/derf/y4m/"
CAVIAR = "https://homepages.inf.ed.ac.uk/rbf/"
VIRAT = "https://data.kitware.com/api/v1/item/{}/download"
BLENDER = "https://download.blender.org/demo/movies/ToS/"

# name in data/, url, bytes, sha256, in the demo set
FILES = [
    ("hall_monitor_cif.y4m", XIPH + "hall_monitor_cif.y4m", 45621044, "d0d4d6ebd137378bb5a7448c04d0cb73c20458a91d41d0f00cb090bd84bedd93", True),
    ("akiyo_cif.y4m", XIPH + "akiyo_cif.y4m", 45621044, "0f7a1f997930217d601ab324b1428859fdb599dde8d33772a24bfe2803a9b067", False),
    ("xiph_bridge_close_cif.y4m", XIPH + "bridge_close_cif.y4m", 304140044, "dadfd512f8fd4ca3f9a8bb482d5dbdb63b8a4eb56d7cb0784dd686b82691caa3", False),
    ("xiph_bridge_far_cif.y4m", XIPH + "bridge_far_cif.y4m", 319499114, "de82bc28bbd89427b24a45d73a9b1780d090aadbbb796c26018cedf94f7679d2", False),
    ("caviar_LeftBag.mpg", CAVIAR + "CAVIARDATA1/LeftBag/LeftBag.mpg", 17201025, "8f5d1dbf43d58da1bce57ca1b3f9491b8467773cfd66d2846f62f7d38d4a45e7", True),
    ("caviar_Walk1.mpg", CAVIAR + "CAVIARDATA1/Walk1/Walk1.mpg", 7208095, "a5d6dc7d7def01c1bdaeea8ff620195c148b366e1a52589bf4e46f15b6daa0c2", False),
    ("caviar_Meet_Crowd.mpg", CAVIAR + "CAVIARDATA1/Meet_Crowd/Meet_Crowd.mpg", 5700132, "223f01cdd4c6c31121a93387cb4e39305c2f6f37254d726121f39cac3729689c", False),
    ("caviar_Fight_Chase.mpg", CAVIAR + "CAVIARDATA1/Fight_Chase/Fight_Chase.mpg", 5087190, "f8cd06377f1b5e821cf6f397bf8caf0f8173e82f988947fe90a4e33c059f8f4f", False),
    ("caviar_WalkByShop1cor.mpg", CAVIAR + "CAVIARDATA2/WalkByShop1cor/WalkByShop1cor.mpg", 13629643, "8adc5aa3a54080fc389f47d5d44e6a326e592fd8843db645ab1d7d5af079183f", False),
    ("caviar_EnterExitCrossingPaths1cor.mpg", CAVIAR + "CAVIARDATA2/EnterExitCrossingPaths1cor/EnterExitCrossingPaths1cor.mpg", 2257192, "f8363a5492e57787647c0e9e525b749a864e3bcbaee48a7353060525428a1367", False),
    ("caviar_OneLeaveShopReenter1cor.mpg", CAVIAR + "CAVIARDATA2/OneLeaveShopReenter1cor/OneLeaveShopReenter1cor.mpg", 2301827, "14d55141103159aedfaf9f2be798396536cdc00a4e5113f99063276181edf937", False),
    ("VIRAT_S_000200_00.mp4", VIRAT.format("56f5851c8d777f753209ca59"), 72268926, "a8888e14bb2a1d38368a10cfd949916fcf9acc9972d3dacc3cbbc502c8d8a568", True),
    ("VIRAT_S_000200_01.mp4", VIRAT.format("56f585248d777f753209ca5c"), 43659360, "aa0a6a9b2148cf44e0b4bd934225b34203955b318e2e8808696f6ff17f8aac26", False),
    ("VIRAT_S_040000_00b_1080p.mp4", VIRAT.format("56f5882c8d777f753209cced"), 49063526, "a197864d8bd9b93e1cf96bf692b9bf59e20c66412ae0781a6dfa3f31faff6497", False),
    ("VIRAT_S_050000_05_1080p.mp4", VIRAT.format("56f58e4b8d777f753209cd7e"), 46933093, "61075cd0fb0f23ec93fec2447716a04286d86073ca2198cb1f6faa8af398f0fd", False),
    ("virat_010000_02.mp4", VIRAT.format("56f5877b8d777f753209cad4"), 5142135, "fc6ce9f62b27a552c991dfe4d806df350721eac214309e0fdfa976e0ec66302d", False),
    ("virat_010106_01.mp4", VIRAT.format("56f587af8d777f753209cb7f"), 4375781, "5051197eb0355e7f06cb57d0ac35b7626bf1851f4b008e4162744d9d095c1306", False),
    ("virat_010109_00.mp4", VIRAT.format("56f587c28d777f753209cb9d"), 12022279, "a4b5b7228b35149f05411ee610c0e089d772979e56b2a279b916f457988b1088", False),
    ("virat_010112_00.mp4", VIRAT.format("56f587d48d777f753209cbd6"), 14104986, "e4e032326060f6c22cf35dd864ca4649b8dc852b47c80353d6cb0bd468039c29", False),
    ("virat_010200_02.mp4", VIRAT.format("56f587e98d777f753209cc0c"), 6492646, "345331440714a09bf0a1015e8ffff45b4819f5980a60ea17e6851194d2c34d44", False),
    ("virat_010204_01.mp4", VIRAT.format("56f588038d777f753209cc75"), 28507466, "2a3e2944edab05d90d5efdc5b473d3388b770cd66b6e3642dd6c23a3589d95e7", False),
    ("virat_040000_02.mp4", VIRAT.format("56f5883f8d777f753209ccf3"), 76461435, "d62d21dfc3709c6797e1ce55542f5411fda40daf885aae5427e5b023b7fedded", False),
    ("virat_040103_08.mp4", VIRAT.format("56f58cf08d777f753209cd54"), 82864881, "8b750c3ac4b8c84ac768ab614682a509fd18c52a2cbb8bd1591349a5f11c8181", False),
    ("virat_050201_03.mp4", VIRAT.format("56f58ed88d777f753209cda8"), 92510542, "bea8528a958037eb0eece7d04e2b0d320cd1aa91032d364e7fbbb2b80a8af9a6", False),
    ("virat_050300_00.mp4", VIRAT.format("56f590078d777f753209cdf6"), 89562541, "6d8d11ae42bf075a89d4c35ae802786e3159c10b73e3cfeadd3bef8a77ef3d6d", False),
]
MOVIE = [("tears_of_steel_720p.mov", BLENDER + "tears_of_steel_720p.mov", 372178639, "efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f", False)]
UA = {"User-Agent": "certified-skip-fetch/1.0 (+https://github.com/nsquaredzz/certified-skip)"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def good(path, size, digest):
    return path.exists() and path.stat().st_size == size and sha256(path) == digest


def remote_size(url):
    """Size the server reports, from a one-byte range request (some of these servers refuse HEAD)."""
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range")
        return int(cr.rsplit("/", 1)[1]) if cr and "/" in cr else int(r.headers.get("Content-Length", -1))


def download(url, dest, size):
    part = dest.with_name(dest.name + ".part")
    t0 = time.time(); done = 0
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(part, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); done += len(chunk)
            if sys.stderr.isatty():
                sys.stderr.write(f"\r    {done / 1e6:7.1f} / {size / 1e6:.1f} MB  {done / 1e6 / max(time.time() - t0, 1e-3):5.1f} MB/s")
    if sys.stderr.isatty():
        sys.stderr.write("\r" + " " * 60 + "\r")
    return part


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("which", nargs="?", default="demo", choices=["demo", "all", "movie"])
    ap.add_argument("--dest", default=str(ROOT / "data")); ap.add_argument("--only", default=None, help="comma-separated file names")
    ap.add_argument("--check", action="store_true", help="ask the servers, download nothing")
    ap.add_argument("--verify", action="store_true", help="hash the files on disk, download nothing")
    a = ap.parse_args()
    files = MOVIE if a.which == "movie" else [f for f in FILES if a.which == "all" or f[4]]
    if a.only:
        files = [f for f in FILES + MOVIE if f[0] in set(a.only.split(","))]
    dest = Path(a.dest); dest.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} files, {sum(f[2] for f in files) / 1e6:.0f} MB, into {dest}")
    bad = 0
    for name, url, size, digest, _ in files:
        path = dest / name
        try:
            if a.check:
                n = remote_size(url); ok = n == size
                print(f"  {'ok     ' if ok else 'CHANGED'} {name}  ({n} bytes on the server, {size} expected)"); bad += not ok
            elif a.verify:
                ok = good(path, size, digest); print(f"  {'ok     ' if ok else 'MISSING' if not path.exists() else 'BAD    '} {name}"); bad += not ok
            elif good(path, size, digest):
                print(f"  have    {name}")
            else:
                print(f"  get     {name}  {size / 1e6:.1f} MB", flush=True)
                part = download(url, path, size)
                if part.stat().st_size != size or sha256(part) != digest:
                    print(f"  BAD     {name}: the download does not match the recorded size and SHA-256 (kept as {part.name})"); bad += 1
                else:
                    part.replace(path)
        except Exception as e:                                    # one dead server should not stop the others
            print(f"  FAILED  {name}: {e}"); bad += 1
    if bad:
        sys.exit(f"{bad} of {len(files)} files need attention")
    print("all good")


if __name__ == "__main__":
    main()
