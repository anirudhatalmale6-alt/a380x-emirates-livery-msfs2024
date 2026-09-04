"""Assemble the MSFS 2024 livery package: DDS conversion, cfgs, manifest, layout."""
import json, os, shutil, subprocess, sys, glob

SRC = sys.argv[1] if len(sys.argv) > 1 else 'paint4k'
DEST = sys.argv[2] if len(sys.argv) > 2 else 'package'
SUFFIX = 'EK'
AC_DIR = 'FlyByWire_A380_842_' + SUFFIX
PKG = 'flybywire-aircraft-a380-842-livery-emirates-a6edy'

names = {k: v for k, v in json.load(open('texnames.json')).items() if not k.startswith('_')}

root = os.path.join(DEST, PKG)
tex = os.path.join(root, 'SimObjects', 'AirPlanes', AC_DIR, 'TEXTURE.' + SUFFIX)
if os.path.exists(root):
    shutil.rmtree(root)
os.makedirs(tex)

# ---- textures ----------------------------------------------------------
for mat, dds in names.items():
    png = os.path.join(SRC, mat + '.png')
    if not os.path.exists(png):
        print('MISSING', png)
        continue
    out = os.path.join(tex, dds)
    subprocess.run(['convert', png, '-define', 'dds:compression=dxt5',
                    '-define', 'dds:mipmaps=9', out], check=True)
    # No .DDS.json companions: the schema varies between sim versions and a
    # wrong one is worse than none. The sim reads the DDS header itself.
    print('%-26s -> %-40s %6.1f MB' % (mat, dds, os.path.getsize(out) / 1e6))

# ---- texture.cfg -------------------------------------------------------
up = '..\\' * 4
with open(os.path.join(tex, 'texture.cfg'), 'w') as f:
    f.write('[fltsim]\n'
            'fallback.1=%sflybywire-aircraft-a380-842\\SimObjects\\AirPlanes\\FlyByWire_A380_842\\TEXTURE\n'
            % up)

# ---- aircraft.cfg ------------------------------------------------------
acfg = """[VERSION]
major = 1
minor = 0

[VARIATION]
base_container = "..\\..\\..\\..\\flybywire-aircraft-a380-842\\SimObjects\\AirPlanes\\FlyByWire_A380_842"

[FLTSIM.0]
title = "Airbus A380-842 FlyByWire Emirates A6-EDY"
model = ""
panel = ""
sound = ""
texture = "{suffix}"
kb_checklists = ""
kb_reference = ""
description = "Emirates A380-842, registration A6-EDY."
ui_manufacturer = "Airbus"
ui_type = "A380-842"
ui_variation = "Emirates"
ui_typerole = "Commercial Airliner"
ui_createdby = "FlyByWire Simulations"
ui_thumbnailfile = ""
ui_certified_ceiling = 43000
ui_max_range = 8000
ui_autonomy = 15
ui_fuelburnrate = 12000
atc_id = "A6-EDY"
atc_id_enable = 1
atc_airline = "EMIRATES"
atc_flight_number = "1"
atc_heavy = 1
atc_id_color = "0x00000000"
atc_parking_types = "GATE"
atc_parking_codes = "UAE"
icao_airline = "UAE"
isAirTraffic = 0
isUserSelectable = 1
""".format(suffix=SUFFIX)
with open(os.path.join(root, 'SimObjects', 'AirPlanes', AC_DIR, 'aircraft.cfg'), 'w') as f:
    f.write(acfg)

# ---- manifest + layout -------------------------------------------------
with open(os.path.join(root, 'manifest.json'), 'w') as f:
    json.dump({
        "dependencies": [],
        "content_type": "AIRCRAFT",
        "title": "A380X Emirates A6-EDY",
        "manufacturer": "Airbus",
        "creator": "Anirudha Talmale",
        "package_version": "1.0.0",
        "minimum_game_version": "1.0.0",
        "release_notes": {"neutral": {"LastUpdate": "", "OlderHistory": ""}},
        "total_package_size": "00000000000000000000"
    }, f, indent=2)

EPOCH = 11644473600            # seconds between 1601-01-01 and 1970-01-01
content = []
for path in sorted(glob.glob(os.path.join(root, '**', '*'), recursive=True)):
    if os.path.isdir(path):
        continue
    rel = os.path.relpath(path, root).replace(os.sep, '/')
    if rel in ('layout.json', 'manifest.json'):
        continue
    st = os.stat(path)
    content.append({"path": rel, "size": st.st_size,
                    "date": int((st.st_mtime + EPOCH) * 10_000_000)})
with open(os.path.join(root, 'layout.json'), 'w') as f:
    json.dump({"content": content}, f, indent=2)

total = sum(c['size'] for c in content)
print('\npackage %s: %d files, %.1f MB' % (PKG, len(content), total / 1e6))
