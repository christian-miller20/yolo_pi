# Person ID Pi

Motion-triggered face identity and beer counting for a Raspberry Pi 5 and Pi Camera.

The service watches a fixed camera cheaply, captures a short event when someone enters, resolves one clear face across visits, and counts persistent cup or can detections. It stores generated identities, beer events, and annotated audit clips locally.

## Weekend-v1 behavior

- Capture: 1280×720 at 15 FPS with a two-second pre-roll.
- Trigger: low-resolution motion sampled at 5 FPS.
- Event: ends after three quiet seconds or 30 seconds maximum.
- Identity: one person at a time; clear unknown faces become `user_NNNN`.
- Beer: all YOLO cups and can-shaped YOLO bottle proposals.
- Evidence: five observations are required before a container counts.
- Dedupe: each identity can receive at most one new count every ten minutes.
- Audit: every finalized event is logged; recorded clips are marked counted/rejected.

This is object-presence counting, not proof that someone drank or dispensed a beer.

## Raspberry Pi setup

Use 64-bit Raspberry Pi OS on a Pi 5 with the Pi Camera enabled.

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-venv libgl1
sudo mkdir -p /opt/person-id-pi
sudo chown -R pi:pi /opt/person-id-pi
```

Copy the repository to `/opt/person-id-pi`, then install the Python environment:

```bash
cd /opt/person-id-pi
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
```

Run in the foreground first:

```bash
.venv/bin/python -m person_id_pi.cli serve --config config/person-id-pi.json
```

Install the boot service after the foreground smoke test succeeds:

```bash
sudo cp deploy/person-id-pi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now person-id-pi
```

Operations:

```bash
sudo systemctl status person-id-pi
journalctl -u person-id-pi -f
sudo systemctl restart person-id-pi
sudo systemctl stop person-id-pi
```

Tune camera, motion, event, detector, and storage settings in `config/person-id-pi.json`. Restart the service after changing configuration.

## Review evidence and label people

The service creates stable generated IDs such as `user_0000`. After watching an annotated evidence video, attach a human-readable display name without changing that stable ID:

```bash
python -m person_id_pi.cli review-users --data-dir .
python -m person_id_pi.cli label-user user_0001 alice --data-dir .
```

`review-users` prints every generated identity, current display name, template count, beer total, and associated evidence-video paths. `label-user` requires a non-empty name that is unique across identities.

The stable ID remains the key in historical beer events and cooldown logic. Future annotated videos display `alice`; videos rendered before labeling retain the original burned-in `user_0001` label.

For an isolated Mac smoke test, point the same commands at `smoke`:

```bash
python -m person_id_pi.cli review-users --data-dir smoke
python -m person_id_pi.cli label-user user_0001 alice --data-dir smoke
```

## Stored data and identity maintenance

- Identity templates and display names: `profiles/face_templates.json`
- Counted beers: `profiles/beverage_events.json`
- Annotated evidence clips: `events/*.mp4`
- Accepted and rejected decisions: `events/decisions.jsonl`

Display names can be assigned while the service is running. Stop the service before deleting or changing stable IDs:

```bash
sudo systemctl stop person-id-pi
.venv/bin/python -m person_id_pi.cli list-users
.venv/bin/python -m person_id_pi.cli delete-user user_0003
sudo systemctl start person-id-pi
```

Back up JSON stores before resetting them. Store writes are atomic, and SIGTERM finalizes an active event before shutdown.

## Offline diagnostics

The original face-only commands remain available for testing recordings:

```bash
python -m person_id_pi.cli enroll alice data/clip.mp4
python -m person_id_pi.cli identify data/clip.mp4 --auto-enroll-unknown
```

On a development machine, verify webcam access without saving frames:

```bash
python -m person_id_pi.cli camera-test --seconds 3
```

Run the complete pipeline against a local webcam with isolated stores:

```bash
python -m person_id_pi.cli smoke-test --seconds 60
```

Stand out of frame briefly to establish the baseline, then enter with a clear face. Introduce the cup or can only after entering. Results and audit clips are written under `smoke/`.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
pytest -q
```

The live state machine and detectors use injectable camera/model interfaces, so tests do not require Pi Camera hardware or model downloads.
