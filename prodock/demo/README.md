# ProDock live demo on real Docker

The simulated evaluation (`run_experiment.py`) replays 7 days of numbers in software. **This demo
runs it for real:** real containers burn real CPU and hold real memory, the Linux kernel really
throttles them (CFS quota) or OOM-kills them, and ProDock really resizes them with
`docker update`, with no restart.

## What runs

| Container | Load it replays | Who sets its limits |
|---|---|---|
| `web-frontend` | web-frontend trace | **ProDock** (Isolation Forest + LSTM + allocator) |
| `web-frontend-reactive` | the same trace | threshold rules (>80% used → ×1.5, <30% → ×0.75) |
| `web-frontend-static` | the same trace | nobody: fixed limit, like most deployments |
| `batch-worker` | batch-worker trace (job every 2 sim-hours) | **ProDock** |

All of them start at the same static limit. Each container runs `loadgen/burn.py`. Every tick, it
looks up its row in the trace, spins that many CPU cores (a duty cycle across several worker
processes), and holds that many GB in 16 MB blocks of touched memory.

**Time is compressed:** 1 simulated minute = 2 real seconds. ProDock ticks once per simulated
minute, so the model sees exactly the time scale it was trained on (30 minutes of history,
peak of the next 3 minutes). A 12-minute demo covers 6 hours of the test day, with two flash
crowds and two batch jobs. Everyone follows the same clock in `clock.json`.

ProDock never sees the trace. It only sees what `docker stats` reports: CPU used, throttled
time, memory and network. The trace is used only afterwards, to score SLO misses fairly.

## Before the demo (one time)

1. Start **Docker Desktop** and wait until it says "Engine running".
2. `pip install -r requirements.txt`
3. `python run_experiment.py`, if `models/prodock_pipeline.pkl` is missing or was made with a
   different scikit-learn version.
4. Rehearse the whole thing once. The first `up` downloads `python:3.12.7-slim-bookworm` (~50 MB).

## Running it (three terminals, from the `prodock/` folder)

```bash
# terminal 1: start the containers (the clock starts 15 s later)
python demo/demo.py up --minutes 12

# terminal 1: ProDock + the reactive baseline, really resizing
python run_live.py --demo --apply

# terminal 2: the live chart
python demo/dashboard.py

# terminal 3 (optional, very convincing): Docker's own view of the limits changing
docker stats web-frontend web-frontend-reactive web-frontend-static batch-worker

# afterwards
python demo/demo.py down
```

`run_live.py` stops by itself at the end of the trace and prints a scoreboard. Ctrl+C stops
it early, and it prints the scoreboard anyway. A snapshot of the chart:
`python demo/dashboard.py --snapshot results/live_demo.png`.

Options: `demo.py up --tick 1` (twice as fast), `--no-batch` (less CPU), `--minutes N`.
Peak load is about 7 to 9 busy cores, so close heavy apps and keep the laptop plugged in.

## What to point at (timeline at 2 s per tick)

| Real time | Sim time | What happens |
|---|---|---|
| 0:00–1:20 | 00:00–00:40 | ProDock fills its 40-tick history ("warm-up"). batch-worker's job runs, so ProDock's memory floor learns how big a job gets. |
| ~1:20 | 00:40 | ProDock starts deciding. It shrinks web-frontend from 2.18 cores toward ~1 core. Static keeps 2.18 (waste). |
| ~4:00 | 02:00 | batch-worker job starts: CPU 0.1 → 2.5 cores, memory 0.25 → 1.85 GB. The memory limit is already high enough, so no OOM. CPU is late for a tick or two (known limitation: the job is every 2 h, the LSTM sees 30 min). |
| ~5:40 | 02:49 | **Flash crowd 1 on web-frontend.** The burst score crosses the line, ProDock prints `burst-up`, and the limit jumps (≈0.9 → 1.7 → 3+ cores) as the spike begins. Reactive only reacts after it is already >80% used. |
| ~9:50 | 04:55 | Flash crowd 2. Same story. |
| end | | Scoreboard: SLO misses, kernel-throttled ticks, avg limit, wasted core-minutes, resizes, OOM kills. |

## Reading the numbers

- **SLO miss**: true demand > CPU limit + 2%. This is the same rule as the simulated evaluation.
- **kernel thr**: ticks where ≥10% of the kernel's 100 ms CFS periods were throttled (from
  `cpu.stat`). This is the operating system's own count. ProDock runs containers closer to their
  limit, so it shows some kernel throttling that static never shows. That is the honest price of
  using far less CPU.
- **wasted**: core-minutes reserved but not used. This is where static loses badly.
- **OOM/deaths**: the container died (restart count went up) and Docker restarted it.

## If something goes wrong

- `run_live.py --demo --check` tests that Docker is reachable, the containers exist and stats work.
- "cannot talk to Docker": start Docker Desktop.
- A container name is already in use: an old container has the same name. `docker rm -f <name>`.
- Fallback: record the screen during the rehearsal, and keep `results/live_demo.sqlite` plus a
  dashboard snapshot from a good run.
