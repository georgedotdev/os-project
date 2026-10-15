# ProDock: Task List

Status as of 2026-10-14. Priority: **P0** = needed for the live demo, **P1** = should do before the review, **P2** = nice to have or future work.

---

## A. Environment setup (P0)

- [x] **A1.** Start Docker Desktop. It's installed (v29.1.3), but the engine isn't running, so `docker info` fails.
- [x] **A2.** Install the missing Python packages: `pip install -r prodock/requirements.txt`. `scikit-learn` and the `docker` SDK aren't installed. numpy, pandas and matplotlib may also be missing.
- [ ] **A3.** Pin the package versions in `requirements.txt`. The file currently uses `>=`. `models/prodock_pipeline.pkl` is a pickle containing a scikit-learn IsolationForest, and loading it with a different scikit-learn version can fail or give warnings.
- [x] **A4.** Run `python run_experiment.py` once, so the model is rebuilt with the scikit-learn version installed on this machine (about 1 to 2 minutes, fixed seed, same numbers).
- [ ] **A5.** Run `git init` and make a first commit. The project currently has no version control. The only "backup" is the zip file.

## B. Live demo on real Docker (P0): what the teacher is asking for

The current "demo" (`run_demo.py`) only replays simulated numbers and prints `[dry-run] docker update ...`. No container exists and nothing is ever resized. The real-Docker script (`run_live.py`) exists but has never been run, and it has several problems that would make a live run look broken:

- [x] **B1. Fix the time-scale mismatch (the biggest bug).** The model was trained on 1 tick = 1 minute: it looks back 30 minutes and predicts the peak of the next 3 minutes. `run_live.py` ticks every 2 s, so the LSTM sees 60 s of history and "predicts" 6 s ahead. The burst detector was calibrated on minute-level noise, so 2-second jitter would set off constant false "bursts". The 3-hour memory lookback also shrinks to 6 minutes, and the cooldown to 10 s.
  **Fix:** add a "compressed clock" mode where **1 tick = 1 simulated minute = N real seconds** (for example 2 s), and the time-of-day feature comes from the simulated clock, not the wall clock.
- [x] **B2. Replace the flat stress-ng workload.** `demo/docker-compose.yml` runs `stress-ng` at a *constant* load, with no spikes and no daily pattern, so ProDock would have nothing to predict. Build a small **trace-driven load generator** (`demo/loadgen/burn.py`, image `python:3.12-slim`) that reads the per-minute CPU/memory trace for its container and **really burns** that many cores (a duty-cycle busy loop across several processes) and **really holds** that much memory, advancing one trace row every N seconds.
- [x] **B3. New `docker-compose.yml`** with containers named exactly like the trained model expects (`web-frontend`, `batch-worker`, and optionally `api-service` and `cache-db`). Pin the image versions. Plus a **static "twin"** (`web-frontend-static`) that replays the same trace under a fixed limit and is never resized, as the live baseline.
- [x] **B4. Measure SLO violations live from the kernel.** Read `nr_throttled` / `throttled_time` from the container's cgroup `cpu.stat` (Docker exposes these in `docker stats` as `throttling_data`) and count throttled ticks for the ProDock container and for the static twin. Because the trace gives the true demand, you can also check the collector's "usage + throttled" demand estimate against the real value.
- [x] **B5. Pre-fill the history buffer.** (Done: the clock starts 40 ticks early at minute 7200, a batch-job start like the simulation, and ProDock's memory floor is seeded from the warm-up readings.) The model needs 40 ticks before its first decision (80 s at 2 s per tick). Either seed the buffer from the trace rows just before the start, or accept the warm-up and talk over it.
- [x] **B6. Live dashboard.** Add a live chart that reads `results/live_decisions.sqlite` (a matplotlib animation is the simplest; Streamlit is the nicest). Panels: demand vs CPU limit for ProDock and for the static twin, the burst score against its threshold, memory, and running throttle counts.
- [x] **B7. Detect OOM kills live.** Watch `docker events --filter event=oom` or check `State.OOMKilled`, and log them.
- [x] **B8. Executor robustness.** Handle a refused `docker update` (for example, a memory limit below current usage) by retrying with the memory floor and logging it. Never shrink memory below current usage plus a margin.
- [x] **B9. Write and rehearse a demo script** (`demo/README.md`): start compose, run `--check`, start ProDock with `--apply`, open the dashboard, point at a burst-up, show `docker inspect` / `docker stats` changing in a second terminal, then show the throttle counts for the twin vs ProDock.
- [ ] **B10. Fallback plan.** Record the full live run (screen recording plus the SQLite log), in case Docker misbehaves on the day.

## C. Using real workloads instead of a replayed trace (P1/P2)

- [ ] **C1.** Write `train_from_csv.py`. `run_live.py --record` writes a CSV, and the docs say "retrain on that CSV", **but no script does that**. It should train a Pipeline from recorded samples with any container names.
- [ ] **C2.** Make container names configurable. Right now the model only knows the 4 simulated names.
- [ ] **C3.** Optional realistic demo: nginx or Flask plus a load generator (`hey` / `k6` / `locust`) firing bursts, record about an hour, retrain with C1, then run live.

## D. Model / method holes (P1/P2)

- [ ] **D1.** Scheduled jobs (`batch-worker`) are often missed on their first minute, because the window only covers 30 minutes and the job runs every 2 hours. Add a "same time yesterday" or "time since last job" feature.
- [ ] **D2.** The Isolation Forest scores its own training rows (in-sample). Fit it on days 1 to 3 and score day 4, or use out-of-fold scores.
- [ ] **D3.** The demand estimate adds `throttled_time / dt`, which can over-count when several threads are throttled at once. Document this or cap it.
- [ ] **D4.** Add tests for `docker_io.py`, using a fake Docker stats dict so they run without Docker, and for the pipeline round trip (`save` → `load` → same forecast).
- [ ] **D5.** Small comment error in `pipeline.py`: "29th-column style relay" should say 10th column.

## E. Paper (`AI_Docker_Resource_Allocation_Review_with_ProDock.docx`) (P1)

A text scan of the .docx still finds:
- [ ] **E1.** "Various authors" appears **12 times**. Put in real author names (references [1] to [5] and [8] at least).
- [ ] **E2.** "Member 1" placeholders appear **7 times**. Add names and the course code.
- [ ] **E3.** The leftover "Section B" is still there. Change it to the correct section number.
- [ ] **E4.** "likely" appears 3 times. Fix the one in Section 2.7 (confirm what the paper does, or drop the sentence).
- [ ] **E5.** Check whether ExtremoNet really uses a TCN ("TCN" appears 5 times), check ConLAR's title ("Workloads", plural), and check the year of arXiv 2311.07818.
- [ ] **E6.** Add a short subsection on the **live Docker experiment** once part B works: setup, the twin comparison, and the measured throttle counts. Keep the simulated 7-day results as the main evaluation.
- [ ] **E7.** Reference numbering order (jump from [7] to [22]) and the patent-confidentiality note in Section 5.4.

## G. Live demo: built 2026-10-14

- [x] `demo/loadgen/burn.py`: trace-driven load generator (real CPU duty cycle plus real touched memory).
- [x] `demo/docker-compose.yml`: 4 containers (`web-frontend` ProDock, `-reactive`, `-static` twins, `batch-worker` ProDock).
- [x] `demo/demo.py up|down`: exports the trace, computes static limits, writes the shared `clock.json`, starts compose.
- [x] `run_live.py --demo`: compressed clock (1 tick = 1 sim minute = 2 s), reactive baseline, kernel throttle counters, OOM/restart detection, scoreboard.
- [x] `demo/dashboard.py`: live matplotlib chart plus scoreboard (`--snapshot` saves a PNG).
- [x] Fixed: `run_live.py` used to skip memory-only changes (the policy reports `hold` when only memory moved).
- [x] Fixed: the executor applies the CPU part when Docker refuses a memory change.
- [x] Fixed: `--check` only checks model names for the containers ProDock controls.
- [ ] **B10.** Record a screen capture of a good rehearsal as a fallback.
- [ ] Add a screenshot of the live dashboard and the scoreboard table to the paper (E6).
- [ ] Optional: a "ConLAR-style predictive" twin as well, so all 4 policies run live.

## F. Housekeeping (done 2026-10-14)

- [x] Unpacked `ProDock_code.zip` into `prodock/`. The code had only existed inside the zip.
- [x] Removed duplicate loose files from the project root (`fig1`–`fig9` PNGs, `docker_io.py`, `run_live.py`). They are identical to the copies in `prodock/figures/` and `prodock/`.
- [x] Ran `python prodock/tests/test_core.py`: 3/3 pass.
