# event-fetcher

Event Fetcher retrieves QuakeML events and the associated MiniSEED waveforms, applies
inventory-aware filtering, and optionally exports rotated/denoised streams.

## Quick start

```
eventfetcher -c config/eventfetcher.yml -e fr2023mlexeo
```

```
INFO:EventFetcher:event_id=fr2023mlexeo, earthquake
T0=2023-08-17T10:28:17.189095Z, lat=45.39720, lon=6.07875, depth_km=4.9
magnitude=1.26 MLv

INFO:EventFetcher:52 Trace(s) in Stream:
FR.CHA2.00.EHE | 2023-08-17T10:28:17.190000Z - 2023-08-17T10:29:17.000000Z | 100.0 Hz, 5982 samples
FR.CHA2.00.EHZ | 2023-08-17T10:28:17.190000Z - 2023-08-17T10:29:17.000000Z | 100.0 Hz, 5982 samples
...
MT.GUI.00.EHE  | 2023-08-17T10:28:17.190000Z - 2023-08-17T10:29:17.000000Z | 200.0 Hz, 11963 samples
MT.GUI.00.EHN  | 2023-08-17T10:28:17.190000Z - 2023-08-17T10:29:17.000000Z | 200.0 Hz, 11963 samples
MT.GUI.00.EHZ  | 2023-08-17T10:28:17.190000Z - 2023-08-17T10:29:17.000000Z | 200.0 Hz, 11963 samples
```

## Key capabilities

- Fetch QuakeML metadata and waveform picks from multiple FDSN profiles.
- Normalize waveform IDs, apply blacklist patterns (shell or regex), and enforce 3‑component stations.
- Use station inventory data to prefer the highest sample rate and velocimeter channels when duplicates exist.
- Filter stations by epicentral distance and trim/merge traces automatically.
- Export caches as pickle, MiniSEED, or PhaseNet bundles; optional RT rotation and denoising hooks.

## Simplified workflow

1. **Load configuration** (`eventfetcher.yml`, CLI overrides) and fetch QuakeML.
   - Alternatively provide virtual coordinates/time via CLI to synthesize an event without QuakeML.
2. **Build waveform list** from weighted picks (or distance-based search).
3. **Apply blacklist** rules and fetch station inventory for the remaining IDs.
4. **Inventory filtering**: convert to DataFrame, compute sample-rate maps, pick preferred channels.
5. **Bulk preparation**: keep only stations with 3 components (optional) and filter by distance.
6. **Data acquisition**: request traces via SDS or FDSN dataselect (parallelized if configured).
7. **Post-processing**: merge segments, attach inventory metadata, remove flat/missing traces, trim window.
8. **Persist/export** caches and optionally rotate or denoise streams.

## Configuration highlights

```yaml
starttime_offset: 0
station_max_dist_km: 350
keep_only_3channels_station: true
black_listed_waveforms_id: ["XX.GPIL", "XX.GP*", "1K.EO*"]

output:
  backup_dirname: "."
  enable_read_cache: false
  enable_write_cache: true
  write_cache_format: mseed

fdsnws:
  default_url_mapping: seiscomp
  fdsn_debug: false
  url_mapping:
    seiscomp:
      ws_base_url: http://10.90.30.115
      ws_event_url: http://10.90.30.115:8080/fdsnws/event/1/
      ws_station_url: http://10.90.30.115:8080/fdsnws/station/1/
      ws_dataselect_url: http://10.90.30.115:8080/fdsnws/dataselect/1/
```

## Parallel FDSN dataselect

The configuration files (`config/eventfetcher*.yml`) expose `fdsn_max_workers`, which controls
how many parallel `get_waveforms_bulk` chunks are issued per request:

```yaml
station_max_dist_km: 350
fdsn_max_workers: 5   # number of concurrent workers (>=1)
```

The default is `1`. Both the single-worker and parallel paths share the same retry logic
(exponential backoff, alternative location codes, per-station error isolation), so a single
unreachable station never aborts the whole fetch regardless of `fdsn_max_workers`. Increasing the
value allows faster downloads when the remote FDSN server accepts multiple simultaneous clients.
Tune the number to match the service limits of your provider.

## CLI overrides

Key command-line switches let you bypass YAML defaults when needed:

- `-e/--eventid <qml_id>` — fetch a QuakeML event by id (mutually exclusive with virtual options).
- `--virtual-lat/--virtual-lon/--virtual-time` — synthesize a virtual event using epicenter coordinates and origin time (depth defaults to 0 km).
- `--station-max-dist` — override `station_max_dist_km` from the configuration file.
- `--time-length` — override `time_length` (extraction window, in seconds) from the configuration file.
- `-o/--output <dir>` — override `backup_dirname` from the configuration file; the directory must be empty.
- `-l/--loglevel` — set the log level (`debug`, `info`, `warning`, `error`).
- `-d/--denoise` — currently disabled; the CLI exits with an error if passed (denoising is not implemented yet).

When invoking a virtual event, do **not** pass `-e/--eventid`; the CLI enforces that only one mode is selected. Example:

```
eventfetcher -c config/eventfetcher.yml --virtual-lat 48.9 --virtual-lon 7.8 \
  --virtual-time "2025/12/12 20:01" --station-max-dist 10