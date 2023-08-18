# event-fetcher
Event Fetcher : event (quakeml) and waveforms (mseed, ...)

```
./eventfetcher.py -c eventfetcher.yml -e fr2023mlexeo
```

```
INFO:EventFetcher:event_id=fr2023mlexeo, earthquake
T0=2023-08-17T10:28:17.189095Z, lat=45.39720, lon=6.07875, depth_km=4.9
magnitude=1.26 MLv

INFO:EventFetcher:52 Trace(s) in Stream:
FR.CHA2.00.EHE | 2023-08-17T10:28:17.190000Z - 2023-08-17T10:29:17.000000Z | 100.0 Hz, 5982 samples
FR.CHA2.00.EHN | 2023-08-17T10:28:17.190000Z - 2023-08-17T10:29:17.000000Z | 100.0 Hz, 5982 samples
FR.CHA2.00.EHZ | 2023-08-17T10:28:17.190000Z - 2023-08-17T10:29:17.000000Z | 100.0 Hz, 5982 samples
...
MT.GUI.00.EHE  | 2023-08-17T10:28:17.190000Z - 2023-08-17T10:29:17.000000Z | 200.0 Hz, 11963 samples
MT.GUI.00.EHN  | 2023-08-17T10:28:17.190000Z - 2023-08-17T10:29:17.000000Z | 200.0 Hz, 11963 samples
MT.GUI.00.EHZ  | 2023-08-17T10:28:17.190000Z - 2023-08-17T10:29:17.000000Z | 200.0 Hz, 11963 samples
```