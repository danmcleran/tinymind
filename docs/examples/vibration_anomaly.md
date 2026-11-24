---
title: On-device Vibration Anomaly Detection
parent: Examples
nav_order: 64
layout: default
---

# On-device Vibration Anomaly Detection

This is a sensor-node pipeline that learns what normal vibration looks like on the device itself, then raises alarms when it changes. Every step of it uses integer arithmetic only. The machine is synthetic: its speed wanders between 28 and 32 Hz and its load moves the level by ±6 dB. Three faults (bearing, imbalance and looseness) each grow from nothing to full severity.

## How it works

- `QLogMelSpectrogram<256, 32>` turns each 256-sample frame (3,200 Hz, hop 128) into 32 mel bands. The frame's mean band is subtracted, so the features describe the spectrum's shape rather than its level, and they are mapped onto int8.
- `QKMeansAnomalyTrainer<32, 8>` runs online k-means for 120 s, one frame at a time, and measures how much each feature varies around its centroid.
- The first `calibrate()` freezes the centroids and weights each feature inversely to that variance. Over the next 60 s, it measures each cluster's mean squared radius.
- The score is the weighted distance to the nearest centroid, divided by that cluster's radius, in Q8. The alarm threshold is 1.25 times the largest score seen over a further 60 s of normal running.
- Two comparison detectors are scored on the same features: the same trainer with the weights off (a plain Euclidean distance), and a diagonal Gaussian fitted on the host to the learning frames.

## Build and run

```bash
cd examples/vibration_anomaly
make release
make run
make plot      # needs matplotlib in a venv
```

## Output

![Vibration anomaly detection]({{ site.baseurl }}/assets/plots/vibration_anomaly.png)

The k-means learned on the device matches the Gaussian fitted on the host. It catches the bearing defect at 8% of full severity and looseness at 16%. It raises alarms on 69% of the imbalance fault's frames, with 0.17% false alarms in normal running.

The Euclidean k-means never raises a sustained alarm on the imbalance fault. That fault changes a few low-frequency bands, and an unweighted distance over 32 features loses the change among bands that vary a lot in normal running. The per-feature weights recover it.

Over machine seeds 11 to 15, the weighted k-means averages alarm rates of 92.0% (bearing), 65.2% (imbalance) and 84.9% (looseness), with 0.10 to 0.17% false alarms.

For a Cortex-M0+ at `-Os`, the detector is 1,274 bytes of code and the trainer 1,256 bytes of RAM, with at most 72 bytes of stack. With the log-mel front end, the node needs about 3 KB of RAM. On the NUCLEO-L031K6 at 32 MHz a k-means score takes 0.74 ms and a learning step 0.96 ms; with the 5.8 ms log-mel frame, scoring takes 8% of the CPU at 3.2 kHz ([`benchmarks/tier1_onboard`](https://github.com/danmcleran/tinymind/tree/master/benchmarks/tier1_onboard)).

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/vibration_anomaly)
