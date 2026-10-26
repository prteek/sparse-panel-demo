---
license: cc-by-4.0
pretty_name: Sparse panel demo (Beijing air quality)
tags:
  - time-series
  - air-quality
  - missing-data
  - smoothing
size_categories:
  - 100K<n<1M
configs:
  - config_name: predictions
    data_files: predictions.parquet
  - config_name: readings
    data_files: readings.parquet
  - config_name: truth
    data_files: truth.parquet
  - config_name: gaps
    data_files: gaps.parquet
  - config_name: labels
    data_files: labels.parquet
  - config_name: metrics
    data_files: metrics.parquet
---

# Sparse panel demo (Beijing air quality)

Precomputed scenarios behind the [sparse-panel-demo Space](https://huggingface.co/spaces/prteek/sparse-panel-demo) (code: [GitHub](https://github.com/prteek/sparse-panel-demo)), which accompanies the paper [PM2.5, PM10 ~ Station + Time, Analysis](https://www.alphaxiv.org/abs/2610.pm25-pm10-joint-smoothing) on alphaXiv.

Daily PM2.5 and PM10 at 12 Beijing monitoring stations (March 2013 – February 2017) are thinned to a few hourly readings a day, with gaps, and three estimators try to recover the slow underlying level:

- **joint**: a small network modelling both pollutants together (area embedding + station offset, Gaussian time bumps, curvature penalty);
- **single-target**: the same network fitted to one pollutant at a time;
- **±15-day mean**: hour-weighted mean of the readings within 15 days, falling back to the station's area where it has none.

The benchmark ("truth") is the centred 31-day moving average of the full daily means, on the log scale.

## Scenarios

One global setting, `lam`: the Poisson mean number of hourly readings kept per station-day and pollutant (0.5, 1.2, 3.0). Each station also has a fixed **label** that degrades it relative to `lam`:

| label | stations | treatment |
|---|---|---|
| regular | 4 | Poisson(`lam`) hours, two 20–60 day gaps per pollutant |
| sparse | 2 | Poisson(0.4 × `lam`) hours, same gaps |
| very sparse | 2 | as regular, then 15% of observed days kept |
| long gap | 2 | as regular, but the gaps are 90–120 days |
| dropped series | 2 | one pollutant removed entirely (`dropped` column) |

Random draws are fixed per cell across scenarios, so a higher `lam` keeps a superset of a lower one's hours and gaps sit in the same place.

## Files

| file | rows | columns |
|---|---|---|
| `labels.parquet` | 12 | station, label, area, dropped |
| `truth.parquet` | 35,064 | station, pollutant, date, value (31-day MA, µg/m³) |
| `readings.parquet` | 46,927 | station, date, pollutant, value (daily mean of kept hours), n (hours kept), lam |
| `gaps.parquet` | 2,454 | station, pollutant, date (days inside a gap block; same for every `lam`) |
| `predictions.parquet` | 315,576 | station, date, pollutant, pred, lam, estimator |
| `metrics.parquet` | 315 | lam, label, pollutant, estimator, day_type, days, error, bias |

`error` is the mean absolute log error against the 31-day MA (0.15 ≈ 15%), `bias` the mean log error. `day_type` is `all`, `observed` (a reading that day), `unobserved` (no reading, outside the gaps) or `gap`. For the `dropped series` label, `pollutant` is `dropped` and only the dropped series is scored. `meta.json` records the network settings (from look-ahead CV in the paper: curvature penalty 1e5, 30-day bump spacing, 100 steps).

Reproduce with `uv run python precompute.py` in the GitHub repo (~30 s on a laptop CPU).

## Source and license

Derived from the **Beijing Multi-Site Air Quality** dataset (Song Chen, 2017), UCI Machine Learning Repository, [doi:10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), licensed CC BY 4.0. This derived dataset is released under the same license. Changes: daily aggregation, thinning to sparse readings, gap removal, and model predictions as described above.
