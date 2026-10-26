# Time series CV (2026-10-09 00:07)

12 sequential expanding folds (cutoffs 2016-03-20 to 2017-02-13, every 30 days), 15-day look-ahead validation; 12 configs × 6 step counts. Metric: hour-weighted mean absolute log error (0.10 ≈ 10% typical error). `sparse_*` = against the thinned readings (used for selection); `truth_mean` = against full-data daily means; `level_mean` = against the 31-day centred mean of the full data (the true smooth level).

## Top 10

|   rough |   spacing |   step |   sparse_mean |   sparse_sd |   sparse_PM2.5 |   sparse_PM10 |   truth_mean |   level_mean |
|--------:|----------:|-------:|--------------:|------------:|---------------:|--------------:|-------------:|-------------:|
|  100000 |        30 |    100 |        0.8178 |      0.196  |         0.9192 |        0.7163 |       0.6828 |       0.2506 |
|  100000 |        15 |    100 |        0.8181 |      0.1971 |         0.9195 |        0.7166 |       0.6831 |       0.2504 |
|  100000 |        15 |    200 |        0.8208 |      0.1992 |         0.923  |        0.7185 |       0.6858 |       0.2581 |
|  100000 |        60 |    100 |        0.8208 |      0.1988 |         0.9214 |        0.7203 |       0.685  |       0.2607 |
|  100000 |        30 |    200 |        0.8212 |      0.1982 |         0.9232 |        0.7192 |       0.6855 |       0.258  |
|  100000 |        30 |    300 |        0.8222 |      0.198  |         0.9244 |        0.72   |       0.6873 |       0.2624 |
|  100000 |        15 |    300 |        0.8223 |      0.2002 |         0.925  |        0.7196 |       0.687  |       0.2629 |
|  100000 |        60 |    200 |        0.8224 |      0.2018 |         0.924  |        0.7207 |       0.6873 |       0.2642 |
|  100000 |        30 |    500 |        0.824  |      0.1982 |         0.927  |        0.7209 |       0.6896 |       0.2695 |
|  100000 |        60 |    300 |        0.8252 |      0.2023 |         0.9285 |        0.7219 |       0.6902 |       0.2709 |

## Baseline: trailing 30-day mean of the sparse data per station

| kind   |   PM2.5 |   PM10 |
|:-------|--------:|-------:|
| level  |  0.3549 | 0.2725 |
| sparse |  0.9313 | 0.7291 |
| truth  |  0.791  | 0.6067 |
