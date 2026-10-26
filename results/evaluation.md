# Final evaluation (2026-10-09 22:27)

Settings from look-ahead time series CV (15 days): rough 100000, spacing 30 days, 100 steps. Error = mean absolute log error against the truth (0.10 ≈ 10%).

| truth   | day_type       | pollutant   |   joint |   single-target |   ±15-day mean |   days |
|:--------|:---------------|:------------|--------:|----------------:|---------------:|-------:|
| daily   | gap            | PM10        |   0.553 |           0.549 |          0.7   |    956 |
| daily   | gap            | PM2.5       |   0.692 |           0.688 |          0.805 |   1003 |
| daily   | observed       | PM10        |   0.572 |           0.569 |          0.558 |  11744 |
| daily   | observed       | PM2.5       |   0.715 |           0.712 |          0.701 |  11499 |
| daily   | other-observed | PM10        |   0.574 |           0.572 |          0.599 |   3788 |
| daily   | other-observed | PM2.5       |   0.718 |           0.716 |          0.745 |   4019 |
| daily   | unobserved     | PM10        |   0.577 |           0.575 |          0.588 |   4788 |
| daily   | unobserved     | PM2.5       |   0.72  |           0.718 |          0.739 |   4965 |
| level   | gap            | PM10        |   0.144 |           0.135 |          0.384 |    972 |
| level   | gap            | PM2.5       |   0.161 |           0.148 |          0.401 |   1006 |
| level   | observed       | PM10        |   0.143 |           0.137 |          0.143 |  11744 |
| level   | observed       | PM2.5       |   0.168 |           0.159 |          0.166 |  11499 |
| level   | other-observed | PM10        |   0.143 |           0.137 |          0.176 |   3788 |
| level   | other-observed | PM2.5       |   0.164 |           0.155 |          0.193 |   4033 |
| level   | unobserved     | PM10        |   0.142 |           0.137 |          0.148 |   4816 |
| level   | unobserved     | PM2.5       |   0.164 |           0.156 |          0.169 |   5027 |

Raw sparse readings as an estimate of the level on observed days: PM2.5 0.836, PM10 0.696
