---
title: Sparse Panel Demo
emoji: 🌫️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: Recovering smooth levels from sparse, gappy air-quality data
datasets:
  - prteek/sparse-panel-demo
---

Interactive companion to the blog post [PM2.5, PM10 ~ station + time](https://prteek.github.io/regression_room/posts/beijing_air_quality/). Pick how many readings are kept and a group of stations, and compare a joint network, single-target networks and a ±15-day mean against the true 31-day level.

Code: [github.com/prteek/sparse-panel-demo](https://github.com/prteek/sparse-panel-demo) · Data: [prteek/sparse-panel-demo](https://huggingface.co/datasets/prteek/sparse-panel-demo), derived from the UCI Beijing Multi-Site Air Quality dataset (CC BY 4.0).
