"""Smooth 'prevailing' levels from sparse, gappy panels: code shared by the blog post, its CV runs and the demo.

- data: UCI Beijing download, daily means, the blog's uniform thinning and the demo's label-dependent thinning
- model: the joint two-target network (fit / predict)
- baselines: the 31-day truth and the ±15-day mean estimator
- cv, interp_cv, evaluate: the blog's tuning and final-evaluation runs (python -m sparse_panel.<name>)
"""
