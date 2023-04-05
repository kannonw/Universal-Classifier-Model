import splitfolders

splitfolders.ratio("datasets", output="data",
    seed=7965, ratio=(.85, .1, .05), group_prefix=None, move=False)