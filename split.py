import splitfolders

splitfolders.ratio("datasets/Liver", output="liver",
    seed=7965, ratio=(.85, .1, .05), group_prefix=None, move=False)