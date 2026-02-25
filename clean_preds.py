"""
Module for cleaning duplicate prediction logs from TrackEval output directories.
"""
import glob

paths = glob.glob('/home/seanachan/RMOT/exps/default/results_epoch99/*/*/predict.txt')
for p in paths:
    with open(p, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    seen = set()
    clean = []
    for l in lines:
        parts = l.strip().split(',')
        if len(parts) >= 2:
            key = (parts[0], parts[1])
            if key not in seen:
                seen.add(key)
                clean.append(l)
    with open(p, 'w', encoding='utf-8') as f:
        f.writelines(clean)
print(f"Cleaned {len(paths)} prediction logs!")
