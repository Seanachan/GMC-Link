import os
base = "/home/seanachan/TempRMOT/exps/default_rk/results_epoch0"
out = []
for d in ['0005', '0011', '0013']:
    exprs = os.listdir(os.path.join(base, d))
    for expr in exprs:
        if os.path.isdir(os.path.join(base, d, expr)):
            out.append(f"{d}+{expr}")

with open("/home/seanachan/GMC-Link/seqmap-all.txt", "w") as f:
    f.write("\n".join(out) + "\n")
