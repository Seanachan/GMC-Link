import glob

def clean_file(p):
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
    if len(clean) != len(lines):
        with open(p, 'w', encoding='utf-8') as f:
            f.writelines(clean)
        return True
    return False

paths = glob.glob('/home/seanachan/TempRMOT/exps/default_rk/results_epoch0/*/*/*.txt')
cleaned = sum(1 for p in paths if clean_file(p))
print(f"Cleaned {cleaned} text prediction logs out of {len(paths)}!")
