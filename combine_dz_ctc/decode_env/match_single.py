import os
import numpy as np

hz2id_map = dict()
for idx, line in enumerate("hz_utf.txt"):
    hz2id_map[line.strip()] = idx


files = ["output.txt"]

out_file = open("output.txt.tmp", "w", encoding="utf8")

right = 0
all = 0

res = []
idx_list = []
for file in files:
    for line in open(file, encoding="utf8"):
        ttt = line.strip().split()
        idx = int(ttt[0])
        res.append([idx, line])
        idx_list.append(idx)
idx_list = set(idx_list)

res = sorted(res, key=lambda x: x[0])
for idx, line in res:
    if 103758 <= idx:
        out_file.write(line)
        all += 1

ans = []
for i in range(103758, 203332):
    if i not in idx_list:
        ans.append(i)

print(ans)

print(len(ans))
print(all)
            
os.system("mv output.txt.tmp output.txt")





