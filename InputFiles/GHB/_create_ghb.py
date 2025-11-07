import pandas as pd
import numpy as np

nlay = 10
gcell = pd.read_csv("ghb_cells.dat", sep="\\s+")
gcell["Cond"] = 10000
gcell["Head"] = 5

nghb = len(gcell) * nlay
with open("GHB.ghb", "w") as f:
    f.write(f"{nghb:<10}50 NOPRINT\n")
    for i in range(1):
        f.write(f"{nghb:<10}\n")
        for l in range(nlay): 
            gcell["Layer"] = l + 1
            f.write(gcell["Layer row column_ Head Cond".split()].to_string(index=False, header=False)+"\n")
    for i in range(487):
        f.write("-1\n")

