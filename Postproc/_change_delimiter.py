import pandas as pd
import numpy as np
from io import StringIO

files = [
    "outlet_brownsvalley.csv",
    "outlet_napa.csv",
    "times _sp.csv",
    "gage_napa.csv",
    "gage_helena.csv",
]

for f in files:
    df = pd.read_csv(f)
    with open(f.rstrip("csv") + "dat", "w") as fw:
        fw.write(df.to_string(index=False))

f = "times.csv"
df = pd.read_csv(f)
with open(f.rstrip("csv") + "dat", "w") as fw:
    fw.write(pd.concat([df,]*5).sort_values("time").to_string(index=False))

