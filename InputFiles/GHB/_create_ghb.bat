@echo off
echo 420       50 NOPRINT > GHB.ghb
echo 420 >> GHB.ghb
..\..\bin\ArrayMath -d 70  2 -rn --add valley.ghb - 1.0  --add valley.dh - 1.0 >>  GHB.ghb
..\..\bin\ArrayMath -d 350 2 -rn --add upland.ghb - 1.0  --add upland.dh - 1.0 >>  GHB.ghb
FOR /L %%x IN (2, 1, 474) DO echo -%%x >> GHB.ghb
 