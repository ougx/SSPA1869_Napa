@echo off
echo 860       50 NOPRINT > GHB.ghb
echo 860 >> GHB.ghb
..\..\bin\ArrayMath -d 440 2 -rn --add  uwest.ghb - 1.0  --add  uwest.dh - 0.5 >>  GHB.ghb
..\..\bin\ArrayMath -d 70  2 -rn --add valley.ghb - 1.0  --add valley.dh - 0.6 >>  GHB.ghb
..\..\bin\ArrayMath -d 350 2 -rn --add  ueast.ghb - 1.0  --add  ueast.dh - 0.5 >>  GHB.ghb
FOR /L %%x IN (2, 1, 474) DO echo -%%x >> GHB.ghb
 