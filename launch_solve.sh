#! /bin/sh

EXECDIR=$PWD

EXEC="solve"
N=400
ITV="intervals.txt"
K=550
XTOL=3000

./$EXEC $N $ITV $K $XTOL
