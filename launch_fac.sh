#! /bin/sh

EXECDIR=$PWD

EXEC="factorize"
N=400
D_O=0.06
RE_S="re_half_N=400_r0=16"
IM_S="im_half_N=400_r0=16"
ITV=$1
NT=-1


cp ../$RE_S $EXECDIR/
cp ../$IM_S $EXECDIR/
cp ../$ITV $EXECDIR/intervals.txt
./$EXEC $N $D_O $RE_S $IM_S intervals.txt $NT
