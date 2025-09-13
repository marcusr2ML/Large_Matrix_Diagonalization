#!/bin/bash

# Usage: ./copy_results <result dir> <interval#> <prefix>
# Example: ./copy_results ~/physics/results_fp64_0.06_400 0 E_

printUsage() {
	echo 'Usage: ./copy_results <result directory name> <# of intervals> <prefix to copy>'
	echo 'Example: "./copy_results results_n400 3 E_" would create directories:'
	echo '         ~/physics/results_n400/0, ~/physics/results_n400/1, ~/physics/results_n400/2,'
	echo '         and copy the eigenvalues (files starting with "E_") to the corresponding result directories.'
	exit
}

re='^[0-9]+$'
if [[ $# != 3 ]] || ! [[ $2 =~ $re ]]; then
	printUsage
fi

RES_DNAME=$1
N_ITV=$2
PREFIX=$3

for (( i=0; i<$N_ITV; i++ ));
do
	echo "mkdir -p ~/physics/$RES_DNAME/$i"
	mkdir -p ~/physics/$RES_DNAME/$i
done

RES_DIR=~/physics/$RES_DNAME

for (( i=0; i<$N_ITV; i++ ));
do
	echo "cp ./fac-interval$i/$PREFIX* $RES_DIR/$i/"
	cp ./fac-interval$i/$PREFIX* $RES_DIR/$i/
done
