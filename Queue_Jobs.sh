#!/bin/bash

nitv=$1

for ((i=0;i<$nitv;i++)); do
	echo "sbatch Job_fac.sbash interval${i}.txt double"
	output=$(sbatch Job_fac.sbash interval${i}.txt double)
	read -ra ARR <<< "$output"
	jobid=${ARR[-1]}
	echo $jobid
	echo "sbatch -d afterok:${jobid} --gres=gpu:V100:2 Job_solve.sbash fac-interval${i} double"
	sbatch -d afterok:${jobid} --gres=gpu:V100:2 Job_solve.sbash fac-interval${i} double
done

