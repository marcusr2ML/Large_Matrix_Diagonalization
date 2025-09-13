#!/bin/bash

srun --partition=physics,secondary,secondary-eth --time=00:30:00 --nodes=1 --exclusive --mincpus=$1 --pty /bin/bash
