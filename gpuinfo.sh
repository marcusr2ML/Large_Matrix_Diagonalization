#! /bin/sh
sinfo -p secondary,secondary-eth -o "%.9N %.4c %.10m %.16G %.16P %.8T %.12l %.11L %20f" -N | head -1; sinfo -p secondary,secondary-eth -o "%.9N %.4c %.10m %.16G %.16P %.8T %.12l; %.11L %30f" -N | grep gpu
