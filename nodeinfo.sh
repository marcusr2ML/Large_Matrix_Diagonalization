#!/bin/sh
sinfo -p physics,secondary,secondary-eth -o "%.9N %.4c %.10m %.16G %.16P %.8T %.12l %.11L %20f" -N | grep $1
