#!/bin/bash
DATE=`date '+%Y-%m-%d-%H-%M-%S'`
MAIN='src/selene/run.py'
(time python $MAIN) |& tee outputs/src/ae/logs/exp_$DATE.log
