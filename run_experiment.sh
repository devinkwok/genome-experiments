DATE=`date '+%Y-%m-%d-%H-%M-%S'`
MAIN='src/ae/experiment.py'
(time python $MAIN) |& tee outputs/src/ae/autoencoder/exp_$DATE.log
