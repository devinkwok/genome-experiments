DATE=`date '+%Y-%m-%d-%H-%M-%S'`
MAIN='src/ae/autoencoder.py'
(time python $MAIN) |& tee outputs/src/ae/autoencoder/exp_$DATE.log
