


MYLANGUAGES=(arabic english french german hungarian portuguese russian spanish)

for MYLANGUAGE in ${MYLANGUAGES[@]} ; do
    python baseline/run_baseline_gen.py -d 2019 -l $MYLANGUAGE -m all -t 1000 --progress-bar

    #for EXPEVERSION in {0..9} ; do
    #    python test_ret_base.py -b 512 -v 5000 -t 10000 --max_epochs 50 --gpus -1 -d 2019 -l $MYLANGUAGE -V $EXPEVERSION
    #done
done