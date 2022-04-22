#!/bin/sh
cd $1               # enter the input directory
num=$2              # num is the number (the second input)
last=$(( num - 1 )) # last evaluated model

# if result.txt/result.wer doesn't exist, generate one
if [ ! -f ../result.txt ]; then
    touch ../result.txt
fi

if [ ! -f ../result.wer ]; then
    touch ../result.wer
fi


for FILE in `ls -rt *.rlt${last}`; do
    KEY=`basename $FILE .rlt${last}` # extract the basename
    # if filename already exists in result.wer, skip this model
    if (( $(grep "${KEY}" ../result.wer | wc -l) )); then
        continue
    fi

    echo ${KEY} >> ../index.txt

    sum=0.0
    echo -e "$KEY:" >> ../result.txt
    for n in `seq 0 ${last}`; do
        # extract the line recording result
        WERn=`tail -6 ${KEY}.rlt${n} | grep '\[WER'`
        # extract the error rate
        wern=`foo=${WERn#*WER: }; bar=${foo%SUB*}; echo ${bar%\%*}`

        echo ${wern} >> ../wer${n}.wer

        sum=`echo $sum + $wern | bc`

        echo -e "\t$WERn" >> ../result.txt
    done
    # get the average error rate    
    sum=`echo $sum / ${num} | bc -l`


    echo -e "\tAverage = $sum%" >> ../result.txt
done
cd .. > /dev/null

if [ -f index.txt ]; then
    cp index.txt result.wer.app
    # aggregate result files into a single file
    for n in `seq 0 ${last}`; do
        paste result.wer.app wer${n}.wer > dump
        cp dump result.wer.app
        rm dump
    done
    rm -f wer*.wer index.txt
    cat result.wer.app >> result.wer
    rm result.wer.app
fi
