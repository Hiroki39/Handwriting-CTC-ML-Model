mkdir -p log
while true; do
    if [[ -z `ps xf | grep "python3 -u model_v4_wei.py" | grep -v grep` ]]; then
        cur_model=`ls output -t | head -1`
        log=log/nohup.out_$(date +%Y%m%d%H%M%S)
        echo "train"
        python3 -u model_v4_wei.py ${cur_model} &> ${log} & wait
    fi
    sleep 60
done




