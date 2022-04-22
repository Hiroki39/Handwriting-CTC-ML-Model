#!/bin/bash
export LD_LIBRARY_PATH=/mnt/AM4_disk1/chenxu/run_env/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/AM4_disk1/raopenghao/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/home/disk1/liangmingxin/anaconda3/bin:$PATH    

# some variables
decode_dir="decode_env"
result_dir="decode_results"
model_dir="output"

keyref_dir="/mnt/AM4_disk1/raopenghao/wtsmp_strokes_test"
wer_bin="/mnt/AM4_disk2/gaoyunze/decode_handwriting/wer_tools/wer_client"
wer_conf="/mnt/AM4_disk2/gaoyunze/decode_handwriting/wer_tools/wer_client.conf"

mkdir -p $decode_dir/$result_dir
cd $decode_dir

# create symbolic link to the parent directory
find ../ -maxdepth 1 | tail -n +2 | xargs -I{} ln -snf {}
ln -snf /home/slurm/data/Speech_Data/

# remove the symbolic link to itself
rm $decode_dir

# some useful scripts
cp /mnt/AM4_disk1/chenxu/scripts/check_n_result.sh .
cp /mnt/AM4_disk1/raopenghao/handwriting_paddle/evaluate_ywf/feat.lst .
cp /mnt/AM4_disk1/raopenghao/handwriting_paddle/evaluate_ywf/label.lst .
cp /mnt/AM4_disk1/raopenghao/handwriting_paddle/evaluate_ywf/hz_utf.txt .
cp /mnt/AM4_disk1/raopenghao/handwriting_paddle/evaluate_ywf/decode_env/match_single.py .


set -x
while true; do
    # indicate whether there's new model
    new_model_flag=0
    model_lst=`ls -t ${model_dir}` # ls -t: sort by time
    for model in ${model_lst}; do
        echo ${model}
        MODEL=model_file_${model}
        # if model_tested.lst exists and the length of the command `grep $MODEL 
        # model_tested.lst` is not zero
        # equivalent to "if the model is tested"
        if [[ -f model_tested.lst ]] && [[ ! -z `grep $MODEL model_tested.lst` ]]; then
            continue
        fi
        new_model_flag=1
        # evaluate the model
        CUDA_VISIBLE_DEVICES=5 python -u mulpro_decode_model_evaluate.py -m=${model_dir}/${model} &> score.log
        python match_single.py

        # store text recognition result to a temporary file
        awk '{print $3}' output.txt > ${MODEL}.tmp.raw
        # concatenate the answer key and the model output
        paste ${keyref_dir}/part2_all.key ${MODEL}.tmp.raw > ${MODEL}.out

        # split the answer key and the model output by type
        grep -Fwf ${keyref_dir}/part2_single.key ${MODEL}.out > ${MODEL}.out0
        grep -Fwf ${keyref_dir}/part2_sfree.key ${MODEL}.out > ${MODEL}.out1
        grep -Fwf ${keyref_dir}/part2_ffree.key ${MODEL}.out > ${MODEL}.out2
        grep -Fwf ${keyref_dir}/part2_multi.key ${MODEL}.out > ${MODEL}.out3

        # calculate the error rate by part
        ${wer_bin} ${keyref_dir}/part2_single.ref ${MODEL}.out0 ${MODEL}.rlt0 ${wer_conf}
        ${wer_bin} ${keyref_dir}/part2_sfree.ref ${MODEL}.out1 ${MODEL}.rlt1 ${wer_conf}
        ${wer_bin} ${keyref_dir}/part2_ffree.ref ${MODEL}.out2 ${MODEL}.rlt2 ${wer_conf}
        ${wer_bin} ${keyref_dir}/part2_multi.ref ${MODEL}.out3 ${MODEL}.rlt3 ${wer_conf}

        # copy files to result directory and remove current files
        cp ${MODEL}.* ${result_dir}
        rm ${MODEL}.*

        # add current model to model_tested file
        echo ${MODEL} >> model_tested.lst 
    done
    # if there's new model, check for the new model
    if [[ new_model_flag -eq 1 ]]; then
        sh check_n_result.sh ${result_dir} 4
    else
        sleep 600
    fi
    
done

