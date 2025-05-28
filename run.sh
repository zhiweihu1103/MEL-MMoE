export GPU=1

export CONFIG=./config/wikimel.yaml
export LOG=./logs/wikimel_baseline.logs

#export CONFIG=./config/wikidiverse.yaml
#export LOG=./logs/wikidiverse_baseline.logs

#export CONFIG=./config/richpediamel.yaml
#export LOG=./logs/richpediamel_baseline.logs

CUDA_VISIBLE_DEVICES=$GPU nohup python -u ./main.py --config $CONFIG \
    > $LOG 2>&1 &