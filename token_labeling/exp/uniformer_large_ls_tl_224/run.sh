python /code/SSAAFormer/token_labeling/main.py \
  --model uniformer_large_ls \
  --batch-size 32 \
  --apex-amp \
  --lr 1.2e-3 \
  --img-size 224 \
  --drop-path 0.4 \
  --model-ema \
  --train-csv /code/SSAAFormer/dataset/train.csv \
  --test-csv /code/SSAAFormer/dataset/test.csv \
  --output /code/SSAAFormer/output \
  --csv-res /code/SSAAFormer/output \
  2>&1 | tee /code/SSAAFormer/output/log.txt
