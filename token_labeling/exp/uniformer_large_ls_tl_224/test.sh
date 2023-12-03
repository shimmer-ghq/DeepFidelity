python /code/SSAAFormer/token_labeling/validate.py \
  --model uniformer_large_ls \
  --checkpoint /code/SSAAFormer/checkpoint/model_best.pth.tar \
  --no-test-pool \
  --img-size 224 \
  --batch-size 128 \
  --use-ema \
  --csv-res /code/SSAAFormer/output/result_2class.csv
  2>&1 | tee /code/SSAAFormer/output/log.txt
