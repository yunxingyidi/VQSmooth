python ppl_eval.py  --model "/home/zhangtairan/models/llama-2-7b-hf/" \
 --act_scales_path act_scales/llama-2-7b.pt  \
 --eval-only --use-vq --group-quantize  \
 --alpha 1 \
 --assignment-chunk-size 32  \
 --dataset wikitext2 \
 --device npu:0 \
 --vq-devices npu:0 \
 --profile --profile-dir ./prof_result