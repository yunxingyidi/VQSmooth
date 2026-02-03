# VQSmooth
Combine VQ and Smoothquant to reduce the size of LLM

python ppl_eval.py  --model "/root/autodl-tmp/model/llama-2-7b-hf/"  --act_scales_path act_scales/llama-2-7b.pt  --smooth  --group-quantize  --alpha 1 --use-vq  --assignment-chunk-size 32  --dataset wikitext2

python act_scales/generate_scales.py \
    --model  "/home/zhangtr/WorkSpace/models/llama-2-7b-hf/"\
    --output-path "act_scales/llama-2-7b.pt" \
    --dataset-path "/home/zhangtr/WorkSpace/VQSmooth/datasets/train-00000-of-00001.parquet"

python accuracy_eval.py  --model "/home/zhangtr/WorkSpace/models/llama-2-7b-hf/"  --act_scales_path act_scales/llama-2-7b.pt  --alpha 1 --use-vq  --assignment-chunk-size 32  --dataset wikitext2
