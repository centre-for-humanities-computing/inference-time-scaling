export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

models=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct")
methods=("kl-divergence")
datasets=("math500" "gsm8k-en" "gsm8k-da")

# save output to a log file
exec > output.log 2>&1

# Forward Pass -------------------------------------------------------------------------
# for model in "${models[@]}"; do
#     for dataset in "${datasets[@]}"; do
#         uv run python main.py --method "forward-pass" --model_id "$model" --dataset "$dataset" --n_proposal 4 # --overwrite True
#     done
# done


# Inferene time Scaling ----------------------------------------------------------------
for dataset in "${datasets[@]}"; do
   for model in "${models[@]}"; do
       for method in "${methods[@]}"; do
           for proposal in 2 4 8; do
               uv run python main.py --method "$method" --n_proposal "$proposal" --model_id "$model" --dataset "$dataset"
           done
       done
   done
done

# screen -S its  # create a new screen session named "its"
# Press Ctrl + A, then D (mnemonic: Detach).
# screen -ls
# screen -r session_name  # reattach, or use the PID (e.g., `screen -r 12345`)