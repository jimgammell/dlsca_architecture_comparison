python main.py --dataset ASCAD-V1-Fixed-DS100 --htune-classifier convmixer --seed 0 --device cuda:0 --num-wandb-agents 4 --save-dir convmixer_trial0 &> trial0.txt &
python main.py --dataset ASCAD-V1-Fixed-DS100 --htune-classifier convmixer --seed 0 --device cuda:1 --num-wandb-agents 4 --save-dir convmixer_trial1 &> trial1.txt &
python main.py --dataset ASCAD-V1-Fixed-DS100 --htune-classifier convmixer --seed 0 --device cuda:2 --num-wandb-agents 4 --save-dir convmixer_trial2 &> trial2.txt &
