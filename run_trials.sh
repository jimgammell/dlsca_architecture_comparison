python main.py --dataset ASCAD-V1-Fixed --htune-classifier vgg --seed 0 --device cuda:0 --num-wandb-agents 2 --save-dir vgg_trial0 &> trial0.txt &
python main.py --dataset ASCAD-V1-Fixed --htune-classifier vgg --seed 0 --device cuda:1 --num-wandb-agents 2 --save-dir vgg_trial1 &> trial1.txt &
python main.py --dataset ASCAD-V1-Fixed --htune-classifier vgg --seed 0 --device cuda:2 --num-wandb-agents 2 --save-dir vgg_trial2 &> trial2.txt &
