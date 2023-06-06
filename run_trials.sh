python main.py --dataset ASCAD-V1-Fixed-DS100 --train-classifier convmixer --seed 0 --device cuda:0 --lr 2e-4 --save-dir convmixer_trial0 &> trial0.txt &
python main.py --dataset ASCAD-V1-Fixed-DS100 --train-classifier convmixer --seed 0 --device cuda:1 --lr 4e-4 --save-dir convmixer_trial1 &> trial1.txt &
python main.py --dataset ASCAD-V1-Fixed-DS100 --train-classifier convmixer --seed 0 --device cuda:2 --lr 8e-4 --save-dir convmixer_trial2 &> trial2.txt &
