for i in 1 3
do
python main_PPO_TUC.py -epochs 1000 -runs 1 \
    -L_num 200 -alpha 1e-3 -gamma 0.99 \
    -clip_epsilon 0.2 -question ${i} -ppo_epochs 1 \
    -batch_size 1 -gae_lambda 0.95  -seed 40 \
    -delta 0.5 -rho 0.01 \
    -beta 1.0 -tau 0.5 -zeta 0.01
done
