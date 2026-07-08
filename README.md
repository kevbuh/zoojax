![ZOOJAX](zoojax.png)

single-file implementations of RL algorithms so i can actually do research

| Algorithm | Code | W&B |
|---|---|---|
| Actor-Critic | [src/torch_actor_critic.py](src/torch_actor_critic.py) | |
| APS | [src/aps.py](src/aps.py), [src/aps_minidoom.py](src/aps_minidoom.py) | |
| CRL | [src/crl.py](src/crl.py), [src/scaling_crl.py](src/scaling_crl.py) | |
| DQN | [src/torch_cleanrl_dqn_atari.py](src/torch_cleanrl_dqn_atari.py) | |
| PPO | [src/ppo.py](src/ppo.py), [src/torch_cleanrl_ppo_atari.py](src/torch_cleanrl_ppo_atari.py) | [Brax](https://wandb.ai/kevinbuhler/zoojax/runs/14neyqzq) |
| Rainbow | [src/torch_cleanrl_rainbow_atari.py](src/torch_cleanrl_rainbow_atari.py) | |
| DIAYN | [src/diayn.py](src/diayn.py) | |
| E3B | [src/e3b.py](src/e3b.py), [src/e3b_minidoom.py](src/e3b_minidoom.py) | |
| ICM | [src/icm.py](src/icm.py), [src/icm_vizdoom.py](src/icm_vizdoom.py) | |
| Predictron | [src/torch_predictron.py](src/torch_predictron.py) | |
| Recurrent Visual Attention | [src/torch_recurrent_visual_attention.py](src/torch_recurrent_visual_attention.py) | |
| REINFORCE | [src/torch_reinforce.py](src/torch_reinforce.py) | |
| RND | [src/rnd.py](src/rnd.py), [src/rnd_atari.py](src/rnd_atari.py) | [Brax](https://wandb.ai/kevinbuhler/rnd-ppo-brax/runs/nzwknnrw), [Atari](https://wandb.ai/kevinbuhler/rnd-atari-jax/reports/ZOOJAX-Atari-RND--VmlldzoxNzQzOTYzOA) |
| SMiRL | [src/smirl.py](src/smirl.py) | |
| SMM | [src/smm.py](src/smm.py) | |
| Squeeze-Excitation | [src/squeeze_excitation.py](src/squeeze_excitation.py) | |
| VAE | [src/torch_vae.py](src/torch_vae.py) | |
| VQ-VAE | [src/torch_vq_vae.py](src/torch_vq_vae.py) | |
| EBD | [src/ebd.py](src/ebd.py) | |


similar projects
- [purejaxrl](https://github.com/luchris429/purejaxrl)
- [cleanrl](https://github.com/vwxyzjn/cleanrl)
- [acme](https://github.com/google-deepmind/acme/tree/master/acme/agents/jax)
- [stable-baselines-3](https://github.com/DLR-RM/stable-baselines3.git)
- [tianshou](https://github.com/thu-ml/tianshou/tree/master/tianshou/algorithm/modelfree)
- [unifloral](https://github.com/emptyjackson/unifloral)