import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import  EvalCallback , StopTrainingOnRewardThreshold
#Tworzymy ENV
enviroment_name = 'CartPole-v1'
env = gym.make(enviroment_name)
#Episode = jedna gra

#Ścieżka zapisu logów , pierwsze co robimy
log_path = os.path.join('Training' , 'logs')
env = gym.make(enviroment_name)
env = DummyVecEnv(([lambda: env]))
model = PPO('MlpPolicy' , env , verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000)
PPO_path = os.path.join('Training' , 'Saved Models' , "PPO_model_Cartpole")
model.save(PPO_path)
del model
model = PPO.load(PPO_path , env = env)
evaluate_policy(model , env ,n_eval_episodes=10 , render=True)
episodes = 2
for episodes in range(1 , episodes + 1):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        #określa akcje z przedziału 0 lub 1
        action , _ = model.predict(obs) #tutaj używamy już modelu
        obs , reward , done , info = env.step(action)
        score +=reward
    print('Episode:{} Score:{}'.format(episodes, score))
train_log_path = os.path.join(log_path , "PPO_9")
env.close()
save_path = os.path.join('Training' , 'Saved Models')
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200 , verbose=1)
eval_callback = EvalCallback(env,callback_on_new_best=stop_callback ,
                             eval_freq= 10000 ,
                             best_model_save_path = save_path ,
                             verbose = 1)
model = PPO('MlpPolicy' , env , verbose=1 , tensorboard_log=log_path)
model.learn(total_timesteps=20000,callback=eval_callback)