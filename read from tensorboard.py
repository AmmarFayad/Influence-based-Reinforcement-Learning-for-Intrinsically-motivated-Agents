from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
ea = event_accumulator.EventAccumulator(r"C:\Users\lenovo\Desktop\RL Research\our code\pytorch-soft-actor-critic-master\runs\2020-06-01_03-38-26_SAC_HopperBulletEnv-v0_Gaussian_",
size_guidance={
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 1,
})
ea.Reload()
ea.Tags()
pd.DataFrame(ea.Scalars('avg_reward/test')).to_csv('return2.csv')
pd.DataFrame(ea.Scalars('entropy_temprature/alpha')).to_csv('loss2.csv')
#print(ea.Scalars('entropy_temprature/alpha'))