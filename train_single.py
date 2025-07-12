# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:38:05 2024

@author: user
"""
from antenna.utils import *
config.device = "cpu"

import torch.nn as nn
import numpy as np
import torch
import argparse
from antenna import *

from antenna.models import (
    OldGEN, HFSSNet
)
from antenna.patch import (
    SinglePortSimulator, custom_loss_g, custom_loss_r
)
from antenna.smodels import OldSM
#%% 
###* Basic Config ###
RESULT_PATH, is_connect_run = get_result_path('test')
TEMP = Record("temp", rootdir=RESULT_PATH, load=True)
# sys.excepthook = global_exception_handler

path_pic = RESULT_PATH.joinpath("pic").not_exist_create()
path_checkpoint = RESULT_PATH.joinpath("checkpoint").not_exist_create()


config['Name'] = RESULT_PATH.stem
config['File'] = __file__
config.setWarning()
config.epochs = 1000
config.lr = 0.003
config.checkpoint_save_path = path_checkpoint

config['HFSS.lr'] = 0.001
config['HFSS.min_loss'] = 0.00005
config['HFSS.max_epoch'] = 2000

logger.info(f"The results will be saved in {RESULT_PATH.absolute()} (Continue: {is_connect_run}, CUDA: {torch.cuda.is_available()})")

###* Set Antemma Pattern ###
AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
lower = AntennaPattern(torch.ones((5, 5)), (10, 15, 20, 25))

simulator = SinglePortSimulator(
    record_path = RESULT_PATH,
)
AntennaPattern.register_simulator(simulator)


###* Set Antenna Response ###
AntennaResponse.registerLabels('S11', 'Gain', x = 'n257')
x = AntennaResponse.x()

#? S11 S22 -> high low high
returnloss = AntennaResponse.registerTargetResponse(-1.25, -15, (4, 2, 5, 2, 4), label="S11")
returnloss_upper = AntennaResponse.registerTargetResponse(0, -10, (4, 2, 5, 2, 4), label="returnloss_upper")
returnloss_lower = AntennaResponse.registerTargetResponse(-2.5, -50, (3, 4, 3, 4, 3), label="returnloss_lower")

AntennaResponse.registerLossHook(custom_loss_r, label = "S11")

#? Gain -> low high low
gain = AntennaResponse.registerTargetResponse(-19, 0, (3, 0, 11, 0, 3), label="Gain")
gain_upper = AntennaResponse.registerTargetResponse(-17, 0, (2, 3, 7, 3, 2), label="gain_upper")
gain_lower = AntennaResponse.registerTargetResponse(-22, -3, (4, 2, 5, 2, 4), label="gain_lower")

AntennaResponse.registerLossHook(custom_loss_g, label = "Gain")

with Figure('Target Response', (1, 2), rootdir=RESULT_PATH, save=True, size=(18*2, 9*2)) as fig:
    fig.addAll()
    
    fig[0].set_title('S11')
    fig[0].plot(x, returnloss.detach().numpy(), color='red', marker="o")
    fig[0].plot(x, returnloss_upper, color='blue', marker="o")
    fig[0].plot(x, returnloss_lower, color='blue', marker="o")
    fig[0].grid(True)
    # fig[0].set_ylim(-13, 1)
    
    fig[1].set_title('Gain')
    fig[1].plot(x,gain.detach().numpy(), color='red', marker="o")
    fig[1].plot(x, gain_upper, color='blue', marker="o")
    fig[1].plot(x, gain_lower, color='blue', marker="o")
    fig[1].grid(True)
    fig[1].grid(True)

###*  初始化神經網絡模型 ###
model = OldGEN()
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=config.lr, betas=(0.5, 0.999)
)
smodel = OldSM()

###* 斷點續跑 ###
if is_connect_run:
    last_model = path_checkpoint.joinpath(f"gen_model_{TEMP('epoch')}.pth")
    Antenna_checkpoint_loaded = last_model.load_torch()
    model.load_state_dict(Antenna_checkpoint_loaded['state_dict'])
    optimizer.load_state_dict(Antenna_checkpoint_loaded['optimizer'])


# Optimizer setting
# optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr)
# optimizer = torch.optim.RMSprop(params=model.parameters(), lr=init_lr)

config['AntennaResponse'] = AntennaResponse.to_str()
config['Generator'] = model
config['optimizer'] = optimizer
config['SurrogateModel'] = smodel
config.save(rootdir=RESULT_PATH)

###* Training ###
epoch = TEMP('epoch', 0) # 總訓練次數
current_epoch = 0   # 斷掉後的訓練次數
jump = 0 # 跳躍次數
while epoch < config.epochs + 1:

    epoch += 1
    current_epoch += 1

    if current_epoch % 15 == 0 or current_epoch == 1:
        simulator.reopen()

    simulator.start(epoch)
    logger.info(f"Start {epoch} of {config.epochs}")

    model.train()
    optimizer.zero_grad() # adjust_lr(optimizer, epoch, init_lr)

    ###* 生成 pattern ###
    #? target response -> 生成模型 -> pattern
    output_element = AntennaPattern(
        model(AntennaResponse.merge_target_responses())
    ) + lower

    with Figure(f"pattern_{epoch}", save=True, rootdir=path_pic) as  fig:
        fig.addAll()
        output_element.plot(fig[0])

    if (False and (TEMP('patch_pattern_buf') == TEMP['patch_pattern_buf'][-2]).all()):
        jump = jump + 1
    else:

        min_loss = TEMP('min_loss', float('inf'))
        if TEMP('real_loss') <= min_loss:
            min_loss = TEMP('real_loss')
            de = TEMP('de', 0)

        else:
            min_loss = TEMP('min_loss', float('inf'))
            de = TEMP('de', 0) + 1
        jump = 0

    output_result = output_element.simulate()
    real_loss = AntennaResponse.multi_responses_to_loss(output_result)

    ###* 儲存HFSS的輸入與輸出 ###
    TEMP['patch_pattern_buf'] = ~output_element
    TEMP['patch_result_buf'] = stack([ n.response for n in output_result.values()])

    ###* 訓練代理模型並儲存 ###
    sm_loss = smodel.train(output_element.series, TEMP('patch_result_buf'))
    smodel.save(path_checkpoint)

    ###* 權重全部凍結 ###
    # for name, para in model_HFSS.named_parameters():
    #     para.requires_grad_(False)

    ###* 更新GEN ###
    #? target response -> 生成模型 -> pattern -> 代理模型 -> predicted response
    #? calculate loss (target response, predicted response)
    #? update optimizer
    output_element = model(AntennaResponse.merge_target_responses())
    response = smodel(output_element)
    loss = AntennaResponse.multi_responses_to_loss(response)
    loss.backward()
    optimizer.step()
    model.eval()

    ###* 儲存模型 ###
    gen_checkpoint = {
        'model': model,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(gen_checkpoint, path_checkpoint.joinpath(f"gen_model_{epoch}.pth"))

    ###* 儲存real與fake的loss ###
    TEMP['real_loss'] = real_loss.item()
    TEMP['fake_loss'] = loss.item() 

    
    with Figure(f"Result {epoch}",(2,2), rootdir=path_pic, save=True, size=(18*2, 9*2)) as fig:
        fig.addAll()

        fig[0].plot(x,output_result['S11'].response, color='blue')
        fig[0].plot(x,returnloss, color='blue', linestyle='--')
        fig[0].plot(x,returnloss_upper, color='red')
        fig[0].plot(x, returnloss_lower, color='red')
        fig[0].set_title('S11 Response', fontsize=20)
        fig[0].set_ylim(-13,1)

        fig[1].plot(x,output_result['Gain'].response, color='blue')
        fig[1].plot(x,gain, color='blue', linestyle='--')
        fig[1].plot(x,gain_upper, color='red')
        fig[1].plot(x, gain_upper, color='red')
        fig[1].set_title('Gain', fontsize=20)
        fig[1].set_ylim(-13,1)
        
        fig[2].plot(TEMP['real_loss'], color='red', label='real_loss')
        fig[2].plot(TEMP['fake_loss'], color='purple', label='fake_loss', alpha=0.8)
        fig[2].legend()
        fig[2].set_title("Loss Curve", fontsize=20)

        fig[3].set_title('sm_loss', fontsize=20)
        fig[3].plot(sm_loss)

    
    exe_time = simulator.end()
    logger.info(f"End {epoch} of {config.epochs}, Loss: {TEMP('real_loss'):4f}, Time: {exe_time} s")

    TEMP['de'] = de     #  np.save(path_save_data.joinpath("de.npy"), de)
    TEMP['epoch'] = epoch
    TEMP["min_loss"] = min_loss    # np.save(path_save_data.joinpath("min_loss.npy"), min_loss.detach().numpy())

    TEMP.save(f"{epoch} times")

logger.info(f"Training Finished! (Min Loss: {TEMP.custom('real_loss', min)})")

#%%
simulator.save()
simulator.quit()

