# Yuling Wu's Undergraduate Thesis Experiment - Pong Game
This is an Experiment about *Dynamically Rule-Interposing Deep Reinforcement Learning and Rules Generalization*. 

The idea is about proposing a method that combines both knowledge representation and DRL.

## Quick Start
- To use GPU, CUDA and cuDNN should be installed correctly.

    *How to install?* 
    + [CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
    + [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)
    
- To record the final traning model playing the game, ffmpeg should be installed in your system.

    *How to install?*
    + [FFmpeg](http://ffmpeg.org/download.html)
    
1.  Install the requirements using:  
    ```bash 
    pip install -r requirements.txt
    ```
2.  Run the DRIL.  
    ```bash 
    python pong_DRIL.py
    ```  
    Run the DQN.  
    ```bash 
    python pong_DQN.py
    ```
3.  If you need to check how to tweak the parameters. Using:  
    `python pong_DRIL.py -h` or `python pong_DQN.py -h`

    *However, the Rules-Interposing Decay Models' parameters can only be tweaked in files.*  

4. Load the trained model.
    ```bash 
    python pong_load_model.py
    ```  
   *If you need more help about the load model module, please use `python pong_load_model.py -h`*

## Training Results
Average Q Values on Pong  
> ./plotData/Pong_Avg_Q_Values.png  

Average Rewards on Pong  
> ./plotData/Pong_Avg_Rewards.png

## TODO  
The hyperparameters may not be the best for Pong game in both DRIL and DQN. It still needs more explorations.

## References
[1] Haodi Zhang, Zihang Gao, Yi Zhou, et al. Faster and Safer Training by Embedding High-Level Knowledge into Deep Reinforcement Learning. arXiv preprint. 2019:1910.09986.  

[2] Mnih V., Kavukcuoglu K., Silver D., et al. Human-level control through deep reinforcement learning. Nature,2015,518(7540):529-533.