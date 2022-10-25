# Object Goal Navigation using Goal-Oriented Semantic Exploration

The origin of this repo is [link](https://devendrachaplot.github.io/projects/semantic-exploration)

### Overview:
The Goal-Oriented Semantic Exploration (SemExp) model consists of three modules: a Semantic Mapping Module, a Goal-Oriented Semantic Policy, and a deterministic Local Policy. 
As shown below, the Semantic Mapping model builds a semantic map over time. The Goal-Oriented Semantic Policy selects a long-term goal based on the semantic
map to reach the given object goal efficiently. A deterministic local policy based on analytical planners is used to take low-level navigation actions to reach the long-term goal.

### This repository contains:
- Episode train and test datasets for [Object Goal Navigation](https://arxiv.org/pdf/2007.00643.pdf) task for the Gibson dataset in the Habitat Simulator.
- The code to train and evaluate the Semantic Exploration (SemExp) model on the Object Goal Navigation task.
- Pretrained SemExp model.

## Installing Dependencies

You can use the latest habitat.

Installing habitat-sim:
```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim;
pip install -r requirements.txt; 
python setup.py install --headless
python setup.py install # (for Mac OS)
```

Installing habitat-lab:
```
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab;
pip install -e .
```
Check habitat installation by running `python examples/benchmark.py` in the habitat-lab folder.

- Install [pytorch](https://pytorch.org/) according to your system configuration. The code is tested on pytorch v1.6.0 and cudatoolkit v10.2. If you are using conda:

```
python -m pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html #(Linux with GPU)

```

- Install [detectron2](https://github.com/facebookresearch/detectron2/) according to your system configuration.

### Docker and Singularity images:

TBA

## Setup

Clone the repository and install other requirements:

```bash
git clone https://github.com/devendrachaplot/Object-Goal-Navigation/
cd Object-Goal-Navigation/;
pip install -r requirements.txt
```

### Downloading scene dataset

* Gibson

- Download the Gibson dataset using the instructions here: https://github.com/facebookresearch/habitat-lab#scenes-datasets (download the 11GB file `gibson_habitat_trainval.zip`)
- Move the Gibson scene dataset or create a symlink at `data/scene_datasets/gibson_semantic`. 

* HM3D

TBA

### Downloading episode dataset

- Download the episode dataset:
```
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1tslnZAkH8m3V5nP8pbtBmaR2XEfr8Rau' -O objectnav_gibson_v1.1.zip
```
- Unzip the dataset into `data/datasets/objectnav/gibson/v1.1/`

### Setting up datasets

The code requires the datasets in a `data` folder in the following format (same as habitat-lab):
```
Object-Goal-Navigation/
  data/
    scene_datasets/
      gibson_semantic/
        Adrian.glb
        Adrian.navmesh
        ...
      hm3d/
        train/
          ...
        val/
          ...
    datasets/
      objectnav/
        gibson/
          v1.1/
            train/
            val/
        hm3d/
          v1/
            train/
            val/
```


### Test setup

To verify that the data is setup correctly, run:
```
python test.py --agent random -n 1 --num_eval_episodes 1 --auto_gpu_config 0
```

## Usage

### Training:
For training the SemExp model on the Object Goal Navigation task:

```
python main.py
```

### Downloading pre-trained models

you can download the pretrained model from the original author:

```
mkdir pretrained_models;
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=171ZA7XNu5vi3XLpuKs8DuGGZrYyuSjL0' -O pretrained_models/sem_exp.pth
```

### For evaluation: 

For evaluating the pre-trained model:
```
python main.py --split val --eval 1 --load pretrained_models/sem_exp.pth
```

For visualizing the agent observations and predicted semantic map, add `-v 1` as an argument to the above command.

The pre-trained model should get 0.657 Success, 0.339 SPL and 1.474 DTG.

For more detailed instructions, see [INSTRUCTIONS](./docs/INSTRUCTIONS.md).

## TODO



## Related Projects

- This project builds on the [Active Neural SLAM](https://devendrachaplot.github.io/projects/Neural-SLAM) paper. The code and pretrained models for the Active Neural SLAM system are available at:
https://github.com/devendrachaplot/Neural-SLAM.

- The Semantic Mapping module is similar to the one used in [Semantic Curiosity](https://devendrachaplot.github.io/projects/SemanticCuriosity).

## Acknowledgements
This repository uses [Habitat Lab](https://github.com/facebookresearch/habitat-lab) implementation for running the RL environment.
The implementation of PPO is borrowed from [ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/). 
The Mask-RCNN implementation is based on the [detectron2](https://github.com/facebookresearch/detectron2/) repository. We would also like to thank Shubham Tulsiani and Saurabh Gupta for their help in implementing some parts of the code.
