# ai-capstone-hw0

NYCU AI Capstone 2026 Fall

Spec: https://drive.google.com/file/d/1R2M2fLBaT8iETkZSja0vNdlWh06UssEg/view?usp=sharing

Slide: https://docs.google.com/presentation/d/1FSDUIIN-GM05lxHIfW1OII-CdiMmDhrn/edit?usp=sharing&ouid=114385544772538729562&rtpof=true&sd=true

for windows user: https://drive.google.com/file/d/1lAmHQKl-Lfw1P2uvEHe44BafYjCDAKm9/view?usp=sharing

## Introduction 
In this course, we are going to build an indoor navigation system in Habitat step by step during homework 1 ~ 3 and the final project. This homework 0 will help you to build the environment with essential packages.

## Installation

### Clone the repo
`git clone git@github.com:HCIS-Lab/ai-capstone26.git` to download the repo or create a new fork on you own GitHub account.

```bash
cd ai-capstone26
# Ensure the latest submodules
git submodule update --init --recursive
# Create a conda env
conda create -n habitat python=3.9 cmake=3.19.6
# Activate the conda env
conda activate habitat
# Install requirements
cd hw0
pip install -r requirements.txt
# Install habitat-sim 
conda install habitat-sim=0.3.3 withbullet -c conda-forge -c aihabitat
# Install habitat-lab
cd habitat-lab && pip install -e habitat-lab
```

### Download dataset

Download dataset from [here](https://drive.google.com/file/d/1zHA2AYRtJOmlRaHNuXOvC_OaVxHe56M4/view)
and put the directory under `replica_v1/`

or 
```bash
cd replica_v1
gdown https://drive.google.com/uc?id=1zHA2AYRtJOmlRaHNuXOvC_OaVxHe56M4 -O apartment_0.zip
unzip apartment_0.zip
```

## Tasks

1. Run the `load.py` to explore the scene

    Use keyboard to control the agent
    ```
    w for go forward  
    a for turn left  
    d for trun right  
    f for finish and quit the program
    ```

    **Note**

    You can change agent_state.position to set the agent in the first or second floor. (0, 0, 0) for first floor and (0, 1, 0) for second floor.


2. **Record the video on screen (<30 sec)** at the same time to demonstrate that the simulation can be opened up successfully.

    
