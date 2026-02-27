# Learning Semantic Navigation Primitives for human habitats using DreamerV3 world model

This work is based on the original DreamerV3 code, which was developed by Danijar Hafner et al.
To read the original DreamerV3 paper and explore the original code, please follow Danijar's github 
repository: [Original DreamerV3][original_dreamerv3]

Our contributions include integration of AI2-Thor environment with DreamerV3 to allow using
ProcThor-10k dataset for training world models exhibiting useful behaviours. We call such behaviours
Semantic Navigation Primitives (SNPs).

## SNP DreamerV3

The quickest way to try it out, is to first clone this repository recursively with all the
submodule dependencies, install all required packages in a dedicated Anaconda environment
and then launch it using one of the commands below.

### Download and installation:
```
git clone --recursive https://github.com/archie1983/snp_dreamerv3
cd snp_dreamerv3
conda create -n snp_d3 python=3.11
conda activate snp_d3
conda env update -f dreamer3.yml -n snp_d3
cd ai2_thor_model_training_src
pip install -e .
cd thortils
pip install -e .
cd ../../
```

### Training SNP to find room center:
The following command will launch indoors room center training, using 
ProcThor-10K train dataset. It will use 2 environment instances and it will have 
a batch size of 2. If your system has enough resources to use more environments 
and have a higher batch size, then your results will be better with these 
numbers higher. The training results will be stored in ~/logdir/train_results.

```
python dreamerv3/main.py --configs indoorsrc --run.envs 2 --batch_size 2 --logdir ~/logdir/train_results
```

### Training SNP to find a door and walk through it:
The following command will do the same as above, but for the SNP that finds a door
and transits through it.
```
python dreamerv3/main.py --configs indoorsdoor --run.envs 2 --batch_size 2 --logdir ~/logdir/train_results
```

### Evaluating a trained room center finding SNP
The following command will load a previously trained checkpoint from
"~/logdir/train_results/rc_ckpt/" and evaluate it on the
ProcThor-10K test dataset. It will store data in the "rc_eval_results" directory, which has
to exist prior. It will use 3 AI2-Thor environments and process the episodes in parallel.
The evaluation code has been tested with 3 environments only. If you need more or less,
you may need to adapt the way the test data set is split between the environments.
This is done in "embodied/envs/indoors_flat.py", in the init function of AI2ThorBase class.
```
python dreamerv3/main.py --configs indoorsaeroomcentre_eval --run.envs 3 --batch_size 3 --run.from_checkpoint "~/logdir/train_results/rc_ckpt/" --logdir "rc_eval_results"
```

### Evaluating a trained door finding SNP
The following will do the same as above, but for door finding SNP, storing results in "door_eval_results" and
loading checkpoint from "~/logdir/train_results/door_ckpt/".
```
python dreamerv3/main.py --configs indoorsaedoor_eval --run.envs 3 --batch_size 3 --run.from_checkpoint "~/logdir/train_results/door_ckpt/" --logdir "door_eval_results"
```

### Potential problems
The code is using headless AI2-Thor for training and evaluation. It expects 2 GPUs:
1 for generating AI2-Thor imagery and the other for training. This can be changed in:
"embodied/envs/indoors_flat.py", the "load_habitat" function, by changing parameters, that are
passed to "launch_controller".

Please get in touch, using the GitHub Discussions or Issues sections, should you require
more support to get it running.

[original_dreamerv3]: https://github.com/danijar/dreamerv3