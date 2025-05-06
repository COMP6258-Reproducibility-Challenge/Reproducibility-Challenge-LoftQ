# Setup

1. Create directories in /scratch/username/ for quantization and training outputs:
- quantized_models
- trained_models
- trainer_outputs
- results

2. Create symlinks from these directories to you /custom_loftq folder - ln -s /scratch/USERID/PATH_TO_DIR ./PATH_TO_custom_loftq

3. Adjust any scripts you want to run to include the necessary batch setup

4. For any job requiring a dataset, the very first time you ever try to load the data set (or anything that needs to download) run it on the base iridis - so WITHOUT sbatch - as the compute nodes have no internet access

# Usage

The scripts dir includes several basic tasks from just quanitzing to training a model and loading.

The valid arguments can be found in arguments.py - note that not all arguments (especially for the model) do anything right now as they were copied from the original repo's quantize_save.py. 

We mostly want to use --model_name_or_path; loftq things like --reduced_rank, --int_bit, etc; --from_saved: this will look in the --save_dir (or default) for an already quantized model, --train_small (train on a very small dataset to test it works), --no_train (only quantize the model). 

--save_steps xxx - change how many training steps between checkpoints

The training args are an extension of the default HF Transformers Training Args so include the ability to resume from a checkpoint - very helpful if the sbatch runs out of time - I think checkpoints are stored in the trainer_output, just give the checkpoint_X dir in the --resume_from_checkpoint argument.

# Current state
Currently, I have done the work for the preprocessing and running of DeBertaV3-base on the Glue set. If you want to train on other things - eg gsm8k you will need to implement the preprocessing etc. Please do this in a similar way that I have by creating a seperate file or on your own branch to not mess anyone elses stuff up. For Glue I set up classification_utils.py - please do something similar. You will also need to edit or duplicate and edit run_loftq.py.

# B16 commands
Open an 'anaconda prompt'
If does exist, install conda from https://www.anaconda.com/download/success

Once open may need to conda init

conda create - C:\Users\USERNAME\.conda\envs\loftq_env
conda activate loftq
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
conda install nvidia::cuda-toolkit
conda install nvidia::nvcc

Until we find a fix - delete deepspeed from requirements.txt

You'll need to do normal git stuff to install the repo
pip install -r requirements.txt
accelerate config (this takes a while) - choose default options (except numa??)

see example bat file unless you can figure out how to run .sh on these machines

PS It is possible to do all of this on the filestore, but it takes forever to do anything - even running a python script takes 2mins just to import libraries so its way faster to install everything to C drive (especially if you'll be in labs all day)
