# CoEvo Bandit Learning on Matrix Games with Bandit Feedback

This repository contains the code for our research paper accepted to IJCAI 2025. 
The code includes Python implementation of algorithms for matrix games with bandit feedback.
All algorithms are in folder name after `Algorithms`.


## Installation

We recommend using either **virtualenv** or **Conda** to install the environment:
Follow the instructions for your preferred method below.

### Step 1: Create and Activate the Virtual Environment

#### Option 1: Using `virtualenv`

First, navigate to your project directory and create a virtual environment using `virtualenv`:

```bash
# Navigate to your project directory
cd my-project

# Create a virtual environment with Python 3.11
virtualenv --python=python3.11 venvgame

```

#### Option 2: Using `conda`

```bash
# Create a Conda environment named 'venvgame' with Python 3.11
conda create -n venvgame python=3.11

# Activate the environment
conda activate venvgame
```

### Step 2: Install the required packages


```bash
# Activate the virtual environment (Linux/macOS)
source venvgame/bin/activate

pip3 install -r requirements.txt

```

## Usage

Take Rock-Paper-Sicssor game as an exmaple.

```bash
cd RPS_Game

python3 <script_name>.py
```

For example, 
```bash
python3 Exp3_SelfPlay.py
```

## Parallelisation and Cluster Execution
Our scripts use Python's multiprocessing library to parallelise each experiment run, making it efficient for larger experiments. 
This is especially useful when running experiments on a cluster or multi-core system.

To run experiments in parallel, you don't need to modify the scripts. 
The multiprocessing is handled internally to distribute the workload across available CPU cores.

For clusters or larger experiments:

You can use cluster job submission tools (like SLURM, PBS, etc.) to run multiple jobs in parallel.
Each job can run a separate instance of an experiment, utilising multiprocessing for efficient resource usage.

## License
The code is released under the MIT License.
