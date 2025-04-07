conda create -y -n har python=3.12

conda activate har

conda install -c conda-forge numpy pandas scipy scikit-learn seaborn tqdm matplotlib wandb
conda install pytorch -c pytorch
conda install -c conda-forge pytest

pip install ahrs
