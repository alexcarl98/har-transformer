# create_sweep.py
from utils import TConfig
import wandb

def create_sweep():
    sweep_configuration = TConfig.create_sweep_dict()
    sweep_id = wandb.sweep(
        sweep_configuration, 
        project="HAR-PosTransformer",
        entity="alex-alvarez1903-loyola-marymount-university"
    )
    print(f"Created sweep with ID: {sweep_id}")
    print("\nTo run the sweep, use this command:")
    print(f"wandb agent alex-alvarez1903-loyola-marymount-university/HAR-PosTransformer/{sweep_id}")

if __name__ == "__main__":
    create_sweep()

# IN the terminal, run:
# wandb agent alex-alvarez1903-loyola-marymount-university/HAR-PosTransformer/sweep_id
# wandb agent alex-alvarez1903-loyola-marymount-university/HAR-PosTransformer/n7il4x0e