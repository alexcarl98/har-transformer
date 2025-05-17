from data import *
from config import *
import yaml
import shutil
from typing import Dict, List
from itertools import combinations
from train import *
from util import WandBLogger


def cross_sensor_train(config: Config, val_data: HarDataset, logger: Optional[WandBLogger] = None) -> Optional[WandBLogger]:
    set_all_seeds(42)
    
    # Initialize WandB logger if wandb is enabled
    if logger is None and config.wandb.mode != 'disabled':
        logger = WandBLogger(config)
    
    data_loader = GeneralDataLoader(**config.get_data_loader_params())
    train_data = data_loader.get_har_dataset('train')
    print("X_train shape:", train_data.X.shape)
    print("X_val shape:", val_data.X.shape)
    print("Classes:", np.unique(train_data.y))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.transformer.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.transformer.batch_size)
    model_dir = config.output_paths.models_dir

    # === Model, loss, optimizer ===
    model = v1.AccelTransformerV1(**config.get_transformer_params()).to(DEVICE)
    print(model)

    optimizer = Adam(model.parameters(),
                    lr=config.transformer.learning_rate, 
                    weight_decay=config.transformer.weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    train_model(
        config.transformer, train_loader, val_loader, 
        model, optimizer, criterion, model_dir,
        logger=logger
    )
    
    return logger


def cross_sensor_confusion_matrix(config: Config, logger: WandBLogger):
    data_loader = GeneralDataLoader(**config.get_data_loader_params())
    test_data = data_loader.get_har_dataset('test')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.transformer.batch_size)

    model = v1.AccelTransformerV1(**config.get_transformer_params()).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    print("testing most recent model:")
    print(f"===(Testing)===")
    _, last_f1 = evaluate_and_save_metrics(
        config, 'last', model, test_loader, criterion,
        logger=logger
    )

    print("testing f1 best model:")
    _, best_f1 = evaluate_and_save_metrics(
        config, 'best_f1', model, test_loader, criterion,
        logger=logger
    )
    
    return last_f1, best_f1
def report_model_architecture(config: Config, base_path: str):
    '''
    If we made it through atleast one cross-sensor run, 
    we log the model architecture and experiment setup
    '''
    model = v1.AccelTransformerV1(**config.get_transformer_params())
    with open(f"{base_path}/arch.log", "w") as f:
        f.write(str(model))

    # copy the config file to the base_path
    shutil.copy(config.file_path, f"{base_path}/config.yml")
    return



def main():
    valid_sensors = ['ankle', 'wrist', 'waist']
    # Get all permutations
    to_be_trained_on = [list(comb) for r in range(1, len(valid_sensors) + 1) for comb in combinations(valid_sensors, r)]
    print(f"{to_be_trained_on=}")
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = 'CSCM_' + run_time
    # Load config
    location_outcome_dict = {}
    config = Config.from_yaml('config.yml')
    tmp_data_loader = GeneralDataLoader(**config.get_data_loader_params())
    cross_sensor_validation_set = tmp_data_loader.get_har_dataset('val')

    config.wandb.project = test_name
    base_path = config.output_paths.base_path + f'/{test_name}'
    os.makedirs(base_path, exist_ok=True)
    wandb_url = f"https://wandb.ai/{config.wandb.entity}/{config.wandb.project}"
    yaml_pt = f"{base_path}/location_outcomes.yml"
    print(f"{config.data.partitions.mapping}")
    print(tmp_data_loader.current_subjects)
    with open(yaml_pt, 'w') as f:
        yaml.safe_dump(
            {'wandb_url': wandb_url},
            f,
            default_flow_style=False,
        )

    for i, train_sensor in enumerate(to_be_trained_on):
        # if len(train_sensor) == 1:
        #     continue
        config.data.train_on_sensors = train_sensor
        r_id = f'{len(train_sensor)}-{'-'.join(train_sensor)}'
        new_output_path = OutputPathsConfig(base_path, run_id = r_id)
        location_outcome_dict[r_id] = {}
        config.output_paths = new_output_path

        logger = WandBLogger(config, run_name = r_id)
        print(f"Training on {train_sensor}")
        logger = cross_sensor_train(config, cross_sensor_validation_set, logger)

        base_plot_dir = config.output_paths.plots_dir
        
        for test_sensor in valid_sensors:
            config.output_paths.plots_dir = base_plot_dir + f'/{test_sensor}'
            os.makedirs(config.output_paths.plots_dir, exist_ok=True)
            logger.cm_save_dir = f'{test_sensor}_confusion_matrix'
            logger.roc_save_dir = f'{test_sensor}_roc_curve'
            logger.batch_save_dir = f'{test_sensor}_batch_examples'

            config.data.test_on_sensors = [test_sensor]
            print(f"\tTesting on {test_sensor}")
            # print(f"\tsaving to {config.output_paths.plots_dir}")

            last_f1, best_f1 = cross_sensor_confusion_matrix(config, logger)
            results = {
                f'last_f1': last_f1,
                f'best_f1': best_f1
            }
            location_outcome_dict[r_id][test_sensor] = results
            print(f"\t{location_outcome_dict[r_id]=}")

        last_sum = 0
        best_sum = 0
        for sensor_location in location_outcome_dict[r_id]:
            print(f"\t{sensor_location=}")
            last_sum += location_outcome_dict[r_id][sensor_location]['last_f1']
            best_sum += location_outcome_dict[r_id][sensor_location]['best_f1']
            # logger.log_metrics(results)

        avg_last_f1 = last_sum / len(location_outcome_dict[r_id])
        avg_best_f1 = best_sum / len(location_outcome_dict[r_id])
        location_outcome_dict[r_id]['avg_last_f1'] = avg_last_f1
        location_outcome_dict[r_id]['avg_best_f1'] = avg_best_f1
        with open(yaml_pt, 'a') as f:
            yaml.safe_dump(
                {r_id: location_outcome_dict[r_id]},
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=5,
                allow_unicode=True,
                width=120
            )

        logger.log_metrics(location_outcome_dict[r_id])
        logger.finish()
        if i == 0:
            report_model_architecture(config, base_path)

if __name__ == "__main__":

    main()
      
