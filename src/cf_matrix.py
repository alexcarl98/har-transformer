from data import *
from config import *
import yaml

from typing import Dict, List
from itertools import combinations
from train import *

def cross_sensor_confusion_matrix(config: Config):

    set_all_seeds(42)

    data_loader = GeneralDataLoader(**config.get_data_loader_params())

    # run = wandb.init(
    #     entity=config.wandb.entity,
    #     project=config.wandb.project,
    #     config=config.transformer,
    #     mode=config.wandb.mode,
    # )

    train_data = data_loader.get_har_dataset('train')
    val_data = data_loader.get_har_dataset('val')
    test_data = data_loader.get_har_dataset('test')

    print("X_train shape:", train_data.X.shape)
    print("X_val shape:", val_data.X.shape)
    print("X_test shape:", test_data.X.shape)
    print("Classes:", np.unique(train_data.y))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.transformer.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.transformer.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.transformer.batch_size)
    model_dir = config.output_paths.models_dir

    # === Model, loss, optimizer ===
    model = v1.AccelTransformerV1(**config.get_transformer_params()).to(DEVICE)

    print(model)

    optimizer = Adam(model.parameters(),
                      lr=config.transformer.learning_rate, 
                      weight_decay=config.transformer.weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    train_model(config.transformer, train_loader, val_loader, model, optimizer, criterion, model_dir)
    
    print("testing most recent model:")

    plot_dir = config.output_paths.plots_dir
    classNames = data_loader.data_config.classes
    ft_col = data_loader.data_config.ft_col
    print(f"===(Testing)===")
    evaluate_and_save_metrics('last', model, test_loader, criterion, plot_dir, classNames, ft_col)

    print("testing f1 best model:")
    evaluate_and_save_metrics('best_f1', model, test_loader, criterion, plot_dir, classNames, ft_col)


    # run.finish()

    config.output_paths.clean()

    exit()


def main():
    valid_sensors = ['ankle', 'wrist', 'waist']
    # Get all permutations
    to_be_trained_on = [list(comb) for r in range(1, len(valid_sensors) + 1) for comb in combinations(valid_sensors, r)]
    print(f"{to_be_trained_on=}")

    # Load config
    location_outcome_dict = {}
    config = Config.from_yaml('config.yml')

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = config.output_paths.base_path + f'/ConfusionMatrix_{run_time}'

    '''
    TODO:
    - remove redundancy in training the same model three times
    - separate training and testing as their own functions
    - Need to create new classes:
        - Analysis Class
        - Transformer Model class
    '''

    for i, sensor in enumerate(valid_sensors):
        # you're training the same model three times, make the training and testing separate functions. 
        location_outcome_dict[sensor] = []
        config.data.test_on_sensors = [sensor]
        current_base = base_path + f'/tested_on_{sensor}'
        
        for j, train_sensors in enumerate(to_be_trained_on):
            len_sen = len(train_sensors)
            id = f'{len_sen}-{'-'.join(train_sensors)}'
            new_output_path = OutputPathsConfig(current_base, run_id = id)
            config.output_paths = new_output_path
            config.data.train_on_sensors = train_sensors
            
            result_dict = {}
            outcome = cross_sensor_confusion_matrix(config)
            result_dict[id] = outcome
            
            location_outcome_dict[sensor].append(result_dict)

    
    # Dump to YAML file

    # For prettier formatting, you can use yaml.safe_dump with additional options:
    with open(f'{base_path}/location_outcomes.yml', 'w') as f:
        yaml.safe_dump(
            location_outcome_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
            indent=5,
            allow_unicode=True,
            width=120
        )




if __name__ == "__main__":

    main()
      
