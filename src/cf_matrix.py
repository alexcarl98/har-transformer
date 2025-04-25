from data import *
from config import *
import yaml

from typing import Dict, List
from itertools import combinations
from train import *

def cross_sensor_confusion_matrix(config, data_config: DataConfig, bio_data: BiometricsData, parts: Dict[str, List[str]], min_sample_count: int):

    data_loader = GeneralDataLoader(data_config, bio_data, parts, min_sample_count)

    set_all_seeds(42)

    decoder_dict = data_loader.decoder_dict

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

    # === Model, loss, optimizer ===
    model = v1.AccelTransformerV1(
        d_model=config.transformer.d_model,
        fc_hidden_dim=config.transformer.fc_hidden_dim,
        num_classes=len(data_loader.data_config.classes),
        in_channels=len(data_loader.data_config.ft_col),
        nhead=config.transformer.nhead,
        num_layers=config.transformer.num_layers,
        dropout=config.transformer.dropout,
        patch_size=config.transformer.patch_size,
        kernel_stride=config.transformer.kernel_stride,
        window_size=data_loader.data_config.window_size,
        extracted_features=config.transformer.extracted_features
    ).to(DEVICE)

    print(model)

    optimizer = Adam(model.parameters(),
                      lr=config.transformer.learning_rate, 
                      weight_decay=config.transformer.weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    train_model(config.transformer, train_loader, val_loader, model, optimizer, criterion, DEVICE, config.output_paths.models_dir)
    

    print("testing most recent model:")
    exit()



def main():
    valid_sensors = ['ankle', 'wrist', 'waist']
    # Get all permutations
    to_be_trained_on = [list(comb) for r in range(1, len(valid_sensors) + 1) for comb in combinations(valid_sensors, r)]
    print(f"{to_be_trained_on=}")

    # Load config
    location_outcome_dict = {}
    config = Config.from_yaml('config.yml')
    data_settings= config.get_data_config_path()

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = config.output_paths.base_path + f'/ConfusionMatrix_{run_time}'
    bio_data = BiometricsData.from_csv()

    '''
    TODO:
    - remove redundancy in training the same model three times
    - separate training and testing as their own functions
    - Need to create new classes:
        - Analysis Class
        - Transformer Model class
    '''

    data_config = DataConfig.from_yaml(data_settings)
    for i, sensor in enumerate(valid_sensors):
        # you're training the same model three times, make the training and testing separate functions. 
        location_outcome_dict[sensor] = []
        data_config.test_on_sensors = [sensor]
        current_base = base_path + f'/tested_on_{sensor}'
        
        for j, train_sensors in enumerate(to_be_trained_on):
            len_sen = len(train_sensors)
            id = f'{len_sen}-{'-'.join(train_sensors)}'
            new_output_path = OutputPathsConfig(current_base, data_settings = data_settings, run_id = id)
            config.output_paths = new_output_path
            data_config.train_on_sensors = train_sensors
            parts, min_sample_count = obtain_standard_partitions(bio_data, data_config.classes, config.get_data_config_path())
            
            result_dict = {}
            outcome = cross_sensor_confusion_matrix(config, data_config, bio_data, parts, min_sample_count)
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
      
