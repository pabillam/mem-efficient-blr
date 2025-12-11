import yaml
import sys

def parse_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to space-separated strings for Bash compatibility
            items.append((new_key, " ".join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)

def merge_dicts(dict1, dict2):
    merged = dict1.copy()
    merged.update(dict2)  # dict2 overrides dict1
    return merged

def main(config_file, layer_config_file):
    config_dict = parse_yaml(config_file)
    layer_config_dict = parse_yaml(layer_config_file)

    merged_dict = merge_dicts(config_dict, layer_config_dict)

    flattened_dict = flatten_dict(merged_dict, sep='_')

    for key, value in flattened_dict.items():
        print(f'{key}="{value}"')

if __name__ == '__main__':
    config_file = sys.argv[1]
    layer_config_file = sys.argv[2]
    main(config_file, layer_config_file)
