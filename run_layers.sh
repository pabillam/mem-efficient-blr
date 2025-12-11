#!/bin/bash

device_name=$(python3 -c "import yaml; print(yaml.safe_load(open('./configs/config.yaml'))['device_name'])")

all_layers=false
layer_config=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --all)
            all_layers=true
            shift
            ;;
        --layer_config)
            layer_config="$2"
            shift 2
            ;;
	--gpu_profile)
	   gpu_profile=true
	   shift
	   ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ "$all_layers" == true ]]; then
    for layer_config in ./configs/layers/*.yaml; do
        filename=$(basename -- "$layer_config")
        network_name="${filename%%.*}"
        layer_name="${filename#*.}"
        layer_name="${layer_name%.yaml}"

        echo "Processing: Network = $network_name, Layer = $layer_name, Device = $device_name"
        
        python3 run_layers.py --device "$device_name" --layer "$layer_name" --network "$network_name"
	if [[ "$gpu_profile" == true ]]; then
	    ./gpu_profile.sh "$network_name" "$layer_name"
	fi
    done
elif [[ -n "$layer_config" ]]; then
    filename=$(basename -- "$layer_config")
    network_name="${filename%%.*}"
    layer_name="${filename#*.}"
    layer_name="${layer_name%.yaml}"

    echo "Processing: Network = $network_name, Layer = $layer_name, Device = $device_name"
    
    python3 run_layers.py --device "$device_name" --layer "$layer_name" --network "$network_name"
    if [[ "$gpu_profile" == true ]]; then
        ./gpu_profile.sh "$network_name" "$layer_name"
    fi
else
    echo "Error: Must specify either --all or --layer_config <path>"
    exit 1
fi
