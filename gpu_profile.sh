#!/bin/bash

parse_and_merge_yaml() {
    local config_file=$1
    local layer_config_file=$2

    eval $(python ./helper/parse_and_merge_yaml.py "$config_file" "$layer_config_file")
}

convert_to_array() {
    local var_name=$1
    local var_value=$2
    IFS=' ' read -r -a "$var_name" <<< "$var_value"
}

check_best_config() {
    local function_name=$1
    local short_function_name="${function_name##*.}"

    if [[ "$function_name" == *"triton"* && "$function_name" == *"bmm"* ]]; then
        local dtype_suffix=""
        if [[ "$function_name" == *"int8"* ]]; then
            dtype_suffix="_int8"
        fi

        local suffixes=("xv" "sxv" "usxv")
        for s in "${suffixes[@]}"; do
            local config_path="$output_dir/triton_blast_bmm_${s}${dtype_suffix}_fp16_best_config.pkl"
            if [[ ! -f "$config_path" ]]; then
                echo "Skipping $function_name: best_config not found ($config_path)"
                return 1
            fi
        done

        return 0
    fi

    local config_path="$output_dir/${short_function_name}_best_config.pkl"
    if [[ ! -f "$config_path" ]]; then
        echo "Skipping $function_name: best_config not found ($config_path)"
        return 1
    fi

    return 0
}

nsys_command() {
    local num_batches=$1
    local num_seq=$2
    local in_f=$3
    local out_f=$4
    local function_name=$5
    local config_dir=$6
    local is_compiled=$7
    local b=${8:-1}
    local rank=${9:-0}

    local short_function_name="${function_name##*.}"

    command=(
        "nsys" "profile" "-w" "true"
        "-t" "cuda,nvtx,osrt,cudnn,cublas"
        "-s" "cpu" "-o" "nsight" "-f" "true"
        "--cudabacktrace=true" "--osrt-threshold=10000"
        "-x" "true" "python3" "gpu_profile.py"
        "--function_name" "$function_name"
        "--num_batches" "$num_batches"
        "--num_seq" "$num_seq"
        "--in_f" "$in_f"
        "--out_f" "$out_f"
        "--mode" "nsys"
        "--config_dir" "$config_dir"
    )

    if [[ "$is_compiled" == "True" || "$is_compiled" == "true" ]]; then
        command+=("--compile")
    fi

    if [[ "$function_name" == *"monarch"* || "$function_name" == *"blast"* ]]; then
        command+=("--b" "$b" "--rank" "$rank")
    fi

    if [[ "$function_name" == *"low_rank"* ]]; then
        command+=("--rank" "$rank")
    fi

    echo "Running nsys command: ${command[@]}"
    "${command[@]}"
    mv nsight.nsys-rep "$output_dir/${short_function_name}.nsys-rep"
}

ncu_command() {
    local num_batches=$1
    local num_seq=$2
    local in_f=$3
    local out_f=$4
    local function_name=$5
    local config_dir=$6
    local is_compiled=$7
    local b=${8:-1}
    local rank=${9:-0}

    local short_function_name="${function_name##*.}"

    # Detect Jetson device by checking for "NVIDIA Jetson" in tegra chip info
    if [[ -f /etc/nv_tegra_release ]]; then
        ncu_exec="sudo ncu --set full"
    else
        ncu_exec="ncu --set full"
    fi

    command=(
        $ncu_exec
        "-o" "ncu"
	"$(which python3)" "gpu_profile.py"
        "--function_name" "$function_name"
        "--num_batches" "$num_batches"
        "--num_seq" "$num_seq"
        "--in_f" "$in_f"
        "--out_f" "$out_f"
        "--mode" "ncu"
        "--config_dir" "$config_dir"
    )

    if [[ "$is_compiled" == "True" || "$is_compiled" == "true" ]]; then
        command+=("--compile")
    fi

    if [[ "$function_name" == *"monarch"* || "$function_name" == *"blast"* ]]; then
        command+=("--b" "$b" "--rank" "$rank")
    fi

    if [[ "$function_name" == *"low_rank"* ]]; then
        command+=("--rank" "$rank")
    fi

    echo "Running ncu command: ${command[@]}"
    "${command[@]}"
    mv ncu.ncu-rep "$output_dir/${short_function_name}.ncu-rep"
}

network_name=$1
layer_name=$2
config_file="./configs/config.yaml"
layer_config_file="./configs/layers/${network_name}.${layer_name}.yaml"
output_dir="./output/${network_name}/${layer_name}"

parse_and_merge_yaml "$config_file" "$layer_config_file"

convert_to_array blast_funcs_array "$triton_blast_benchmark_funcs"
convert_to_array blast_profile_array "$triton_blast_benchmark_profile"
convert_to_array blast_sym_quant_funcs_array "$triton_blast_sym_quant_benchmark_funcs"
convert_to_array blast_sym_quant_profile_array "$triton_blast_sym_quant_benchmark_profile"
convert_to_array monarch_funcs_array "$triton_monarch_benchmark_funcs"
convert_to_array monarch_profile_array "$triton_monarch_benchmark_profile"
convert_to_array low_rank_funcs_array "$triton_low_rank_benchmark_funcs"
convert_to_array low_rank_profile_array "$triton_low_rank_benchmark_profile"
convert_to_array dense_funcs_array "$triton_dense_benchmark_funcs"
convert_to_array dense_profile_array "$triton_dense_benchmark_profile"
convert_to_array torch_low_rank_funcs_array "$torch_low_rank_funcs"
convert_to_array torch_low_rank_profile_array "$torch_low_rank_profile"
convert_to_array torch_dense_funcs_array "$torch_dense_funcs"
convert_to_array torch_dense_profile_array "$torch_dense_profile"
convert_to_array torch_monarch_funcs_array "$torch_monarch_funcs"
convert_to_array torch_monarch_profile_array "$torch_monarch_profile"
convert_to_array torch_blast_funcs_array "$torch_blast_funcs"
convert_to_array torch_blast_profile_array "$torch_blast_profile"

if [[ -n "$triton_blast_benchmark_funcs" ]]; then
    for i in "${!blast_funcs_array[@]}"; do
        function_name="${blast_funcs_array[$i]}"
        profile_value="${blast_profile_array[$i]}"

        if [[ "$profile_value" == "True" || "$profile_value" == "true" ]]; then
            check_best_config "$function_name" || continue
            nsys_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" false "$blast_b" "$blast_rank"
            ncu_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" false "$blast_b" "$blast_rank"
        fi
    done
fi

if [[ -n "$triton_blast_sym_quant_benchmark_funcs" ]]; then
    for i in "${!blast_sym_quant_funcs_array[@]}"; do
        function_name="${blast_sym_quant_funcs_array[$i]}"
        profile_value="${blast_sym_quant_profile_array[$i]}"

        if [[ "$profile_value" == "True" || "$profile_value" == "true" ]]; then
            check_best_config "$function_name" || continue
            nsys_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" false "$blast_b" "$blast_rank"
            ncu_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" false "$blast_b" "$blast_rank"
        fi
    done
fi

if [[ -n "$triton_monarch_benchmark_funcs" ]]; then
    for i in "${!monarch_funcs_array[@]}"; do
        function_name="${monarch_funcs_array[$i]}"
        profile_value="${monarch_profile_array[$i]}"

        if [[ "$profile_value" == "True" || "$profile_value" == "true" ]]; then
            check_best_config "$function_name" || continue
            nsys_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" false "$monarch_b" "$monarch_rank"
            ncu_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" false "$monarch_b" "$monarch_rank"
        fi
    done
fi

if [[ -n "$triton_low_rank_benchmark_funcs" ]]; then
    for i in "${!low_rank_funcs_array[@]}"; do
        function_name="${low_rank_funcs_array[$i]}"
        profile_value="${low_rank_profile_array[$i]}"

        if [[ "$profile_value" == "True" || "$profile_value" == "true" ]]; then
            check_best_config "$function_name" || continue
            nsys_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" false 1 "$low_rank_rank"
            ncu_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" false 1 "$low_rank_rank"
        fi
    done
fi

if [[ -n "$triton_dense_benchmark_funcs" ]]; then
    for i in "${!dense_funcs_array[@]}"; do
        function_name="${dense_funcs_array[$i]}"
        profile_value="${dense_profile_array[$i]}"

        if [[ "$profile_value" == "True" || "$profile_value" == "true" ]]; then
            check_best_config "$function_name" || continue
            nsys_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" false
            ncu_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" false
        fi
    done
fi

for i in "${!torch_blast_funcs_array[@]}"; do   
    function_name="${torch_blast_funcs_array[$i]}"
    profile_value="${torch_blast_profile_array[$i]}"

    if [[ "$profile_value" == "True" || "$profile_value" == "true" ]]; then
        nsys_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" "$torch_blast_compile" "$blast_b" "$blast_rank"
        ncu_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" "$torch_blast_compile" "$blast_b" "$blast_rank"
    fi
done

for i in "${!torch_monarch_funcs_array[@]}"; do
    function_name="${torch_monarch_funcs_array[$i]}"
    profile_value="${torch_monarch_profile_array[$i]}"
    
    if [[ "$profile_value" == "True" || "$profile_value" == "true" ]]; then
        nsys_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" "$torch_monarch_compile" "$monarch_b" "$monarch_rank"
        ncu_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" "$torch_monarch_compile" "$monarch_b" "$monarch_rank"
    fi
done

for i in "${!torch_low_rank_funcs_array[@]}"; do
    function_name="${torch_low_rank_funcs_array[$i]}"
    profile_value="${torch_low_rank_profile_array[$i]}"
    
    if [[ "$profile_value" == "True" || "$profile_value" == "true" ]]; then
        nsys_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" "$torch_low_rank_compile" 1 "$low_rank_rank"
        ncu_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" "$torch_low_rank_compile" 1 "$low_rank_rank"
    fi
done

for i in "${!torch_dense_funcs_array[@]}"; do
    function_name="${torch_dense_funcs_array[$i]}"
    profile_value="${torch_dense_profile_array[$i]}"
    
    if [[ "$profile_value" == "True" || "$profile_value" == "true" ]]; then
        nsys_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" "$torch_dense_compile"
        ncu_command "$num_batches" "$num_seq" "$in_f" "$out_f" "$function_name" "$output_dir" "$torch_dense_compile"
    fi
done

echo "GPU profiling complete for network: $network_name, layer: $layer_name"
