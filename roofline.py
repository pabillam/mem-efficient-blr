import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

""" Low-Rank Memory/FLOPS"""
def torch_low_rank_baseline_mem_flops(n, i, r, o, num_bytes_per_element):
    return (
        n * i + i * r +
        n * r + r * o +
        n * r + n * o
    ) * num_bytes_per_element, n * r * (i + o) * 2

def triton_low_rank_mem_flops(n, i, r, o, num_bytes_per_element):
    return (
        i * r + r * o +
        n * o + n * i
    ) * num_bytes_per_element, n * r * (i + o) * 2

""" Dense Memory/FLOPS"""
def torch_dense_baseline_mem_flops(n, i, o, num_bytes_per_element):
    return (
        n * i + i * o +
        n * o
    ) * num_bytes_per_element, n * i * o * 2

def triton_dense_mem_flops(n, i, o, num_bytes_per_element):
    return torch_dense_baseline_mem_flops(n, i, o, num_bytes_per_element)

""" BLAST Memory/FLOPS"""
def torch_blast_baseline_mem_flops(n, i, r, o, b1, b2, num_bytes_per_element):
    return (
        n * i + b1 * b2 * r + 
        i * r + 4 * b2 * n * r + 
        4 * b1 * n * r + r * o +
        n * o
    ) * num_bytes_per_element, n * r * (i + o + b1 * b2) * 2

def triton_blast_partial_mem_flops(n, i, r, o, b1, b2, num_bytes_per_element):
    return (
        n * i + b1 * b2 * r + 
        i * r + 2 * b2 * n * r + 
        r * o + n * o
    ) * num_bytes_per_element, n * r * (i + o + b1 * b2) * 2

def triton_blast_bmm_int8_mem_flops(n, i, r, o, b1, b2, num_bytes_per_element):
    return (
        n * i + b1 * b2 * r + 
        i * r + 
        r * o + n * o
    ) * num_bytes_per_element + 2 * b2 * n * r, n * r * (i + o + b1 * b2) * 2

def triton_blast_partial_grouped_mem_flops(n, i, r, o, b1, b2, num_bytes_per_element):
    return triton_blast_partial_mem_flops(n, i, r, o, b1, b2, num_bytes_per_element)

def triton_blast_partial_grouped_persistent_mem_flops(n, i, r, o, b1, b2, num_bytes_per_element):
    return triton_blast_partial_mem_flops(n, i, r, o, b1, b2, num_bytes_per_element)

def triton_blast_full_mem_flops(n, i, r, o, b1, b2, num_bytes_per_element):
    return (
        n * i + b1 * b2 * r + 
        i * r + r * o + n * o
    ) * num_bytes_per_element, n * r * (i + o + b1 * b2) * 2

def triton_blast_bmm_mem_flops(n, i, r, o, b1, b2, num_bytes_per_element):
    return (
        n * i + b1 * b2 * r + 
        i * r + 2 * b2 * n * r + 
        2 * b1 * n * r + r * o +
        n * o
    ) * num_bytes_per_element, n * r * (i + o + b1 * b2) * 2

""" Monarch Memory/FLOPS"""
def triton_monarch_right_mem_flops(n, i, r, o, b, num_bytes_per_element):
    return (
        n * i + i * r +
        2 * b * n * r + 
        r * o + n * o +
        2 * n * o
    ) * num_bytes_per_element, n * r * (i + o) * 2

def triton_monarch_right_ideal_mem_flops(n, i, r, o, b, num_bytes_per_element):
    return (
        n * i + i * r +
        2 * b * n * r + 
        r * o + n * o
    ) * num_bytes_per_element, n * r * (i + o) * 2

def triton_monarch_right_left_mem_flops(n, i, r, o, b, num_bytes_per_element):
    return triton_monarch_right_ideal_mem_flops(n, i, r, o, b, num_bytes_per_element)

def torch_monarch_baseline_mem_flops(n, i, r, o, b, num_bytes_per_element):
    return (
        n * i + i * r + 
        2 * b * n * r + 
        2 * b * n * r + 
        r * o + n * o +
        2 * n * o
    ) * num_bytes_per_element, n * r * (i + o) * 2

def plot_roofline(layer, device):
    memory_bandwidth = device.memory_bandwidth
    peak_performance = device.peak_performance
    break_point = device.break_point
    n = layer.num_batches * layer.num_seq
    i = layer.in_f
    o = layer.out_f

    def performance(ai):
        return np.piecewise(ai, 
                            [ai <= break_point, ai > break_point],
                            [lambda x: memory_bandwidth * x, lambda x: peak_performance])

    def compute_runtime(flops, performance):
        return flops / performance

    arithmetic_intensity = np.linspace(0, 1000, 500)
    performance_values = performance(arithmetic_intensity)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})
    axes[0].plot(arithmetic_intensity, performance_values, label="Performance Model", color="blue")
    axes[0].axvline(break_point, color="green", linestyle="--", label=f"Break Point: {break_point:.2f} FLOPs/Byte")
    axes[0].axhline(peak_performance, color="red", linestyle="--", label=f"Peak Performance: {peak_performance:.2e} FLOPS")

    colormap = plt.cm.get_cmap('tab20')
    
    results = {}
    results["methods"] = []
    results["intensities"] = []
    results["flops"] = []
    results["mem_accesses"] = []
    results["runtimes"] = []
    results["performances"] = []
    colors = []

    for l in range(0, len(layer.function_names)):
        if "triton" in layer.function_names[l]:
            function = layer.function_names[l].rsplit("_", 1)[0]
        else:
            function = layer.function_names[l]

        mem_flops_function = globals()[f"{function}_mem_flops"]
        if "dense" in function:
            mem_accesses, flops = mem_flops_function(n, i, o, 2)

        elif "low_rank" in function:
            mem_accesses, flops = mem_flops_function(n, i, layer.low_rank["rank"], o, 2)
        elif "blast" in function:
            mem_accesses, flops = mem_flops_function(n, i, layer.blast["rank"], o, layer.blast["b"], layer.blast["b"], 2)
        elif "monarch" in function:
            mem_accesses, flops = mem_flops_function(n, i, layer.monarch["rank"], o, layer.monarch["b"], 2)
        intensity = flops / mem_accesses
        results["methods"].append(function)
        results["intensities"].append(intensity)
        results["performances"].append(performance(intensity))
        results["flops"].append(flops)
        results["mem_accesses"].append(mem_accesses)
        results["runtimes"].append(compute_runtime(flops, performance(intensity)))
        color = colormap(l / len(layer.function_names))
        colors.append(color)
        axes[0].scatter(intensity, performance(intensity), color=color, label=function, s=100)

    axes[0].set_xlabel("Arithmetic Intensity (FLOPs/Byte)")
    axes[0].set_ylabel("Performance (FLOPS)")
    axes[0].set_title(f"Nvidia {device.name} GPU Roofline Model")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].bar(results["methods"], results["runtimes"], color=colors)
    axes[1].set_ylabel("Runtime (s)")
    axes[1].set_title("Ideal Runtime Comparison of Methods")
    axes[1].grid(axis='y', linestyle='--')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    headers = ["Method", "Arithmetic Intensity", "Memory Accesses", "FLOPS", "Performance", "Runtime"]
    table = list(zip(results["methods"], results["intensities"], results["mem_accesses"], 
                     results["flops"], results["performances"], results["runtimes"]))

    table_str = tabulate(table, headers=headers, tablefmt="grid", floatfmt=(".2f", ".2e", ".2e", ".2e", ".2e", ".10f"))
    print(table_str)

    output_dir = f"output/{layer.network_name}/{layer.layer_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "roofline_summary.txt")
    
    with open(output_path, "w") as f:
        f.write(repr(device))
        f.write("\n")
        f.write(table_str)

    output_path_plot = os.path.join(output_dir, "roofline_plot.png")
    plt.savefig(output_path_plot, dpi=300, bbox_inches='tight')