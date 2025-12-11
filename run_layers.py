import argparse
from layer import Layer
from device import Device
from roofline import plot_roofline
import os

def main():
    parser = argparse.ArgumentParser(description="Run benchmarking and profiling for a given network layer on a specified device.")
    parser.add_argument("--device", required=True, help="Name of the device (e.g., A40).")
    parser.add_argument("--network", required=True, help="Name of the network (e.g., llama7b).")
    parser.add_argument("--layer", required=True, help="Name of the layer (e.g., gate_up_proj).")

    args = parser.parse_args()

    d = Device(args.device)
    print(d) 

    layer_config_path = f"./configs/layers/{args.network}.{args.layer}.yaml"
    l = Layer(layer_config_path)
    print(l)

    output_dir = f"output/{args.network}/{args.layer}"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "device_summary.txt")
    with open(output_path, "w") as f:
        f.write(repr(d))
    
    output_path = os.path.join(output_dir, "layer_summary.txt")
    with open(output_path, "w") as f:
        f.write(repr(l))   

    l.benchmark("all")
    l.verify("all")
    l.summary()
    plot_roofline(l, d)

if __name__ == "__main__":
    main()
