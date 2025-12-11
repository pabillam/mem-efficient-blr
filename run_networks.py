import argparse
from network import Network

def main():
    parser = argparse.ArgumentParser(description="Run benchmarking and profiling for a given network.")
    parser.add_argument("--network", required=True, help="Name of the network", choices=["llama7b", "llama1b", "gpt2s", "dit_xl", "vit_b_patch16_224"])
    parser.add_argument("--methods", nargs='+', required=False, help="List of methods", default=["blast", "blast_sym_quant", "monarch", "low_rank", "dense"])
    args = parser.parse_args()

    config_path = f"configs/networks/{args.network}.yaml"
    n = Network(config_path)
    n.benchmark(args.methods)
    n.summary()
        
if __name__ == "__main__":
    main()
