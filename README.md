<div align="center">
 
# Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

**[Pierre Abillama](https://scholar.google.com/citations?user=bjIqxTIAAAAJ&hl=en&oi=ao), [Changwoo Lee](http://changwoolee.github.io), [Joy Dong](https://joydddd.github.io), [David Blaauw](https://blaauw.engin.umich.edu), [Dennis Sylvester](https://sylvester.engin.umich.edu) and [Hun-Seok Kim](https://kim.engin.umich.edu)**

University of Michigan

**[[Paper](https://arxiv.org/abs/2512.20861)]**

</div>

## Notice
This repo is being actively updated.
* [arXiv](https://arxiv.org/abs/2512.20861) version is available!

## Dependencies

We recommend installing dependencies within a Python virtual environment (`venv`) to ensure correct package isolation. 

```bash
python3 -m venv venv
source venv/bin/activate 
```

With the environment active (you should see `(venv)` in your prompt), install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage and Examples

```bash
python3 run_layers.py  --device A40 --network llama7b --layer gate_up_proj
python3 run_networks --network llama7b --methods blast monarch low_rank dense
```

## Citation

Please cite our paper if you find this repo or our paper useful
```
@misc{abillama2025arxiv,
      title={Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs}, 
      author={Pierre Abillama and Changwoo Lee and Juechu Dong and David Blaauw and Dennis Sylvester and Hun-Seok Kim},
      year={2025},
      eprint={2512.20861},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.20861}, 
}
```