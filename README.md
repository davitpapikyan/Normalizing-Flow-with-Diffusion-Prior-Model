# Normalizing Flow With Diffusion Models as Prior

Install the requirements:
```console
pip install -r requirements.txt
```

Run a single training:
```console
python main.py hydra.job.chdir=True
```

Launch a dashboard to monitor current training or previous experiments:
```console
cd ./outputs/nf_experiments/aim
aim up
```

<!--
## References:
```
@misc{lippe2022uvadlc,
   title        = {{UvA Deep Learning Tutorials}},
   author       = {Phillip Lippe},
   year         = 2022,
   howpublished = {\url{https://uvadlc-notebooks.readthedocs.io/en/latest/}}
}
```
-->
