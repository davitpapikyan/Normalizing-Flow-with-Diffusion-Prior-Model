# Normalizing Flow With Diffusion Models as Prior

Install the requirements:
```console
pip install -r requirements.txt
```

To run the project:
```console
python main.py hydra.job.chdir=True
```

To open the dashboard with loss visualization and generated samples:
```console
cd ./outputs/nf_experiments/aim
aim up
```


## References:
```
@misc{lippe2022uvadlc,
   title        = {{UvA Deep Learning Tutorials}},
   author       = {Phillip Lippe},
   year         = 2022,
   howpublished = {\url{https://uvadlc-notebooks.readthedocs.io/en/latest/}}
}
```
