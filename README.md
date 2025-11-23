# CSDI
This is the github repository for the NeurIPS 2021 paper "[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)".

## Requirement

Please install the packages in requirements.txt

## Preparation
### Download the healthcare dataset 
```shell
python download.py physio
```
### Download the air quality dataset 
```shell
python download.py pm25
```

### Download the elecricity dataset 
Please put files in [GoogleDrive](https://drive.google.com/drive/folders/1krZQofLdeQrzunuKkLXy8L_kMzQrVFI_?usp=drive_link) to the "data" folder.

## Experiments 

### training and imputation for the healthcare dataset
```shell
python exe_physio.py --testmissingratio [missing ratio] --nsample [number of samples]
```

### imputation for the healthcare dataset with pretrained model
```shell
python exe_physio.py --modelfolder pretrained --testmissingratio [missing ratio] --nsample [number of samples]
```

### training and imputation for the healthcare dataset
```shell
python exe_pm25.py --nsample [number of samples]
```

### training and forecasting for the electricity dataset
```shell
python exe_forecasting.py --datatype electricity --nsample [number of samples]
```

### Traffic dataset

Brief: trajectory datasets are handled by `exe_traffic.py`. Trajectories are stored as time-indexed positions, normalized by training-set mean/std, and represented as sequences (T, features). The code provides optional physics-aware training and projection:

- **Preprocessing:** per-feature mean/std computed on training sims; missing values handled with masks.
- **Physics loss:** kinematic soft constraint (finite-difference velocity/acceleration residuals) enabled with `--use_physics` and weighted by `--lambda_phys`.
- **Projection:** a post-sampling kinematic projection can be applied with `--use_projection` to map generated trajectories back to a feasible manifold.
- **CLI example (train + impute):**
```shell
python exe_traffic.py --testmissingratio 0.2 --nsample 50 --use_physics --lambda_phys 0.1 --use_projection
```

### Fluid (Kaggle / PDEBench)
```shell
python exe_fluid_kaggle.py  --nsample 10 --use_projection
```


### Visualize results
'visualize_examples.ipynb' is a notebook for visualizing results.

## Acknowledgements

A part of the codes is based on [BRITS](https://github.com/caow13/BRITS) and [DiffWave](https://github.com/lmnt-com/diffwave)

## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{tashiro2021csdi,
  title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
