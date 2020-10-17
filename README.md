# Computational Intelligence  WIET AGH course 2020/2021

## Repository setup

#### 1. Create conda environment
```bash
conda create -n ComputationalIntelligence python=3.8
```

#### 2. Activate created environment
```bash
conda activate ComputationalIntelligence
```

#### 3. Install requirements
If the installation process failed due to ``SciPy`` building issue with fortran compiler - install ``gfortran`` package suitable for your OS.

```bash
(ComputationalIntelligence) repository_root$ python -m pip install -r requirements.txt
```

#### 4. Make the environment visible for jupyter notebook
```bash
(ComputationalIntelligence) python -m ipykernel install --user --name ComputationalIntelligence --display-name "ComputationalIntelligence"
```

#### 5. Start jupyter notebook
```bash
(ComputationalIntelligence) repository_root$ jupyter notebook
```


