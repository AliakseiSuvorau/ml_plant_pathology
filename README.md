# Plant Pathology 2020 - FGVC7

https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/overview

---
## How to create PyEnv?

```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

## How to install modules?

```bash
pip install -r ./requirements.txt
```

## Run training

```bash
python train.py
```

The checkpoint is saved to `checkpoint.pth`.

## Run test

```bash
python test.py
```

Creates a file `submission.txt`.
