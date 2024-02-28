# Draw SSM

## Setup

### Dependencies
Install NumPy, PyTorch, and [Mamba state-spaces](https://github.com/state-spaces/mamba)

### Dataset

#### gsutil
Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install) and follow the instructions to set it up

#### Setting up the project
Starting from the root of the project directory

```bash
mkdir data
cd data
gsutil -m cp 'gs://quickdraw_dataset/sketchrnn/*.npz' .
rm -r *.full.npz
```

You are now ready to use the dataset