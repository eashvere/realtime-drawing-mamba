# RealTime Drawing with Mamba

## Poster
![Poster](./figs/CSE_493G1_Poster.png)

## Paper
[Link to Paper PDF](./figs/Deep_Learning_Paper.pdf)

## Setup

### Project Dependencies
Make sure you have CUDA installed (>=11.6)

Install NumPy, PyTorch, and [Mamba state-spaces](https://github.com/state-spaces/mamba)
```bash
pip install -r requirements.txt
```

## Demo
Run the `draw.py` file to start the demo
```bash
python draw.py
```

Draw in the bottom canvas and view the model output in the top canvas. Use your mouse right click to clear the canvas.

## Training Dependencies

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

### Training
You can view our training code in `train.ipynb` and our model class in `customModel.py`. We followed much training parameters set in the [original Mamba paper by Albert Gu and Tri Dao](https://arxiv.org/abs/2312.00752)

