# Persian Diffusion

![Gif](https://github.com/rQBx91/Persian_Diffusion/blob/main/result/persian/samples/gifs/gif1.gif)

A denoising diffusion probabilistic model for persian letters and digits. 

Project done as part of Probabilistic Graphical Models course at Khajeh Nasir Toosi University of Technology.

## Training
[Download](https://drive.google.com/file/d/17wi28DBfS_kmXhMNRqqpUdgWOJNhhON7/view?usp=sharing) persian letters and digits dataset.

Extract dataset and place it in the 'datasets' directory:

```bash
tar xvf persian_dataset.tar.xz
```

Install project requirements:

```bash
pip install -r requirements.txt
```

Start training :

```bash
python diffusion_model_persian
```
