# Machine Learning Nanodegree
# Capstone Project
## Project: Vidyabagish

In this project we will focus on Text Generation. Text Generation is part of [Natural Language Processing](https://en.wikipedia.org/wiki/Natural_language_processing) and can be used to [transcribe speech to text](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf), perform [machine translation](http://arxiv.org/abs/1409.3215), generate handwritten text, image captioning, generate new blog posts or news headlines. 

RNNs are [very effective](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) when understanding sequence of elements and have been used in the past to generate text. I will use a Recurrent Neural Network to generate text inspired on the works of Vidyabagish.

![Basic RNN -> Unrolled RNN](images/basic_unrolled_RNN.png)

In order to generate text, we will look at a class of Neural Network where connections between units form a directed cycle, called Recurrent Neural Network (RNNs). RNNs use an internal memory to process sequences of elements and is able to learn from the syntactic structure of text. Our model will be able to generate text based on the text we train it with.

### Datasets

The datasets used in this repository are included in the dataset folder. It is not necessary to do any manual preprocessing of data as the files included are already clean.

### Install

Highly recommend installing [Anaconda](https://www.continuum.io/downloads). Anaconda conveniently installs Python, the Jupyter Notebook, and other commonly used packages for scientific computing and data science.

This project has the following dependencies:
- Python 3.6
- Jupyter Notebook
- Numpy

After installing Anaconda, open a terminal window and navigate to the directory where the repository is in your local machine.


```bash
cd /path/to/Vidyabagish
```

If you want to install the packages manually:
```bash
conda create --name Vidyabagish -y python=3 numpy
source activate Vidyabagish
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp36-cp36m-linux_x86_64.whl
jupyter notebook
```

Alternatively, you can use the conda environment file included in the repository
```bash
conda env create -f Vidyabagish.yml
jupyter notebook
```

### Run 

Open the [Jupyter](http://jupyter.org/install.html) notebook - mlnd-Vidyabagish.ipynb

### Clean up

To remove the environment just run

```bash
source deactivate
conda remove --name Vidyabagish --all -y
```



