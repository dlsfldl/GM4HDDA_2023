# M3239.006800: Geometric Methods for High-Dimensional Data Analysis
The official repository for the homework assignments of the lecture <Geometric Methods for High-Dimensional Data Analysis> instructed by Frank C. Park in Seoul National University. 

#### Programming Exercise Submission Guide
Send your .ipynb file (and additional files if requested) as an email to **hdda2023@robotics.snu.ac.kr, not to the TAs' email addresses.**

- The title of your email should be in the form of "2021-33133 Gildong Hong HW1".

- The title of your file should be in the form of "2021-33123_HW1".

#### Table of Contents  
[Instructions for Settings](#Instructions-for-Settings)  
[Homework 1](#Homework-1)

#### TA Contacts
- Byeongho Lee (bhlee@robotics.snu.ac.kr, *Head TA*)
- Jihwan Kim (jihwankim@robotics.snu.ac.kr)
- Seokjin Choi (csj@robotics.snu.ac.kr)

## Instructions for Settings
### Install Anaconda3 and Jupyter Notebook
For those unfamiliar with environment setup, Anaconda3 installation is recommended. 
- For Windows user, download and install Anaconda3 from the following link: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution).
- For Linux user, download shell file from the following link: [Anaconda3-2022.05-Linux-x86_64.sh](https://drive.google.com/file/d/1x0mTd3stcNkEC_tY9vvgDUCwwHdPRqVe/view?usp=sharing). After that, execute the shell using the following command:
    ```shell
    chmod +x Anaconda3-2022.05-Linux-x86_64.sh      # change permission
    ./Anaconda3-2022.05-Linux-x86_64.sh             # execute shell file
    ```

After Anaconda3 installation, jupyter is automatically installed and you can run jupyter notebook with the following command:
```shell
jupyter notebook
```

### Guidelines for Anaconda3 and Jupyter Notebook
Here's some commands for conda environment:
```shell
conda create -n {name} python=3.9   # create conda environment
conda activate {name}               # activate conda environment
conda deactivate {name}             # deactivate conda environment
```
If you want to use the created conda environment in jupyter notebook, you need to register the kernel. The command to register is following:
```
pip install ipykernel
python -m ipykernel install --user --name {name} --display-name "{name-in-jupyter}"
```
- {name} and {name-in-jupyter} can be replaced by your choices.

### Environment
The project codes are tested in the following environment.
- python 3.9
- numpy
- matplotlib
- scikit-learn
- *pytorch*
- *torchvision*

To setup the environment except *pytorch* and *torchvision*, run the following script in the command line:
```
pip install -r requirements.txt
```
*Pytorch* and *torchvision* need to be installed separately using the method below.

### Torch installation
Install PyTorch and torchvision from the following link: [https://pytorch.org](https://pytorch.org). 

## Homework 1
Follow the instructions in the ``HW1.ipynb`` file. After you complete and run the HW ipython file, send the result file to ``hdda2023@robotics.snu.ac.kr ``.   
