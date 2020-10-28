# ENGN8501 Advcaned Computer Vision Final Project - Group 24
# Google Landmark Challenge 2020
![](img\Banner.png)<br>
### Train:
* Download the 2020 Dataset, this is also known as GLDv2-Clean Dataset:https://www.kaggle.com/c/landmark-recognition-2020/data. The file downloaded is a zipped file at 97.6GB, so during the unzipping process it will take 2 times the space at most. Make sure you have enough space on your disk.
* Clone this repo
```
git clone https://github.com/skyguidance/ENGN8501-Landmark-2020.git
git checkout submission
```
* Modify the dataset connecter and connect the dataset on your drive by changing the "src/dataset_connector.py" file.
* Install Python package, and PyTorch. https://pytorch.org
```
pip install -U \
        tqdm \
        click \
        logzero \
        gensim \
        optuna \
        tensorboardX \
        scikit-image \
        lockfile \
        pytest \
        Cython \
        pyyaml \
        scikit-learn \
        numpy \
        pandas \
        scipy \
        faiss-gpu \
        tqdm \
        opencv-python \
        joblib \
        seaborn \
        pretrainedmodels \
        plotly \
        albumentations \
        line-profiler \
        tabulate \
        easydict
```
* Train the network from scratch. The devices switch works as a CUDA visible mask. When training, the log will automatically write to the result folder, and the model will be automatically saved on every epoch.
```
python train.py --devices "0,1,2,......."
```
### Predict
```
python predict.py --devices "0" --model-path "/your/path/to/your/model/pth/file"
```
### Pretrained Models:
UPLOADING.

### Collaborators:
Ranked in alphabetical order, with equal contribution.


* Shuaiqun Pan <br>
* Tianyi Qi<br>
* Yijian Fan<br>
* Yufeng Fang<br>
