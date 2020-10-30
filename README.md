# ENGN8501 Advanced Computer Vision Final Project - Group 24
# Google Landmark Challenge 2020
![](https://cloud.google.com/vision/docs/images/moscow.png?hl=it)<br>
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
Pure Arcface Model:

https://engn8501.ap-south-1.linodeobjects.com/Epoch29_arcface.pth

Arcface + Attention Model:

https://engn8501.ap-south-1.linodeobjects.com/ep29_Attention_Arcface.pth


### Collaborators:
Ranked in alphabetical order, with equal contribution.


* Shuaiqun Pan <br>
* Tianyi Qi<br>
* Yijian Fan<br>
* Yufeng Fang<br>

### Reference lists:
1. Implement Additive Margin Softmax Loss from https://github.com/Leethony/Additive-Margin-Softmax-Loss-Pytorch.
2. Implement Large-margin Softmax(L-Softmax) from https://github.com/amirhfarzaneh/lsoftmax-pytorch/blob/master/lsoftmax.py.
3. Implement Arcface loss function from https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py.
4. Implement Cosface loss function from https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py.
5. Implement Spatial attention mechanism from https://github.com/rainofmine/Face_Attention_Network/blob/master/model_level_attention.py
