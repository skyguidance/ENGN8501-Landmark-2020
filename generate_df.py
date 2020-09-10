import pandas as pd
from pathlib import Path
import cv2
import os
from tqdm import tqdm
from src import dataset_connector


def generate_size_info_df(paths, df) -> pd.DataFrame:
    for path in tqdm(list(paths)):
        id_ = str(path).split('/')[-1].replace('.jpg', '')
        img = cv2.imread(str(path))
        h, w, c = img.shape
        df.loc[id_, 'height'] = h
        df.loc[id_, 'width'] = w
    return df.reset_index().sort_values(by='id')


if __name__ == '__main__':
    print("[GLD2020-ENGN8501]Loading CSV....")
    df = pd.read_csv(dataset_connector.train_csv)
    print("[GLD2020-ENGN8501]Enumerating IMG path....")
    paths = Path(dataset_connector.train_dir).glob('**/*.jpg')
    df_path = pd.DataFrame(paths, columns=['path'])
    df_path['path'] = df_path['path'].apply(lambda x: str(x.absolute()))
    df_path['id'] = df_path['path'].apply(
        lambda x: x.split('/')[-1].replace('.jpg', ''))
    df = df.merge(df_path, on='id')
    paths = Path(dataset_connector.train_dir).glob('**/*.jpg')
    df = generate_size_info_df(paths, df)
    df.to_pickle("train.pkl")
