import multiprocessing as mp
from bing_image_downloader import downloader
import os
from tqdm import tqdm

def download_dataset(label_find):
        downloader.download(label_find,limit = 1000,output_dir ='dataset')
        print("Finish download: ",label_find)