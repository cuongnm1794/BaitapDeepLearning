import multiprocessing as mp
from bing_image_downloader import downloader
import os
from tqdm import tqdm
import time
import numpy as np
from sklearn.preprocessing import Binarizer
import itertools
from matplotlib import pyplot as plt

import bai1_downloadFile
import bai2_readFile
import bai3_chuanhoa


path_dataset = "dataset/"
label_list = ["Tulip","Orchid","Carnation","Lily","Rose"] 
list_image_width = []
list_image_height = []

def Bai1_Download(label_list):
    pool = mp.Pool(mp.cpu_count() - 1 )
    result = [ pool.apply(bai1_downloadFile.download_dataset,args=(row,)) for row in label_list ]
    pool.close()

def Bai2_ReadData(label_list):
    """
    Hàm đọc 1 mẫu, cho ID và tìm đọc:
    Ví dụ input: Rose_1 => File nằm: dataset/Rose/Image_1.jpg
    """

    # Lấy đường dẫn từ id
    dir_label = label_list.split("_")
    path_data = "dataset/" + ("_".join(dir_label[:-1])) + "/Image_" + dir_label[-1] + ".jpg"
    result = bai2_readFile.ReadImageData(path_data)
    return [np.array(result), ("_".join(dir_label[:-1]))]
   
def Bai2_ReadData_Mul(label):
    """
    Hàm đọc tất cả mẫu của 1 label
    Kết hợp đa tiến trình
    Ví dụ input: Rose => File nằm: dataset/Rose/
    """

    #get list file name
    images = os.listdir("dataset/"+label)

    pool = mp.Pool(mp.cpu_count() - 1 )
    result_mul = [  pool.apply_async(bai2_readFile.ReadImageData,args=(os.path.join(path_dataset,label,row),)).get() for row in tqdm(images,desc='Read data '+label) ]
    pool.close()
    pool.join()
    result = []
    return [result_mul,label]

def Bai3_ChuanHoa(data_input):
    """
    Hàm này dùng để chuẩn hóa dữ liệu
    input: [data,label] thô
    output: [data,label] đã chuẩn hóa
    """
    
    # Lấy data hình ảnh mở bằng open cv
    # Chuẩn hóa data
    data = [x[2] for x in data_input[0]]
    pool = mp.Pool(mp.cpu_count() - 1 )
    data_chuanhoa = [  pool.apply_async(bai3_chuanhoa.PreprocessingData,args=(row,)).get() for row in tqdm(data,desc='Chuan Hoa '+data_input[1]) ]
    pool.close()
    pool.join()

    #Chuẩn hóa label
    
    labels = [x for x in data_input[1]]
    label_chuanhoa = bai3_chuanhoa.ChuanHoaLabel(labels)


    return [data_chuanhoa,label_chuanhoa]

def Bai4_Xuly(data_input):
    """
    Hàm này in số liệu phân tích hình ảnh gồm:
    - Số lượng mẫu
    - his chiều cao, chiều rộng

    Dữ liệu đầu vào: 
    - 1 list gồm [[width,height,image],label]

    """

    #Lấy chiều cao và rộng
    list_width = [x[0] for x in data_input[0] ]
    list_height = [x[1] for x in data_input[0] ]

    #print("Number of label: {}".format(len(data_input[0])))
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Hist of {} ({})'.format(data_input[1],len(data_input[0])), fontsize=16)
    ax0, ax1 = axes.flatten()
    ax0.hist(list_width)
    ax0.set_title('Hist width: '+data_input[1])

    ax1.hist(list_height)
    ax1.set_title('Hist height: '+data_input[1])
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ====> Bài 1
    #
    #Bai1_Download(label_list)
    for label in label_list:
        # ====> Bài 2
        result_bai2 = Bai2_ReadData_Mul(label)

        # ====> Bài 3
        result_bai3 = Bai3_ChuanHoa(result_bai2) 

        # ====> Bài 4
        Bai4_Xuly(result_bai2)
