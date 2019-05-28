from huffman import HuffmanCoding
import sys
import time
import pickle
import numpy as np
from sklearn.cluster import KMeans
import pdb

path = "sample.txt"

# h = HuffmanCoding(path)
with open('./DJIEncoded.p', 'rb') as f: 
    data = pickle.load(f)
    parsed_data=[]
    freq_count = np.zeros((data.shape[0],2^16))
    for i in range(data.shape[0]//10):
        print("Converting encoded image ", i)
        for j in range(data.shape[1]//16):
            text_array = ""
            index = 0
            for k in range(16):
            	index += 2^(15-k)*data[i,16*j+k]
                text_array += str(data[i,16*j+k])
            parsed_data.append(text_array)
            freq_count[i,index] += 1

    parsed_array = np.array(parsed_data)


    kmeans = KMeans(n_clusters=16, random_state=0).fit(freq_count)

    pdb.set_trace()

    h = HuffmanCoding(None)
    start = time.time()
    h.create_coding_from_binary(parsed_data)
    print("Coding created in ", time.time() - start, " seconds")
    start = time.time()
    output_path = h.compress_from_binary()
    print("Compressed in ", time.time() - start, " seconds")
    print("Compressed file path: " + output_path)

    decom_path = h.decompress(output_path)
    print("Decompressed file path: " + decom_path)

