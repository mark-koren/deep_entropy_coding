from huffman import HuffmanCoding
import sys
import time
import pickle
import numpy as np
import pdb

path = "sample.txt"

# h = HuffmanCoding(path)
with open('./DJIEncoded.p', 'rb') as f: 
    data = pickle.load(f)
    parsed_data=[]
    for i in range(data.shape[0]//10):
        print("Converting encoded image ", i)
        for j in range(data.shape[1]//16):
            text_array = ""
            for k in range(16):
                text_array += str(data[i,16*j+k])
            parsed_data.append(text_array)

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