from huffman import HuffmanCoding
import sys
import time
import pickle
import numpy as np
from sklearn.cluster import KMeans
import pdb

path = "sample.txt"

# widths = [2, 4, 8, 16]
# for width in widths:
#     all_possible_characters = []
#     for i in range(2**width):
#         all_possible_characters.append(np.binary_repr(i, width=width))
#     with open('./DJITotal' + str(width) + '.p', 'wb') as f:
#         pickle.dump(all_possible_characters, f)

# pdb.set_trace()
# h = HuffmanCoding(path)
width = 16
with open('./DJIEncoded.p', 'rb') as f, open(
    './DJIParsed' + str(width) + '.p', 'rb') as f2, open(
    'DJIFreq' + str(width) + '.p', 'rb') as f3, open(
    './DJITotal' + str(width) + '.p', 'rb') as f4:
    data = pickle.load(f)
    parsed_data = pickle.load(f2)
    freq_count = pickle.load(f3)
    all_possible_characters = pickle.load(f4)
    # width = 8 
    # data = pickle.load(f)
    data_train = data.shape[0]//10*6
    data_cv = data.shape[0]//10*6
    data_test = data.shape[0]//10*8
    # parsed_data=[]
    # freq_count = np.zeros((data.shape[0],2**width))
    # # pdb.set_trace()
    # for i in range(data.shape[0]):
    #     print("Converting encoded image ", i)
    #     for j in range(data.shape[1]//width):
    #         text_array = ""
    #         index = 0
    #         for k in range(width):
    #             index += 2**((width-1)-k)*data[i,width*j+k]
    #             text_array += str(data[i,width*j+k])
    #             # print(index)
    #         # print(text_array)
    #         # pdb.set_trace()
    #         parsed_data.append(text_array)
    #         freq_count[i,index] += 1
    # # pdb.set_trace()
    # total_freq_count = np.sum(freq_count, axis=0)
    # for i in range(total_freq_count.shape[0]):
    #     if total_freq_count[i] == 0:
    #         parsed_data.append(np.binary_repr(i, width=width))

    total_freq_count = np.sum(freq_count, axis=0)
    parsed_array = np.array(parsed_data)
    trials = 100
    start = time.time()
    cluster_array = np.arange(16).tolist()[2:-1]
    size_data = np.zeros((trials, len(cluster_array), 3))
    for trial in range(trials):
    # cluster_array = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        
        size_array = np.zeros((len(cluster_array), 3))
        freq_count_adjusted = (10 * freq_count) + 1
        for j in range(len(cluster_array)):
            num_clusters = cluster_array[j]
            kmeans = kmeans = KMeans(
                n_clusters=num_clusters, 
                 max_iter=1000, 
                 verbose=0, 
                 n_init=10).fit(freq_count[0:data_train])
            labels = kmeans.predict(freq_count)
            # pdb.set_trace()
            cluster_sizes = np.zeros((num_clusters,3))
            for cluster in range(num_clusters):
                # a0 = np.argwhere(kmeans.labels_ == 0).squeeze().tolist()
                # print("cluster ", cluster, ": ", np.argwhere(kmeans.labels_ == cluster).squeeze().tolist())
                cluster_data_train = []
                cluster_data_cv = []
                cluster_data_test = []
                for i in np.argwhere(labels == cluster).reshape(-1).tolist():
                    if i < data_train:
                        cluster_data_train += parsed_data[i*(data.shape[1]//16):(i+1)*(data.shape[1]//16)]
                    elif i < data_test:
                        cluster_data_cv += parsed_data[i*(data.shape[1]//16):(i+1)*(data.shape[1]//16)]
                    else:
                        cluster_data_test += parsed_data[i*(data.shape[1]//16):(i+1)*(data.shape[1]//16)]



                # cluster_data = [parsed_data[i*(data.shape[1]//16):(i+1)*(data.shape[1]//16)] for i in np.argwhere(labels == cluster).reshape(-1).tolist()]
                # cluster_data = parsed_array(np.argwhere(kmeans.labels_ == cluster))
                # pdb.set_trace()
                h = HuffmanCoding('DJIEncoded.txt')
                start = time.time()
                h.create_coding_from_binary(10*cluster_data_train+all_possible_characters)
                print("Coding created in ", time.time() - start, " seconds")
                start = time.time()
                encoded_array, size = h.get_encoded_array(cluster_data_train)
                cluster_sizes[cluster, 0] = size
                encoded_array, size = h.get_encoded_array(cluster_data_cv)
                cluster_sizes[cluster, 1] = size
                encoded_array, size = h.get_encoded_array(cluster_data_test)
                cluster_sizes[cluster, 2] = size

            cluster_sum = np.sum(cluster_sizes, axis=0)
            print("Clustered: ", cluster_sum)
            size_array[j, 0] = cluster_sum[0] + data_train * np.log2(num_clusters)
            size_array[j, 1] = cluster_sum[1] + data_cv * np.log2(num_clusters)
            size_array[j, 2] = cluster_sum[2] + data_test * np.log2(num_clusters)
        print(size_array)
        size_data[0,...] = size_array
        print("\nTrial ", trial, " completed in: ", time.time() - start, "\n")
        start = time.time()

    # cluster_sizes = np.zeros(num_clusters)
    # for cluster in range(num_clusters):
    #     # a0 = np.argwhere(kmeans.labels_ == 0).squeeze().tolist()
    #     # print("cluster ", cluster, ": ", np.argwhere(kmeans.labels_ == cluster).squeeze().tolist())
    #     cluster_data = [parsed_data[i+data_test] for i in np.argwhere(labels[data_test:-1] == cluster).reshape(-1).tolist()]
    #     # cluster_data = parsed_array(np.argwhere(kmeans.labels_ == cluster))
    #     h = HuffmanCoding('DJIEncoded.txt')
    #     start = time.time()
    #     h.create_coding_from_binary(cluster_data)
    #     print("Coding created in ", time.time() - start, " seconds")
    #     start = time.time()
    #     encoded_array, size = h.get_encoded_array(cluster_data)
    #     cluster_sizes[cluster] = size

    # print("Clustered (Test only): ", np.sum(cluster_sizes))
    # pdb.set_trace()

    # h = HuffmanCoding('DJIEncoded.txt')
    # start = time.time()
    # h.create_coding_from_binary(parsed_data)
    # print("Coding created in ", time.time() - start, " seconds")
    # start = time.time()
    # encoded_array, size = h.get_encoded_array(parsed_data)
    # cluster_sizes[cluster] = size

    # print("Total: ", size)
    
    # print(size_array2)
    # print("Train unencoded: ", data[0:data_train, :].shape[1]*data_train*16)
    # print("Test unencoded: ", data[data_test:-1, :].shape[1]*data[data_test, :].shape[1]*16)
    with open('Kmeans_results' + str(width) + '.p', 'wb') as fw:
        pickle.dump(size_data, fw)
    pdb.set_trace()
        # output_path, size = h.compress_from_binary(data)
        # print("Compressed in ", time.time() - start, " seconds")
        # print("Compressed file size: " + size)

        # decom_path = h.decompress(output_path)
        # print("Decompressed file path: " + decom_path)

