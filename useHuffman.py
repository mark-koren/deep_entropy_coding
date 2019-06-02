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
    data_train = data.shape[0]//10*6
    data_cv = data.shape[0]//10*6
    data_test = data.shape[0]//10*8
    parsed_data=[]
    freq_count = np.zeros((data.shape[0],2**16))
    # pdb.set_trace()
    for i in range(data.shape[0]):
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

    cluster_array = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    size_array = np.zeros(len(cluster_array))
    for j in range(len(cluster_array)):
	    num_clusters = cluster_array[j]
	    kmeans = kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=1000, verbose=1, n_init=1).fit(freq_count[0:data_train])
	    labels = kmeans.predict(freq_count)
	    # pdb.set_trace()
	    cluster_sizes = np.zeros(num_clusters)
	    for cluster in range(num_clusters):
	        # a0 = np.argwhere(kmeans.labels_ == 0).squeeze().tolist()
	        # print("cluster ", cluster, ": ", np.argwhere(kmeans.labels_ == cluster).squeeze().tolist())
	        cluster_data = []
	        for i in np.argwhere(labels == cluster).reshape(-1).tolist():
                if i < data_test:
	        	  cluster_data += parsed_data[i*(data.shape[1]//16):(i+1)*(data.shape[1]//16)]

	        # cluster_data = [parsed_data[i*(data.shape[1]//16):(i+1)*(data.shape[1]//16)] for i in np.argwhere(labels == cluster).reshape(-1).tolist()]
	        # cluster_data = parsed_array(np.argwhere(kmeans.labels_ == cluster))
	        # pdb.set_trace()
	        h = HuffmanCoding('DJIEncoded.txt')
	        start = time.time()
	        h.create_coding_from_binary(cluster_data)
	        print("Coding created in ", time.time() - start, " seconds")
	        start = time.time()
	        encoded_array, size = h.get_encoded_array(cluster_data)
	        cluster_sizes[cluster] = size

	    print("Clustered: ", np.sum(cluster_sizes))
	    size_array[j] = np.sum(cluster_sizes) + data.shape[0] * np.log2(num_clusters)

    cluster_array = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    size_array = np.zeros(len(cluster_array))
    for j in range(len(cluster_array)):
	    num_clusters = cluster_array[j]
	    kmeans = kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=1000, verbose=1, n_init=1).fit(freq_count[0:data_train])
	    labels = kmeans.predict(freq_count)
	    # pdb.set_trace()
	    cluster_sizes = np.zeros(num_clusters)
	    for cluster in range(num_clusters):
	        # a0 = np.argwhere(kmeans.labels_ == 0).squeeze().tolist()
	        # print("cluster ", cluster, ": ", np.argwhere(kmeans.labels_ == cluster).squeeze().tolist())
	        cluster_data = []
	        for i in np.argwhere(labels == cluster).reshape(-1).tolist():
	        	cluster_data += parsed_data[i*(data.shape[1]//16):(i+1)*(data.shape[1]//16)]

	        # cluster_data = [parsed_data[i*(data.shape[1]//16):(i+1)*(data.shape[1]//16)] for i in np.argwhere(labels == cluster).reshape(-1).tolist()]
	        # cluster_data = parsed_array(np.argwhere(kmeans.labels_ == cluster))
	        # pdb.set_trace()
	        h = HuffmanCoding('DJIEncoded.txt')
	        start = time.time()
	        h.create_coding_from_binary(cluster_data)
	        print("Coding created in ", time.time() - start, " seconds")
	        start = time.time()
	        encoded_array, size = h.get_encoded_array(cluster_data)
	        cluster_sizes[cluster] = size

	    print("Clustered: ", np.sum(cluster_sizes))
	    size_array[j] = np.sum(cluster_sizes) + data.shape[0] * np.log2(num_clusters)

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
    print(size_array)
    print("Train unencoded: ", data[0:data_train, :].shape[1]*data_train*16)
    print("Test unencoded: ", data[data_test:-1, :].shape[1]*data[data_test, :].shape[1]*16)
    pdb.set_trace()
        # output_path, size = h.compress_from_binary(data)
        # print("Compressed in ", time.time() - start, " seconds")
        # print("Compressed file size: " + size)

        # decom_path = h.decompress(output_path)
        # print("Decompressed file path: " + decom_path)

