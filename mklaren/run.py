import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data, sort_XyZ
from mklaren.kernel.kernel import linear_kernel, poly_kernel, rbf_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.align import Align 
from mklaren.mkl.alignf import Alignf
from mklaren.mkl.mklaren import Mklaren


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score

# from mklaren.kernel.kinterface_combineK import KinterfaceCombine  # Test this

# #Load data
# path = '/Users/macbook/Documents/WorkSpace/Experiments_TKL/netNorm-PY/data/AD_CN/'
# modalities = ['CSF.csv', 'GM.csv', 'PET.csv', 'SNP.csv'] 

# X_list = []

# for modality in modalities:
#     data = load_data(path= path + modality, input=True)
#     X_list.append(data)
# y = load_data(path + 'AD_CN_label.csv', input=False)

# X_list, y,Z = sort_XyZ(X_list, y, None)

# print("X_shape: ", X_list[1].shape)
# print("y shape: ", y.shape)




def classify(kernel, y):
    clf = SVC(kernel='precomputed')
    scores = cross_val_score(clf, kernel, y, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()

# map title to modalities
title = {'CSF': 0, 'GM': 1, 'PET': 2, 'SNP': 3}

def run(config):
    # Load data
    path = '/Users/macbook/Documents/WorkSpace/Experiments_TKL/netNorm-PY/data/AD_CN/'
    modalities = ['CSF.csv', 'GM.csv', 'PET.csv', 'SNP.csv']
    X_list = []

    print("Loading data...")
    for modality in modalities:
        data = load_data(path + modality, input=True)
        X_list.append(data)
    y = load_data(path + 'AD_CN_label.csv', input=False)

    # Sort data
    Z = np.array(range(len(y)))
    X_list, y, Z = sort_XyZ(X_list, y, Z)

    # Create kernel matrices
    K_list_linear = []
    K_list_poly = []
    K_list_rbf = []
    K_select = []
    if config['kernel'] == 'linear':
        print("Creating linear kernel matrices...")
        for i,X in enumerate(X_list):
            K = Kinterface(data=X, kernel=linear_kernel)
            K_list_linear.append(K)
            K_select.append(K)
    elif config['kernel'] == 'poly':
        print("Creating polynomial kernel matrices...")
        for X in X_list:
            K = Kinterface(data=X, kernel=poly_kernel, kernel_args={'degree': 2})
            K_list_poly.append(K)
            K_select.append(K)
    elif config['kernel'] == 'rbf':
        print("Creating RBF kernel matrices...")
        for X in X_list:
            K = Kinterface(data=X, kernel=rbf_kernel, kernel_args={'sigma': 1})
            K_list_rbf.append(K)
            K_select.append(K)
    
    # Save the kernel matrices
    # for i,K_option in enumerate(K_select):
    #     K_matrix = K_option[:,:] # Extract the kernel matrix
    #     np.savetxt("results/rbf_kernel_matrix_" + str(i) + ".csv", K_matrix, delimiter=',')

    # Align kernel matrices
    if config['combine'] == 'align':
        align = Align(d=2)

        align.fit(K_select, y)
        K = align.Kappa
        #Visualize the aligned kernel matrix
        # plt.imshow(K)
        # plt.show()
        #===============================================
        # save the aligned kernel matrix
        # np.savetxt("results/aligned_kernel_rbf.csv", K, delimiter=',')
    elif config['combine'] == 'alignf':
        alignf = Alignf(typ="linear")

        alignf.fit(K_select, y)
        K = alignf.Kappa
        #Visualize the aligned kernel matrix
        # plt.figure(figsize=(6, 6))
        # plt.imshow(K, cmap='viridis', interpolation='nearest')
        # plt.colorbar()
        # plt.title('Alignf Kernel Matrix Linear')
        # plt.show()
        #===============================================
        # save the aligned kernel matrix
        # np.savetxt("results/alignF_kernel_rbf.csv", K, delimiter=',')
    
    else : pass


    model = Mklaren(rank=15, lbd=1, delta=30)
    model.fit([K_select[0], K_select[1], K_select[2], K_select[3]], y)

    #:ivar G: (``numpy.ndarray``) Stacked Cholesky factors.
    G_1 = model.data[0]["G"]
    G_2= model.data[1]["G"]
    G_3 = model.data[2]["G"]
    G_4 = model.data[3]["G"]

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 4))
    for i, (name, K, G) in enumerate(zip([f"{str(config['kernel'])}Kernel_CSF", f"{str(config['kernel'])}Kernel_GM",
                                        f"{str(config['kernel'])}Kernel_PET", f"{str(config['kernel'])}Kernel_SNP"], 
                                [K_select[0], K_select[1], K_select[2], K_select[3]], 
                                [G_1, G_2, G_3, G_4])):
        ax[0, i].set_title(name)
        ax[0, i].imshow(K[:, :])
        ax[1, i].imshow(G.dot(G.T))

    ax[0, 0].set_ylabel("Original")
    ax[1, 0].set_ylabel("Mklaren")
    fig.tight_layout()
    #===============================================
    # Save figures

    plt.show()

    # combined kernel to use in classification
    sumMKLaren_kernel = G_1.dot(G_1.T) + G_2.dot(G_2.T) + G_3.dot(G_3.T) + G_4.dot(G_4.T)
    sumMKLaren_kernel = sumMKLaren_kernel / 4
    #===============================================
    # save the sum MKLaren kernel matrix
    # np.savetxt("results/sumMKLaren_kernel.csv", sumMKLaren_kernel, delimiter=',')


    return K


if __name__ == '__main__':
    config = {
        'kernel': 'rbf',
        'combine': 'align'
    }


    run(config)
    # print(run(config))







    



