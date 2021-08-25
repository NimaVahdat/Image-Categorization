from utils.loader import Loader
from utils.model import DeepSNN
import torch
import os

def feature_extraction(prop):
    name = prop["name"]
    epochs_l1 = prop["epochs_l1"]
    epochs_l2 = prop["epochs_l2"]
    trainset, testset = Loader(name)
    
    model = DeepSNN(prop)
    
    # Training The First Layer
    print("-------Training the first layer-------")
    if os.path.isfile(name+"_Layer1.net"):
     	model.load_state_dict(torch.load(name+"_Layer1.net"))
     	print("Loaded from disck!")
    else:
    	for epoch in range(epochs_l1):
    		print("Epoch:", epoch)
    		for data,_ in trainset:
    			model.train_model(data, 1)               
    			print("\nDone!")

    	torch.save(model.state_dict(), name+"_Layer1.net")
    
    # Training The Second Layer
    print("-------Training the second layer-------")
    if os.path.isfile(name+"_Layer2.net"):
     	model.load_state_dict(torch.load(name+"_Layer2.net"))
     	print("Loaded from disck!")
    else:
        for epoch in range(epochs_l2):
        	print("Epoch:", epoch)
        	for data,_ in trainset:
        		model.train_model(data, 2)
        		print("\nDone!")
        		
        torch.save(model.state_dict(), name+"_Layer2.net")
    
    # Classification on trainset and testset
    # Get train data
    for data,target in trainset:
    	train_X, train_y = model.test(data, target, 2)
    	
    
    # Get test data
    for data,target in testset:
    	test_X, test_y = model.test(data, target, 2)
    
    return train_X, train_y, test_X, test_y, (model.conv1.weight, model.conv2.weight)
    
def Classification(train_X, train_y, test_X, test_y, C=2.4): 
    # SVM
    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=C)
    clf.fit(train_X, train_y)
    predicted_train = clf.predict(train_X)
    predicted_test = clf.predict(test_X)
    
    return predicted_train, predicted_test
    
def performance(x, y, predict):
	correct = 0
	silence = 0
	for i in range(len(predict)):
		if x[i].sum() == 0:
			silence += 1
		else:
			if predict[i] == y[i]:
				correct += 1
	return (correct/len(x), (len(x)-(correct+silence))/len(x), silence/len(x))

def confussion_matrix(test_y, predicted_test, labels):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(test_y, predicted_test)
    cmd_obj = ConfusionMatrixDisplay(cm, display_labels=labels)
    # print(cm)
    cmd_obj.plot()
    
    plt.show()    

# %%
Caltech = { "name" : "Caltech",
            "epochs_l1" : 20,
            "epochs_l2" : 100,
            "weight_mean" : 0.8,
              "weight_std" : 0.05,
              "lr" : (0.005, -0.0025),
              "in_channel1" : 4,
              "in_channel2" : 40,
              "out_channel" : 150,
              "k1" : 10,
              "k2" : 25,
              "r1" : 0,
              "r2" : 2,}

train_X, train_y, test_X, test_y, weights = feature_extraction(Caltech)

predicted_train, predicted_test = Classification(train_X, train_y, test_X, test_y)

n = performance(train_X, train_y, predicted_train)
m = performance(test_X, test_y, predicted_test)
print(n)
print(m)

labels = ['Airplane', 'Car_side', 'Faces_easy', 'Motorbikes']
confussion_matrix(test_y, predicted_test, labels)

# %%
MNIST = {"name" : "MNIST",
         "epochs_l1":2,
         "epochs_l2":20,
         "weight_mean" : 0.8,
         "weight_std" : 0.05,
         "lr" : (0.004, -0.003),
         "in_channel1" : 2,
         "in_channel2" : 32,
         "out_channel" : 150,
         "k1" : 5,
         "k2" : 8,
         "r1" : 2,
         "r2" : 1,}

train_X, train_y, test_X, test_y, weights = feature_extraction(MNIST)

predicted_train, predicted_test = Classification(train_X, train_y, test_X, test_y)

n = performance(train_X, train_y, predicted_train)
m = performance(test_X, test_y, predicted_test)
print(n)
print(m)

labels = ['0','1','2','3','4','5','6','7','8','9']
confussion_matrix(test_y, predicted_test, labels)

# %%
# import cv2
# import numpy as np
# w1, w2 = weights
# w1 = torch.reshape(w1, (160, 5, 5))
# # w2 = torch.reshape(w2, (6000, 2, 2))


# def features_pic(w, i):
#     # w = torch.squeeze(w)
#     w -= w.min()
#     w = (w/w.max()) * 255
#     pic = cv2.resize(np.array(w), (100, 100))
#     cv2.imwrite("features/feature" + str(i) + ".jpg", pic)


# for i in range(len(w1)):
#     features_pic(w1[i], i)