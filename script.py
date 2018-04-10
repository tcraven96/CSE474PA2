import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X, y):
	# Inputs
	# X - a N x d matrix with each row corresponding to a training example
	# y - a N x 1 column vector indicating the labels for each training example
	#
	# Outputs
	# means - A k x d matrix containing learnt means for each of the k classes
	# covmat - A single d x d learnt covariance matrix

	# IMPLEMENT THIS METHOD

	N = len(X)
	d = len(X[0])
	k = int(y.max())

	means = np.zeros((int(k), int(d)))
	sum = np.zeros((1, d))
	count = 0
	covmat = np.zeros((int(d), int(d)))

	for cls in range(int(k)):
		for array in range(N):
			if y[array] == cls + 1:
				sum[0, :] = sum[0, :] + X[array, :]
				count = count + 1
		means[cls, :] = sum[0, :] / count
		sum[:, :] = 0
		count = 0

	total_mean = np.zeros((1, int(d)))
	# print(total_mean.shape)
	total_mean[0, :] = np.mean(X, axis=0)
	# print(total_mean.shape)
	ssum = np.zeros((d, d))
	for array in range(N):
		A = np.zeros((1, int(d)))
		A[0, :] = X[array, :] - total_mean
		ssum = ssum + np.dot((A).T, (A))
	covmat = ssum / N

	return means, covmat


def qdaLearn(X, y):
	# Inputs
	# X - a N x d matrix with each row corresponding to a training example
	# y - a N x 1 column vector indicating the labels for each training example
	#
	# Outputs
	# means - A k x d matrix containing learnt means for each of the k classes
	# covmats - A list of k d x d learnt covariance matrices for each of the k classes

	# IMPLEMENT THIS METHOD

	N = len(X)
	d = len(X[0])
	k = int(y.max())

	means = np.zeros((k, d))
	sum = np.zeros((1, d))
	count = 0
	covmats = np.zeros((k, d, d))  # 3-dimension

	for cls in range(k):
		for array in range(N):
			if y[array] == cls + 1:
				sum[0, :] = sum[0, :] + X[array, :]
				count = count + 1
		means[cls, :] = sum[0, :] / count
		sum[:, :] = 0
		count = 0

	ssum = np.zeros((d, d))
	for cls in range(k):
		for array in range(N):
			A = np.zeros((1, int(d)))
			if y[array] == cls + 1:
				A[0, :] = X[array, :] - means[cls, :]
				ssum = ssum + np.dot(A.T, A)
				count = count + 1
		covmats[cls, :, :] = ssum[:, :] / count
		ssum[:, :] = 0
		count = 0

	return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
	# Inputs
	# means, covmat - parameters of the LDA model
	# Xtest - a N x d matrix with each row corresponding to a test example
	# ytest - a N x 1 column vector indicating the labels for each test example
	# Outputs
	# acc - A scalar accuracy value
	# ypred - N x 1 column vector indicating the predicted labels

	# IMPLEMENT THIS METHOD

	k = len(means)
	N = len(Xtest)
	d = len(Xtest[0])
	P = np.zeros((int(k), 1))
	ypred = np.zeros((int(N), 1))

	for i in range(N):
		for j in range(k):
			B = np.zeros((1, int(d)))
			B[0, :] = Xtest[i, :] - means[j, :]
			# P[j] = (Xtest[i, :] - means[j, :]) * inv(covmat) * np.transpose(Xtest[i, :] - means[j, :])
			# print(covmat.shape)
			# print(B.shape)
			P[j] = np.dot(B, np.dot(inv(covmat), B.T))
		# print(P[j])
		pred_cls = np.argmin(P)
		# print(pred_cls)
		ypred[i] = pred_cls + 1
	# print(ypred)

	acc = np.mean((ypred == ytest).astype(float)) * 100

	return acc, ypred


def qdaTest(means, covmats, Xtest, ytest):
	# Inputs
	# means, covmats - parameters of the QDA model
	# Xtest - a N x d matrix with each row corresponding to a test example
	# ytest - a N x 1 column vector indicating the labels for each test example
	# Outputs
	# acc - A scalar accuracy value
	# ypred - N x 1 column vector indicating the predicted labels

	# IMPLEMENT THIS METHOD

	k = len(means)
	N = len(Xtest)
	d = len(Xtest[0])
	P = np.zeros((int(k), 1))
	ypred = np.zeros((int(N), 1))

	for i in range(N):
		for j in range(k):
			B = np.zeros((1, int(d)))
			B[0, :] = Xtest[i, :] - means[j, :]
			# print(covmats[j,:,:])
			# print(det(covmats[j,:,:]))
			# print(sqrt(det(covmats[j,:,:])))
			P[j] = (1 / sqrt(det(covmats[j, :, :]))) * np.exp(-np.dot(B, np.dot(inv(covmats[j, :, :]), B.T)) / 2)
		cls = np.argmax(P)
		ypred[i] = cls + 1

	acc = np.mean((ypred == ytest).astype(float)) * 100

	return acc, ypred

def learnOLERegression(X,y):
    # Inputs:`
    # X = N x d 43
    # y = N x 1                                                               
    # Output: 
    # w = d x 1
    w = np.dot(np.dot((inv(np.dot(X.T,X))), X.T),y)
	
    # IMPLEMENT THIS METHOD                                                   
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    length = len(X[0])

    w = (np.dot(np.dot(inv(lambd*np.eye(length)+np.dot(X.T,X)),X.T),y))
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # mse
    N = len(Xtest)
    mse = np.dot((ytest - np.dot(Xtest,w)).T,(ytest -np.dot(Xtest,w)))/N
    # IMPLEMENT THIS METHOD
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    w1 = w.reshape(-1, 1)
    error = np.dot((y-np.dot(X,w1)).T, (y-np.dot(X,w1)))/2 + ((lambd/2)*np.dot(w1.T, w1))
    error_grad = np.dot(np.transpose(X), (np.dot(X,w1)-y)) + (lambd * w1)
    error = error.flatten()
    error_grad = error_grad.flatten()
    # IMPLEMENT THIS METHOD
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1))
    x1 = x.reshape(-1, 1)
    N = len(x1)
    recur = x1
    if(p==0) :
	    Xp = np.ones((N, 1))
    elif ( p > 0 ) :
	    Xp = np.ones((N, 1))
	    Xp = np.append(Xp, x1, 1)
	    for i in range(p) :
		    recur = recur * x1
		    Xp = np.append(Xp, recur, 1)
    # IMPLEMENT THIS METHOD
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.flatten())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.flatten())
plt.title('QDA')

plt.show()

# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))

for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
	args = (X_i, y, lambd)
	w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
	w_l = np.transpose(np.array(w_l.x))
	w_l = np.reshape(w_l,[len(w_l),1])
	mses4_train[i] = testOLERegression(w_l,X_i,y)
	mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
	i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
	print(p)
	Xd = mapNonLinear(X[:,2],p)
	Xdtest = mapNonLinear(Xtest[:,2],p)
	w_d1 = learnRidgeRegression(Xd,y,0)
	mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
	mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
	w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
	mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
	mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
