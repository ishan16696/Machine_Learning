#!/usr/bin/env python
# coding: utf-8

# ### MultiVariate Linear Regression with L1 Regularization

# In[305]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[306]:


df=pd.read_csv("AdmissionDataset/data.csv")
df.head()


# **Drop the Serial No because it is irrelevant in prediction**

# In[307]:


df.drop('Serial No.',axis=1,inplace=True)


# In[308]:


df.head()


# In[309]:


training_data = df.sample(frac = 0.5, random_state = 200)
testing_data = df.drop(training_data.index)


# ### Feature Normalisation
# ${x_i}$= $\frac{x_i - \mu}{\sigma}$ 
# 
# Feature Normalisation is done because data in some columns is very small in comparison to other columns data.

# #### Preparing the Training data

# In[310]:


columns=training_data.shape[1]

X=training_data.iloc[:,0:columns-1]# features Sets

mu = X.mean()
sigma = X.std()

# features normalisation
X=(X-X.mean())/X.std()

Y=training_data.iloc[:,columns-1:columns] # outputSet
X.insert(0, 'Ones', 1)

print(Y.shape)
print(X.shape)
X.head()


# > Convert the X and Y into numpy matrix because we are going to do the vectorised Implementation

# In[311]:


X_train = np.matrix(X.values)
Y_train = np.matrix(Y.values)
print(X_train.shape)
print(Y_train.shape)


# #### Preparing the validation/Test data

# In[312]:


columns=testing_data.shape[1]

X=testing_data.iloc[:,0:columns-1]# features Sets

Y=testing_data.iloc[:,columns-1:columns] # outputSet

# features normalisation
X=(X-mu)/sigma


X.insert(0, 'Ones', 1)

X_test = np.matrix(X.values)
Y_test = np.matrix(Y.values)

print(X_test.shape)
print(Y_test.shape)

X.head()


# #### Mean Square Error with  L1 Regularization
# ${J(\theta)}$=${\frac{1}{2m}}{\sum_{i=0}^{m}}$(${\hat{y_i}-{y_i})^2 +  \frac{\lambda}{m}*\parallel \theta \parallel}$ 
# 
# 
# J: is cost function
# 
# 
# m : no. of training examples
# 
# ${\theta}$: parameters

# In[391]:


def costCompute_L2(X,Y,theta,lambd):
    j=0.0
    m=X.shape[0]
    
    err = np.power((np.dot(X,theta.T)-Y),2)
    j=np.sum(err)/(2*m)
    reg= (lambd/m)*np.sum(theta)
    
    return j+reg


# #### Gradient Descent algo
# repeat Untill Converges{
# 
# 
# ${{\theta_j} :=}{{\theta_j}}$-${\alpha}$*${\frac{\partial}{\partial {\theta_j}} J(\theta)}$
# 
# 
# }
# 
# ${\alpha}$: Learning rate constant

# In[392]:


#Vectorised Implementation
def gradientDescent(X, y, theta, alpha, iters,lambd):
    
    Jhistory=np.zeros(iters)
    ThetaforCoffiecients=np.zeros((iters,theta.shape[1]))
    reg_theta=theta
    
    
    m=X.shape[0]
    for i in range(iters):
        
        
        pre = np.dot(X,theta.T)-y
        reg_theta[theta>=0]=1
        reg_theta[theta<0]=-1
        
        temp=theta[0,0]-(alpha/m)*np.sum(pre)
        
        delta=np.dot(np.transpose(pre),X)
        theta=theta-(alpha/m)*(delta+lambd*reg_theta)
        
        theta[0,0]=temp
        
        ThetaforCoffiecients[i]=theta
        Jhistory[i] = costCompute_L2(X, y, theta,lambd)
        
        
        

    return theta,Jhistory,ThetaforCoffiecients


# 

# In[397]:


alpha=.01
iters=1000
lambd=0.001

theta = np.matrix(np.random.randn(1,X_train.shape[1]))

print(theta.shape)

minTheta, cost,Big= gradientDescent(X_train, Y_train, theta, alpha, iters,lambd)

print(minTheta.shape)
print(minTheta)


# 

# In[398]:


ig, ax = plt.subplots(figsize=(8,6))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Visualisation of change in Cost w.r.t to iterations')    


# In[ ]:



    


# In[399]:


def prediction_Error(X,Y,finalParameter):
    out= np.dot(X,finalParameter.T)
    
   
    err= np.sum(np.square(out-Y))/X.shape[0]
    
    return err
    


# In[ ]:





# In[400]:


prediction_Error(X_train,Y_train,minTheta)


# In[385]:


choice_lambda=[2,2.5,3,4,5,6,7,8,10,15,20,24,30,35,40]

err_train=[]
err_test=[]
Big_minTheta=np.zeros((len(choice_lambda),theta.shape[1]))
count=0
alpha=.009
iters=1000
for l in choice_lambda:
    theta = np.matrix(np.random.randn(1,X_train.shape[1]))*0.01  ## initialisation of theta
    
    ##training 
    minTheta,_,_= gradientDescent(X_train, Y_train, theta, alpha, iters,l)
    Big_minTheta[count]=minTheta
    count+=1
    err1=prediction_Error(X_train,Y_train,minTheta)
    err2=prediction_Error(X_test,Y_test,minTheta)
    err_train.append(err1)
    err_test.append(err2)


# In[388]:





# In[389]:


plt.rcParams['figure.figsize'] = [16, 8]
plt.scatter(choice_lambda,err_train,label="Training Error")
plt.plot(choice_lambda,err_train,'g')
plt.scatter(choice_lambda,err_test,label="Validation Error")
plt.plot(choice_lambda,err_test,'r')

plt.legend()

plt.xlabel('Regularisation Parameter')
plt.ylabel('Error')
plt.title('Training Error vs Reg. Parameter')


# In[381]:


print(err_train)
print(err_test)


# **Observations**:
# 
# <font color="purple" size="3">
# <ul>
#     <li>  The value of $\lambda$ is a hyperparameter that you can tune using a validation set.</li>
#  <li>    If ${\lambda}$ value is too small then it might leads to the overfit(high variance) which leads to high error on validation dataset and low error on train datasets.</li>
# <li>If ${\lambda}$ value is too large then it leads to the underfit(high bias) which leads to high error on validation dataset and high error on train datasets.</li>
#     </ul>
# </font
# 

# #### Part4

# In[401]:


plt.rcParams['figure.figsize'] = [10, 6]
l1=Big[:,0]
l2=Big[:,1]
l3=Big[:,2]
l4=Big[:,3]
l5=Big[:,4]
l6=Big[:,5]
l7=Big[:,6]
l8=Big[:,7]

it=range(0,iters)
plt.plot(it,list(l1),label="Bias")
plt.plot(it,list(l2),label="GRE")
plt.plot(it,list(l3),label="TOFL")

plt.plot(it,list(l4),label="Uni Rating")
plt.plot(it,list(l5),label="SOP")
plt.plot(it,list(l6),label="LOR")
plt.plot(it,list(l7),label="CGPA")
plt.plot(it,list(l8),label="Research")

plt.xlabel("Iterations")
plt.ylabel("Parameters")
plt.title("Effect of Regularization on parameters")
plt.legend()
plt.show()


# In[232]:


# plt.rcParams['figure.figsize'] = [10, 6]
# l1=Big_minTheta[:,0]
# l2=Big_minTheta[:,1]
# l3=Big_minTheta[:,2]
# l4=Big_minTheta[:,3]
# l5=Big_minTheta[:,4]
# l6=Big_minTheta[:,5]
# l7=Big_minTheta[:,6]
# l8=Big_minTheta[:,7]

# #it=range(0,len(choice_lambda)
# plt.plot(choice_lambda,list(l1),label="Bias")
# plt.plot(choice_lambda,list(l2),label="GRE")
# plt.plot(choice_lambda,list(l3),label="TOFL")

# plt.plot(choice_lambda,list(l4),label="Uni Rating")
# plt.plot(choice_lambda,list(l5),label="SOP")
# plt.plot(choice_lambda,list(l6),label="LOR")
# plt.plot(choice_lambda,list(l7),label="CGPA")
# plt.plot(choice_lambda,list(l8),label="Research")

# plt.xlabel("LAMBDA")
# plt.ylabel("Coefficients")
# plt.title("behaviour of the coefficients")
# plt.legend()
# plt.show()


# 
