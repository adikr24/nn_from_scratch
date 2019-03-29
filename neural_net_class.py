import math
import numpy as np

        

X1 = 5
X1= np.array([X1])
#W1= np.random.rand(X1.shape[0])
W1= np.array([0.3])
A1= W1.dot([X1])
y_pred= sigmoid(A1)
Error = y_act - y_pred
lr =0.005
print(y_pred,W1)
y_act=1

##################################class 
#### 5 layer NN
import numpy as np
X= np.array([[3],[2]])

W1 = np.random.random((3,2))
A1= sigmoid(W1.dot(X))

W2= np.random.random((5,3))
A2= sigmoid(W2.dot(A1))

W3= np.random.random((3,5))
A3= sigmoid(W3.dot(A2))

W4 = np.random.random((2,3))
A4= sigmoid(W4.dot(A3))

W5= np.random.random((1,2))
Y_pred= sigmoid(W5.dot(A4))


Y_act= 1


## backprop

E= Y_act- Y_pred

DE_Y_pred = E * - Y_pred * sd(Y_pred)
DE_DW5 = DE_DY_pred * A4.T

DE_DA4= W5.T.dot(E * - Y_pred * sd(Y_pred)) * sd(A4)
DE_DW4= DE_DA4 * A3.T
assert(W4.shape ==DE_DW4.shape)

DE_DA3 = W4.T.dot(W5.T.dot(E * - Y_pred * sd(Y_pred)) * sd(A4)) *sd(A3)
DE_DW3= DE_DA3 * A2.T

DE_DA2 =W3.T.dot (W4.T.dot(E * -Y_pred *sd(Y_pred)*W5.T *sd(A4)) * sd(A3)) * sd(A2)
DE_DW2= DE_DA2 * A1.T

DE_DA1= W2.T.dot(W2.dot (W4.T.dot(E * -Y_pred *sd(Y_pred)*W5.T *sd(A4)) * sd(A3)) * sd(A2)) * sd(A1)
DE_DW1 =DE_DA1 * X.T

#### backprop_II
DE_Y_pred = E * - Y_pred *sd(Y_pred)
DE_DW5 = DE_DY_pred * A4.T

DE_DA4= W5.T.dot(DE_Y_pred)*sd(A4)
DE_DW4= DE_DA4 * A3.T

DE_DA3= W4.T.dot(DE_DA4)* sd(A3)
DE_DW3= DE_DA3* A2.T

DE_DA2= W3.T.dot(DE_DA3)*sd(A2)
DE_DW2 = DE_DA2 * A1.T

DE_DA1= W2.T.dot(DE_DA2) * sd(A1)
DE_Dw1= DE_DA1 * X.T



for i in range(10):
    DE_Y_pred = E * - Y_pred *sd(Y_pred)
    DE_DW5 = DE_Y_pred * A4.T
    DE_DA4= W5.T.dot(DE_Y_pred)*sd(A4)
    DE_DW4= DE_DA4 * A3.T
    DE_DA3= W4.T.dot(DE_DA4)* sd(A3)
    DE_DW3= DE_DA3* A2.T
    DE_DA2= W3.T.dot(DE_DA3)*sd(A2)
    DE_DW2 = DE_DA2 * A1.T
    DE_DA1= W2.T.dot(DE_DA2) * sd(A1)
    DE_DW1= DE_DA1 * X.T
    W5 = W5 - lr * DE_DW5
    W4 = W4- lr* DE_DW4
    W3 = W3 - lr* DE_DW3
    W2 = W2 - lr* DE_DW2
    W1 = W1 - lr* DE_DW1
    A1= sigmoid(W1.dot(X))
    A2= sigmoid(W2.dot(A1))
    A3= sigmoid(W3.dot(A2))
    A4= sigmoid(W4.dot(A3))
    A5= sigmoid(W5.dot(A4))
    E= Y_act - A5
    
    difference= abs(Y_act- A5)
    print(E)  

################################################################################



import numpy as np

class nn:
    def __init__(self,input_layer, hidden_layer, output_size,Y_act,learning_rate):
        self.input_layer= input_layer
        self.hidden_layer = hidden_layer
        self.output_size= output_size
        self.Y_act= Y_act
        self.weights_dict={}
        self.act_layer= {}
        self.backprop_A={}
        self.backprop_W={}
        self.learning_rate= learning_rate
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def sd(self,x):
        return self.sigmoid(x)* (1-self.sigmoid(x))
    
    def initweights(self):
        for i in range(len(self.hidden_layer)+1):
           if i ==0:
               ww= np.random.random((self.hidden_layer[i],self.input_layer.shape[0]))
               self.weights_dict.update({'W'+str(i+1):ww})
           elif i > 0:
               try:
                   ww= np.random.random((self.hidden_layer[i],self.hidden_layer[i-1]))
                   self.weights_dict.update({'W'+str(i+1):ww})
               except IndexError:
                   ww= np.random.random((self.output_size,self.hidden_layer[i-1]))
                   self.weights_dict.update({'W'+str(i+1):ww})
    
    def activation(self):
        for i in range(len(self.hidden_layer)+1):
            if i==0:
                A= self.weights_dict['W'+str(i+1)].dot(self.input_layer)
                A= self.sigmoid(A)
                self.act_layer.update({'A'+str(i):A})
            elif i >0:
                A = self.weights_dict['W'+str(i+1)].dot(self.act_layer['A'+str(i-1)])
                A= self.sigmoid(A)
                self.act_layer.update({'A'+str(i):A})         
    
    def error(self):
        Y_pred=self.act_layer['A'+str(len(self.hidden_layer))]
        E = self.Y_act - Y_pred
        return abs(E)
   
    def returnweights(self):
       return self.weights_dict

    def returnactive(self):
       return self.act_layer
    
    def backprop(self):
        Y_pred= self.act_layer['A'+str(len(self.hidden_layer))]
        E = self.Y_act-Y_pred
        for i in range(len(self.hidden_layer),-1,-1):
            if i == len(self.hidden_layer):
                DE_DA= E*-Y_pred *self.sd(self.act_layer['A'+str(i)])
                self.backprop_A.update({'DE_DA'+str(i):DE_DA})
            elif i<len(hidden_layer):
                try:
                    DE_DA = self.weights_dict['W'+str(i+2)].T.dot(self.backprop_A['DE_DA'+str(i+1)]) * sd(activation_l['A'+str(i)])
                    self.backprop_A.update({'DE_DA'+str(i):DE_DA})
                except IndexError:
                    pass

        for i in range(len(self.hidden_layer),-1,-1):
            if i ==len(self.hidden_layer):
                DE_DW = self.backprop_A['DE_DA'+str(i)] * self.act_layer['A'+str(i-1)].T
                self.backprop_W.update({'DE_DW'+str(i+1):DE_DW})
            elif i <= len(self.hidden_layer):
                try:
                    DE_DW = self.backprop_A['DE_DA'+str(i)] * self.act_layer['A'+str(i-1)].T
                    self.backprop_W.update({'DE_DW'+str(i+1):DE_DW})
                except KeyError:
                    DE_DW = self.backprop_A['DE_DA'+str(i)]*self.input_layer.T
                    self.backprop_W.update({'DE_DW'+str(i+1):DE_DW})

    
    def updateweights(self):
        for i in range(len(weights_dict)):
            self.weights_dict['W'+str(i+1)] = self.weights_dict['W'+str(i+1)]- self.learning_rate * self.backprop_W['DE_DW'+str(i+1)]
        
        
    def activation_updated(self):
        return self.backprop_A
    
    def return_weights(self):
        return self.weights_dict
    

X=np.array([[1],[2]])
inputs=X
X.shape[0]
hidden_layer =[4,3,5,7,2]
output_size= 2
Y_act= [[1],[0]]
learning_rate= 0.05
mm = nn(X,hidden_layer,output_size,Y_act,learning_rate)
mm.initweights()
mm.activation()
weights_dict= mm.returnweights()
activation_l=mm.returnactive()


#mm.activation_updated()

for i in range(100):
    mm.backprop()
    mm.updateweights()
    mm.activation()
    print(mm.error())
#################################testing the neural net 


#################################testing the neural net 




















            









