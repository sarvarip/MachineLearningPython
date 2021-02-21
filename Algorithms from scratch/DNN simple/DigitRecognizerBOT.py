import argparse
import numpy as np
import pandas as pd

class linear_layer:
    def __init__(self, input_D, output_D):
        '''
        W is of shape input_D-by-output_D
        '''
        self.params = dict()
        self.params['W'] = np.random.normal(0, 0.1, (input_D, output_D))
        self.params['b'] = np.random.normal(0, 0.1, (1, output_D))
        self.gradient = dict()
        self.gradient['W'] = np.zeros((input_D, output_D))
        self.gradient['b'] = np.zeros((1, output_D))
        self.name = 'Linear layer'

    def forward(self, X):
        forward_output = np.dot(X,self.params['W'])+self.params['b']
        return forward_output

    def backward(self, X, grad):
        self.gradient['W'] = np.dot(X.T, grad)
        self.gradient['b'] = np.sum(grad, axis=0)
        backward_output = np.dot(grad, self.params['W'].T)
        return backward_output

class relu:
    def __init__(self):
        self.mask = None
        self.name = 'ReLu layer'

    def forward(self, X):
        self.mask = (X > 0.0)
        forward_output = np.maximum(0.0, X)

        return forward_output

    def backward(self, X, grad):
        backward_output = np.multiply(self.mask, grad)
        return backward_output

class dropout:
    def __init__(self, r):
        self.r = r
        self.mask = None
        self.name = 'Dropout layer'

    def forward(self, X, is_train):
        if is_train:
            self.mask = (np.random.uniform(0.0, 1.0, X.shape) >= self.r).astype(float) * (1.0 / (1.0-self.r))
        else:
            self.mask = np.ones(X.shape)
        forward_output = np.multiply(X, self.mask)
        return forward_output

    def backward(self, X, grad):
        backward_output = np.multiply(grad, self.mask)
        return backward_output

class softmax:
    def __init__(self):
        self.yexpand = None
        self.prob = None

    def forward(self, X, Y):
        y = Y.astype(int).values.reshape(-1) 
        xdim = 10 #Hard-coded, it is known that we have 10 classes
        ydim = len(y)
        self.yexpand = np.zeros((ydim,xdim))
        self.yexpand[np.arange(ydim),y] = 1
        logit_norm = X-np.amax(X, axis=1, keepdims = True)
        sum_exp_logit_norm = np.sum(np.exp(logit_norm), axis=1, keepdims = True)
        self.prob = np.exp(logit_norm) / sum_exp_logit_norm
        forward_output = -np.sum(np.multiply(self.yexpand, logit_norm-np.log(sum_exp_logit_norm))) / X.shape[0]
        return forward_output

    def backward(self, X, Y):
        backward_output = -(self.yexpand-self.prob) / X.shape[0]
        return backward_output

def evaluate(y_pred,y_test):
    '''
    Get the accuracy of the classification problem
    '''
    N = y_pred.shape[0]
    y_pred = y_pred.reshape(-1)
    y_test = y_test.values.reshape(-1)
    acc = np.sum(y_pred == y_test) / N * 100
    return acc

class Model:
    def __init__(self,lr,hn,bs,rate):
        '''
        lr: learning rate
        hn: number of neurons in hidden layer
        bs: batch size
        rate: dropout rate
        '''
        self.lr = lr 
        self.bs = bs 
        self.L1 = linear_layer(784, hn) #hard-coded, dimension known
        self.L2 = relu()
        self.L3 = dropout(rate)
        self.L4 = linear_layer(hn, 10) #hard-coded, number of classes known
        self.softmax = softmax()
        
    def train(self,X_train,y_train,X_test):
        '''
        Forward propagate, backward propagate and 
        update weights via gradient descent.
        Checking accuracy after each epoch to 
        facilitate possible early stopping.
        '''
        #print('Training model...')
        N_train = X_train.shape[0]
        N_test = X_test.shape[0]
        reduce_at = 50
        num = 100 #1000
        
        for t in range(num): #epochs
            
            if (t % reduce_at == 0) and (t != 0):
                self.lr = self.lr * 0.5
            print('Starting epoch ' + str(t) + ' / ' + str(num))
            train_epoch_perm = np.random.permutation(N_train)
            prop = int(np.floor(N_train / self.bs))
            
            for i in range(prop):
                
                random_index = train_epoch_perm[i*self.bs : (i+1)*self.bs]
                x0 = X_train.iloc[random_index, :]
                y = y_train.iloc[random_index, :]
                #x0 = X_train.iloc[i*self.bs : (i+1)*self.bs, :]
                #y = y_train.iloc[i*self.bs : (i+1)*self.bs, :]
                x1 = self.L1.forward(x0)
                x2 = self.L2.forward(x1)
                x3 = self.L3.forward(x2,True)
                x4 = self.L4.forward(x3)
                l = self.softmax.forward(x4,y)
                
                delta_4 = self.softmax.backward(x4,y)
                delta_3 = self.L4.backward(x3, delta_4)
                delta_2 = self.L3.backward(x2, delta_3)
                delta_1 = self.L2.backward(x1, delta_2)
                delta_x = self.L1.backward(x0, delta_1)
                
                modules = [self.L1, self.L4] #only L1 and L4 weights can be updated
                #layernum = 0
                for module in modules:
                    #layernum = layernum + 1 
                    for key, _ in module.params.items():
                        module.params[key] = module.params[key] - self.lr * module.gradient[key]
                        #print('Key ' + key + ' in module ' + module.name + '_' + str(layernum) + ' was updated.')
            
            y_pred = self.predict(X_train)
            acc = evaluate(y_pred, y_train)
            print('Training accuracy after epoch ' + str(t) + ' is ' + str(acc) + ' %')
            
        return 0
                    
    def predict(self,X_test):
        '''
        Forward propagate testing data
        '''
        #print('Predicting output...')
        x1 = self.L1.forward(X_test)
        x2 = self.L2.forward(x1)
        x3 = self.L3.forward(x2,False)
        x4 = self.L4.forward(x3)
        y_pred = np.argmax(x4, axis=1)
        return y_pred

def main(main_params):
    train_im_path = main_params['X_train']
    train_label_path = main_params['y_train']
    test_im_path = main_params['X_test']
    
    X_train = pd.read_csv(train_im_path, header=None)
    y_train = pd.read_csv(train_label_path, header=None)
    X_val = pd.read_csv(test_im_path, header=None)
    
    lr = 0.001
    hn = 1000
    bs = 10
    rate = 0.5
    
    mod = Model(lr,hn,bs,rate)
    mod.train(X_train,y_train,X_val)
    y_pred = mod.predict(X_val)
    
    #y_pred = y_pred.astype(int)
    #np.savetxt("test_predictions.csv", y_pred, delimiter=",")
    y = pd.DataFrame(y_pred, dtype=int)
    y.to_csv('test_predictions.csv', header=False, index=False)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('X_train', default='train_image.csv')
    parser.add_argument('y_train', default='train_label.csv')
    parser.add_argument('X_test', default='test_image.csv')
    args = parser.parse_args()
    main_params = vars(args)
    main(main_params)