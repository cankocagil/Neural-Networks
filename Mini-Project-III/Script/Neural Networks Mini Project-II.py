
# Necessary imports :
import numpy as np
import matplotlib.pyplot as plt
import h5py
# %%
# Necessary imports :
#import numpy as np
#import matplotlib.pyplot as plt
#import h5py
import math
import pandas as pd 
import seaborn as sns
import sys

question = input('Please enter question number 1/3 :')





def can_kocagil_21602218_hw3(question):
    if question == '1' :
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%


        # %%
        def get_data(path) -> dict :
            """
            Given the path of the dataset, return
            training and testing images with respective
            labels.
            """
            with h5py.File(path,'r') as F:
                # Names variable contains the names of training and testing file 
                names = list(F.keys())

                data = np.array(F[names[0]][()])
                invXForm = np.array(F[names[1]][()])
                xForm = np.array(F[names[2]][()])
                      
            return {'data'    : data,
                    'invXForm': invXForm,
                    'xForm'   : xForm}


        path = 'assign3_data1.h5'
        data_h5 = get_data(path)


        # %%
        data = data_h5['data']
        invXForm = data_h5['invXForm']
        xForm = data_h5['xForm']


        # %%
        print(f'The data has a shape: {data.shape}')


        # %%
        data = np.swapaxes(data,1,3)


        # %%
        print(f'The data has a shape: {data.shape}')


        # %%
        class ImagePreprocessing:
          """
          _____Image preprocessor_____
          
          Functions :
          --- ToGray(data)
              -Takes an input image then converts to gray scale by Luminosity Model

          --- MeanRemoval(data)
              -Extracking the mean of each image themselves

          --- ClipStd(data)
              - Clipping the input image within given condition

          --- Normalize(data,min_scale,max_scale)
              - Normalizing input image to [min_scale,max_scale]

          --- Flatten(data)
              - Flattening input image
          """
          def __init__(self):
            pass
              
          def ToGray(self,data):
            """
            Given the input image converting gray scale according to luminosity model
            """
            R_linear = 0.2126
            G_linear = 0.7152
            B_linear = 0.0722
            gray_data = (data[:,:,:,0] * R_linear) + (data[:,:,:,1] * G_linear) + (data[:,:,:,2] * B_linear)    

            return gray_data 

          def MeanRemoval(self,data):
            """
              Given the input image, substracking the mean of pixel intensity of each image
            """
            axis = (1,2)
            mean_pixel = np.mean(data,axis = axis)
            num_samples = data.shape[0]
            
            # Substracking means of each image seperately :
            for i in range(num_samples):
              data[i] -= mean_pixel[i]
            return data

          def ClipStd(self,data,std_scaler):
            """
            Given the data and range of standart deviation scaler,
            return clipped data
            """
            std_pixel = np.std(data)
            
            min_cond = - std_scaler * std_pixel
            max_cond =   std_scaler * std_pixel

            clipped_data = np.clip(data,min_cond,max_cond)

            return clipped_data

          def Normalize(self,data,min_scale,max_scale):
            """
            Given the data, normalize to given interval [min_val,max_val]
            """
            min = data.max()
            max = data.min()

            # First normalize in [0,1]
            norm_data = (data - min) / (max-min)

            # Normalizing in [min_scale,max_scale]
            range = max_scale - min_scale
            interval_scaled_data = (norm_data * range) + min_scale
            
            return interval_scaled_data

          def Flatten(self,data):
            """
            Given the input image data returning flattened version of the data
            """
            num_samples = data.shape[0]
            flatten  = data.reshape(num_samples,-1)

            return flatten


        # %%
        # Defining preprocessor :
        preprocessor = ImagePreprocessing()


        # %%
        # Converting gray scale :
        gray_data = preprocessor.ToGray(data = data)


        # %%
        # Mean removing :
        mean_removed_data = preprocessor.MeanRemoval(data = gray_data)


        # %%
        # Standart deviation clipping :
        clipped_data = preprocessor.ClipStd(data = mean_removed_data,std_scaler = 3)


        # %%
        # Normalized data
        data_processed = preprocessor.Normalize(data = clipped_data, min_scale = 0.1, max_scale = 0.9)


        # %%
        print(f' Maximum val of data :  {data_processed.max()}')
        print(f' Minimum val of data :  {data_processed.min()}')


        # %%
        def plot_patches(data,num_patches, cmap = 'viridis'):
            np.random.seed(15)
            num_samples = data.shape[0]
            random_indexes = np.random.randint(num_samples,size = num_patches)  
            
            plt.figure(figsize = (18,16))
            for i in range(num_patches):
                plt.subplot(20,20,i+1)
                plt.imshow(data[random_indexes[i]],cmap = cmap)
                plt.axis('off')
            plt.show()


        # %%
        plot_patches(preprocessor.Normalize(data = data, min_scale = 0, max_scale = 1),num_patches = 200)


        # %%
        #plot_patches(data,num_patches = 200)


        # %%
        plot_patches(data_processed,num_patches = 200, cmap = 'gray')


        # %%
        class Autoencoder:
            """
            ____Autoencoder____

            Functions :
            --- __init__(input_size,hidden_size)
                - Building overall architecture of the model

            --- InitParams(input_size,hidden_size)
                - Initializing configurable parameters

            --- aeCost(W,data,params)
                - Calculating cost and it's derivatives

            --- Forward(X)
                - Forward pass

            --- Backward(X)
                - Calculation of gradients w.r.t. loss function

            --- KL_divergence()
                - Calculate KL divergence and it's gradients

            --- TykhonowRegulator(X,grad)
                - Computing Tykhonow regularization term and it's gradient

            --- Predict(X)
                - To make predictions

            --- Sigmoid(X, grad)
                - Compute sigmoidal activation and it's gradients

            --- History()
                - To keep track history of the model
            """

            def __init__(self,input_size,hidden_size,lambd): 
                """
                Construction of the architecture of the autoencoder
                """       
                np.random.seed(1500)
                self.lambd = lambd
                self.beta = 1e-1
                self.rho = 5e-2
                self.learning_rate = 9e-1

                self.params = {'L_in'     : input_size,
                               'L_hidden' : hidden_size,
                               'Lambda'   : self.lambd,
                               'Beta'     : self.beta,
                               'Rho'      : self.rho}

                self.W_e = self.InitParams(input_size,hidden_size)
                
                self.loss = []

            def InitParams(self,input_size,hidden_size):
                """
                Given the size of the input node and hidden node, initialize the weights
                drawn from uniform distribution ~ Uniform[- sqrt(6/(L_pre + L_post)) , sqrt(6/(L_pre + L_post))]
                """
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = input_size

                W1_high = self.w_o(input_size,hidden_size)
                W1_low = - W1_high
                W1_size = (input_size,hidden_size)
                self.W1 = np.random.uniform(W1_low,W1_high,size = W1_size)

                B1_size = (1,hidden_size)
                self.B1 = np.random.uniform(W1_low,W1_high,size = B1_size)

                W2_high = self.w_o(hidden_size,self.output_size)
                W2_low = - W2_high
                W2_size = (hidden_size,self.output_size)

                self.W2 = np.random.uniform(W2_low,W2_high,size = W2_size)

                B2_size = (1,self.output_size)
                self.B2 = np.random.uniform(W1_low,W1_high,size = B2_size)
         
                return  {'W1' : self.W1,
                         'W2' : self.W2,
                         'B1' : self.B1,
                         'B2' : self.B2}


            def w_o(self,L_pre,L_post):
                return np.sqrt(6/(L_pre + L_post))


            def sigmoid(self,X, grad = True):
                """
                Computing sigmoid and it's gradient w.r.t. it's input

                """
                sig = 1/(1 + np.exp(-X))

                return sig * (1-sig) if grad else sig

          
            def forward(self,X): 
                """
                Forward propagation
                """      
                W1 = self.W_e['W1'] 
                W2 = self.W_e['W2']
                B1 = self.W_e['B1']
                B2 = self.W_e['B2'] 
                
                Z1 = np.dot(X,W1) + B1         
                A1 = self.sigmoid(Z1,grad = False) 

                Z2 = np.dot(A1,W2) + B2
                A2 = self.sigmoid(Z2,grad = False)
                            
                return {"Z1": Z1,"A1": A1,
                        "Z2": Z2,"A2": A2}
            
            def total_loss(self,outs,label):
                W1 = self.W_e['W1']
                W2 = self.W_e['W2']

                Lambda = self.params['Lambda']
                beta = self.params['Beta']
                rho = self.params['Rho']

                J_mse = self.MSE(outs['A2'],label, grad = False)
                J_tykhonow = self.TykhonowRegularization(W1 = W1, W2 = W2,lambd = Lambda, grad = False)
                J_KL = self.KL_divergence(rho = rho,expected = np.mean(outs['A1']), beta = beta, grad = False)

                return J_mse + J_tykhonow + J_KL

            def MSE(self,pred,label, grad = True):
                """
                Calculating Mean Sqaured Error and it's gradient w.r.t. output
                """
                return 1/2 *  (pred - label) if grad else 1/2 * np.sum((pred - label)**2)/pred.shape[0]

            def aeCost(self,data):
                
                outs = self.forward(data)        
                loss = self.total_loss(outs,data)        
                grads = self.backward(outs,data)
                
                return {'J'      : loss,
                        'J_grad' : grads}

            def KL_divergence(self,rho,beta,expected,grad = True):
                """
                Computing KL-divergence and it's gradients, note that gradients is only for W1
                """       
                return np.tile(beta * (-(rho/expected) + (1-rho)/(1-expected) ), reps = (10240,1)) if grad else beta * (np.sum((rho * np.log(rho/expected)) + ((1-rho)*np.log((1-rho)/(1-expected)))))

            def TykhonowRegularization(self,W1,W2,lambd,grad = True):
                """
                    L2 based regularization computing and it's gradients
                """
                return {'dW1': lambd * W1, 'dW2': lambd * W2} if grad else (lambd/2) * (np.sum(W1**2) + np.sum(W2**2))
               

            def backward(self,outs,data):
                """
                Given the forward pass outputs, input and their labels,
                returning gradients w.r.t. loss functions
                """       
                m = data.shape[0]
                
                Lambda = self.params['Lambda']
                beta = self.params['Beta']
                rho = self.params['Rho']

                W1 = self.W_e['W1'] 
                W2 = self.W_e['W2']
                B1 = self.W_e['B1']
                B2 = self.W_e['B2']

                Z1 = outs['Z1']
                A1 = outs['A1']
                Z2 = outs['Z2']
                A2 = outs['A2']
                
                L2_grad = self.TykhonowRegularization(W1,W2,lambd = Lambda , grad = True)
                KL_grad_W1 = self.KL_divergence(rho,beta,expected = np.mean(A1),grad = True)

                dZ2 = self.MSE(A2,data, grad = True) * self.sigmoid(Z2, grad = True)        
                dW2 = (1/m) * (np.dot(A1.T,dZ2) + L2_grad['dW2'])
                dB2 = (1/m) * (np.sum(dZ2, axis=0, keepdims=True))
                
                dZ1 = (np.dot(dZ2,W2.T) + KL_grad_W1)  * self.sigmoid(Z1,grad = True)        
                dW1 = (1/m) * (np.dot(data.T,dZ1) + L2_grad['dW1'])
                dB1 = (1/m) * (np.sum(dZ1, axis=0, keepdims=True))
               
                assert (dW1.shape == W1.shape and dW2.shape == W2.shape)      

                return {"dW1": dW1, "dW2": dW2,
                        "dB1": dB1, "dB2": dB2}   
            
           
            def fit(self,data,epochs = 5000,verbose = True):
                """
                Given the traning dataset,their labels and number of epochs
                fitting the model, and measure the performance
                by validating training dataset.
                """
                for epoch in range(epochs):
                 
                    loss_and_grads = self.aeCost(data)            
                    self.step(grads = loss_and_grads['J_grad'])
                    
                    if verbose:
                        print(f"[{epoch}/{epochs}]     ----------> Loss  :{loss_and_grads['J']}")
                        
                    self.loss.append(loss_and_grads['J'])

            def step(self,grads):
                """
                Updating configurable parameters according to full-batch stochastic gradient update rule
                """       
                self.W_e['W1'] += -self.learning_rate * grads['dW1']
                self.W_e['W2'] += -self.learning_rate * grads['dW2'] 
                self.W_e['B1'] += -self.learning_rate * grads['dB1']                
                self.W_e['B2'] += -self.learning_rate * grads['dB2']
                self.learning_rate *= 0.9999

            
            def evaluate(self):
                plt.plot(self.loss, color = 'orange')
                plt.xlabel(' # of Epochs')
                plt.ylabel('Loss')
                plt.title('Training Loss versus Epochs')
                plt.legend([f'Loss : {self.loss[-1]}'])
          
            def display_weights(self):
                """
                Display weights as a image for feature representation
                """
                W1 = self.W_e['W1']
                num_disp = W1.shape[1]
                fig = plt.figure(figsize = (9,8))
                for i in range(num_disp):
                    plt.subplot(8,8,i+1)
                    plt.imshow(W1.T[i].reshape(16,16),cmap = 'gray')
                    plt.axis('off')
                fig.suptitle('Hidden Layer Feature Representation')    
                plt.show()

            def display_outputs(self,output,data,num = 4):

                """
                Displaying outputs, please give only sqaured values, i.e., 1,4,16,...
                """
                random_indexes = np.random.randint(output.shape[0],size = num)
                plt.figure(figsize=(12, 4))
                for i in range(len(random_indexes)):
                    ax = plt.subplot(2,5,i+1)
                    plt.imshow(output[random_indexes[i]].reshape(16,16),cmap = 'gray')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    plt.title("Reconstructed Image")
                    #plt.axis('off')
                    ax = plt.subplot(2, 5, i + 1 + 5)
                    plt.imshow(data[random_indexes[i]].reshape(16,16),cmap = 'gray')
                    plt.title("Original Image")
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                
                plt.show() 

         
            def parameters(self):
                """
                Returns configurable parameters

                """
                return self.W_e

            def history(self):
                return {'Loss' : self.loss}
                            
                       


        # %%
        class Solver:
            """
            Given  as input, A Solver encapsulates all the logic necessary for training then
            implement gradients solver to minimize the cost.The Solver performs stochastic gradient descent.

            """
            def __init__(self, model,data):

                self.model = model
                self.data = data       

            def train(self,epochs = 5000,verbose = False):
                """
                Optimization of the model by minimizing cost by solving gradients
                """
                self.model.fit(self.data,epochs,verbose)

            def parameters(self):
                """
                Returning configurable parameters of the network
                """
                return self.model.parameters()


        # %%
        data_feed = preprocessor.Flatten(data_processed)
        input_size = data_feed.shape[1]
        hidden_size = 64
        autoencoder = Autoencoder(input_size = input_size, hidden_size = hidden_size,lambd = 5e-4)


        # %%
        solver = Solver(model = autoencoder, data = data_feed)
        solver.train(verbose = True)


        # %%
        net_params = solver.parameters()
        net_history = autoencoder.history()


        # %%
        autoencoder.evaluate()


        # %%
        autoencoder.display_weights()


        # %%
        preds = autoencoder.forward(data_feed)
        autoencoder.display_outputs(preds['A2'],data_feed)


        # %%
        hidden_size_1 = 10
        lambd_1 = 0
        autoencoder_1 = Autoencoder(input_size = input_size, hidden_size = hidden_size_1, lambd = lambd_1)
        solver_1 = Solver(model = autoencoder_1, data = data_feed)
        solver_1.train()
        autoencoder_1.evaluate()
        autoencoder_1.display_weights()
        preds_1 = autoencoder_1.forward(data_feed)
        autoencoder_1.display_outputs(preds_1['A2'],data_feed)


        # %%
        hidden_size_2 = 10
        lambd_2 = 1e-3
        autoencoder_2 = Autoencoder(input_size = input_size, hidden_size = hidden_size_2, lambd = lambd_2)
        solver_2 = Solver(model = autoencoder_2, data = data_feed)
        solver_2.train()
        autoencoder_2.evaluate()
        autoencoder_2.display_weights()
        preds_2 = autoencoder_2.forward(data_feed)
        autoencoder_2.display_outputs(preds_2['A2'],data_feed)


        # %%
        hidden_size_3 = 10
        lambd_3 = 1e-5
        autoencoder_3 = Autoencoder(input_size = input_size, hidden_size = hidden_size_3, lambd = lambd_3)
        solver_3 = Solver(model = autoencoder_3, data = data_feed)
        solver_3.train()
        autoencoder_3.evaluate()
        autoencoder_3.display_weights()
        preds_3 = autoencoder_3.forward(data_feed)
        autoencoder_3.display_outputs(preds_3['A2'],data_feed)


        # %%
        hidden_size_4 = 50
        lambd_4 = 0
        autoencoder_4 = Autoencoder(input_size = input_size, hidden_size = hidden_size_4, lambd = lambd_4)
        solver_4 = Solver(model = autoencoder_4, data = data_feed)
        solver_4.train()
        autoencoder_4.evaluate()
        autoencoder_4.display_weights()
        preds_4 = autoencoder_4.forward(data_feed)
        autoencoder_4.display_outputs(preds_4['A2'],data_feed)


        # %%
        hidden_size_5 = 50
        lambd_5 = 1e-3
        autoencoder_5 = Autoencoder(input_size = input_size, hidden_size = hidden_size_5, lambd = lambd_5)
        solver_5 = Solver(model = autoencoder_5, data = data_feed)
        solver_5.train()
        autoencoder_5.evaluate()
        autoencoder_5.display_weights()
        preds_5 = autoencoder_5.forward(data_feed)
        autoencoder_5.display_outputs(preds_5['A2'],data_feed)


        # %%
        autoencoder_5.display_outputs(preds_5['A2'],data_feed)


        # %%
        hidden_size_6 = 50
        lambd_6 = 1e-5
        autoencoder_6 = Autoencoder(input_size = input_size, hidden_size = hidden_size_6, lambd = lambd_6)
        solver_6 = Solver(model = autoencoder_6, data = data_feed)
        solver_6.train()
        autoencoder_6.evaluate()
        autoencoder_6.display_weights()
        preds_6 = autoencoder_6.forward(data_feed)


        # %%
        autoencoder_6.display_outputs(preds_6['A2'],data_feed)


        # %%
        hidden_size_7 = 100
        lambd_7 = 0
        autoencoder_7 = Autoencoder(input_size = input_size, hidden_size = hidden_size_7, lambd = lambd_7)
        solver_7 = Solver(model = autoencoder_7, data = data_feed)
        solver_7.train()
        autoencoder_7.evaluate()
        #autoencoder_7.display_weights()
        preds_7 = autoencoder_7.forward(data_feed)
        autoencoder_7.display_outputs(preds_7['A2'],data_feed)


        # %%
        #autoencoder_7.display_weights()
        W1 = autoencoder_7.W_e['W1']
        num_disp = W1.shape[1]
        fig = plt.figure(figsize = (9,8))
        for i in range(num_disp):
            plt.subplot(10,10,i+1)
            plt.imshow(W1.T[i].reshape(16,16),cmap = 'gray')
            plt.axis('off')
        fig.suptitle('Hidden Layer Feature Representation')    
        plt.show()
        preds_7 = autoencoder_7.forward(data_feed)
        autoencoder_7.display_outputs(preds_7['A2'],data_feed)


        # %%
        autoencoder_7.display_outputs(preds_7['A2'],data_feed)


        # %%
        hidden_size_8 = 100
        lambd_8 = 1e-3
        autoencoder_8 = Autoencoder(input_size = input_size, hidden_size = hidden_size_8, lambd = lambd_8)
        solver_8 = Solver(model = autoencoder_8, data = data_feed)
        solver_8.train()
        autoencoder_8.evaluate()
        W1 = autoencoder_8.W_e['W1']
        num_disp = W1.shape[1]
        fig = plt.figure(figsize = (9,8))
        for i in range(num_disp):
            plt.subplot(10,10,i+1)
            plt.imshow(W1.T[i].reshape(16,16),cmap = 'gray')
            plt.axis('off')
        fig.suptitle('Hidden Layer Feature Representation')    
        plt.show()
        preds_8 = autoencoder_8.forward(data_feed)
        autoencoder_8.display_outputs(preds_8['A2'],data_feed)


        # %%
        hidden_size_9 = 100
        lambd_9 = 1e-5
        autoencoder_9 = Autoencoder(input_size = input_size, hidden_size = hidden_size_9, lambd = lambd_9)
        solver_9 = Solver(model = autoencoder_9, data = data_feed)
        solver_9.train()
        autoencoder_9.evaluate()
        W1 = autoencoder_9.W_e['W1']
        num_disp = W1.shape[1]
        fig = plt.figure(figsize = (9,8))
        for i in range(num_disp):
            plt.subplot(10,10,i+1)
            plt.imshow(W1.T[i].reshape(16,16),cmap = 'gray')
            plt.axis('off')
        fig.suptitle('Hidden Layer Feature Representation')    
        plt.show()
        preds_9 = autoencoder_9.forward(data_feed)


        # %%
        autoencoder_9.display_outputs(preds_9['A2'],data_feed)


        # %%
        import tensorflow as tf
        from tensorflow.keras import layers
        from tensorflow import keras
        import tensorflow.keras.backend as K


        # %%
        if tf.test.gpu_device_name(): 
            print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
        else:
           print("NO GPU, that's okey")


        # %%
        def MeanSquaredError():
          def customMeanSquaredError(pred,label):
            return 1/2 * K.sum((pred - label)**2)/pred.shape[0]
          return customMeanSquaredError

        def KL_divergence(rho, beta):
          def customKL(out):
            kl1 = rho*K.log(rho/K.mean(out, axis=0))
            kl2 = (1-rho)*K.log((1-rho)/(1-K.mean(out, axis=0)))
            return beta*K.sum(kl1+kl2)
          return customKL


        # %%
        def create_model(hidden_size,lambd):
          tf_weights = tf_weight_initializer(inp_dim = inp_dim, hidden_dim = hidden_size)
          input_img = keras.Input(shape=(inp_dim,))
          encoded = layers.Dense(encoding_dim,activation='sigmoid',
                               kernel_regularizer=tf.keras.regularizers.l2(lambd),
                               activity_regularizer=KL_divergence(rho,beta),
                               kernel_initializer = tf_weights['W1'],
                               bias_initializer = tf_weights['B1'])(input_img)

          decoded = layers.Dense(inp_dim,activation='sigmoid',
                               activity_regularizer=tf.keras.regularizers.l2(lambd),
                               kernel_initializer = tf_weights['W2'],
                               bias_initializer = tf_weights['B2'])(encoded)

          tf_autoencoder = keras.Model(input_img,decoded)
          optimizer = tf.keras.optimizers.SGD(learning_rate=0.9,momentum=0,nesterov=False)
          tf_autoencoder.compile(optimizer=optimizer,loss=MeanSquaredError())                   

          

          return tf_autoencoder

        def plot_tf_weights(W1):  
          num_disp = W1.shape[1]
          fig = plt.figure(figsize = (9,8))
          for i in range(num_disp):
              plt.subplot(10,10,i+1)
              plt.imshow(W1.T[i].reshape(16,16),cmap = 'gray')
              plt.axis('off')
          fig.suptitle('Hidden Layer Feature Representation')    
          plt.show()


        # %%
        tf_model_1 = create_model(hidden_size = 10,lambd = 0)
        tf_model_1.fit(data_feed, data_feed,
                        epochs=5000,
                        batch_size=data_feed.shape[0])

        tf_history_1 = tf_model_1.history.history
        plt.plot(tf_history_1['loss'])
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Loss versus Epoch')
        #plt.legend([f'Loss : {tf_history_1['loss'][-1]}'])

        tf_preds_1 = tf_model_1.predict(data_feed)
        autoencoder.display_outputs(tf_preds_1,data_feed)
        tf_weights_1 = tf_model_1.get_weights()
        plot_tf_weights(tf_weights_1[0]) 
          


        # %%
        tf_model_2 = create_model(hidden_size = 10,lambd = 1e-3)
        tf_model_2.fit(data_feed, data_feed,
                        epochs=5000,
                        batch_size=data_feed.shape[0])

        tf_history_2 = tf_model_2.history.history
        plt.plot(tf_history_2['loss'])
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Loss versus Epoch')

        tf_preds_2 = tf_model_2.predict(data_feed)
        autoencoder.display_outputs(tf_preds_2,data_feed)
        tf_weights_2 = tf_model_2.get_weights()
        plot_tf_weights(tf_weights_2[0]) 


        # %%
        tf_model_3 = create_model(hidden_size = 10,lambd = 1e-5)
        tf_model_3.fit(data_feed, data_feed,
                        epochs=5000,
                        batch_size=data_feed.shape[0])

        tf_history_3 = tf_model_3.history.history
        plt.plot(tf_history_3['loss'])
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Loss versus Epoch')

        tf_preds_3 = tf_model_3.predict(data_feed)
        autoencoder.display_outputs(tf_preds_3,data_feed)
        tf_weights_3 = tf_model_3.get_weights()
        plot_tf_weights(tf_weights_3[0]) 


        # %%
        tf_model_4 = create_model(hidden_size = 50,lambd = 0)
        tf_model_4.fit(data_feed, data_feed,
                        epochs=5000,
                        batch_size=data_feed.shape[0])

        tf_history_4 = tf_model_4.history.history
        plt.plot(tf_history_4['loss'])
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Loss versus Epoch')

        tf_preds_4 = tf_model_4.predict(data_feed)
        autoencoder.display_outputs(tf_preds_4,data_feed)
        tf_weights_4 = tf_model_4.get_weights()
        plot_tf_weights(tf_weights_4[0]) 


        # %%
        tf_model_5 = create_model(hidden_size = 50,lambd = 1e-3)
        tf_model_5.fit(data_feed, data_feed,
                        epochs=5000,
                        batch_size=data_feed.shape[0])

        tf_history_5 = tf_model_5.history.history
        plt.plot(tf_history_5['loss'])
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Loss versus Epoch')

        tf_preds_5 = tf_model_5.predict(data_feed)
        autoencoder.display_outputs(tf_preds_5,data_feed)
        tf_weights_5 = tf_model_5.get_weights()
        plot_tf_weights(tf_weights_5[0]) 


        # %%
        tf_model_6 = create_model(hidden_size = 50,lambd = 1e-5)
        tf_model_6.fit(data_feed, data_feed,
                        epochs=5000,
                        batch_size=data_feed.shape[0])

        tf_history_6 = tf_model_6.history.history
        plt.plot(tf_history_6['loss'])
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Loss versus Epoch')

        tf_preds_6 = tf_model_6.predict(data_feed)
        autoencoder.display_outputs(tf_preds_6,data_feed)
        tf_weights_6 = tf_model_1.get_weights()
        plot_tf_weights(tf_weights_6[0]) 


        # %%
        tf_model_7 = create_model(hidden_size = 100,lambd = 0)
        tf_model_7.fit(data_feed, data_feed,
                        epochs=5000,
                        batch_size=data_feed.shape[0])

        tf_history_7 = tf_model_7.history.history
        plt.plot(tf_history_7['loss'])
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Loss versus Epoch')

        tf_preds_7 = tf_model_7.predict(data_feed)
        autoencoder.display_outputs(tf_preds_7,data_feed)
        tf_weights_7 = tf_model_7.get_weights()
        plot_tf_weights(tf_weights_7[0]) 


        # %%
        tf_model_8 = create_model(hidden_size = 100,lambd = 1e-3)
        tf_model_8.fit(data_feed, data_feed,
                        epochs=5000,
                        batch_size=data_feed.shape[0])

        tf_history_8 = tf_model_8.history.history
        plt.plot(tf_history_8['loss'])
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Loss versus Epoch')

        tf_preds_8 = tf_model_8.predict(data_feed)
        autoencoder.display_outputs(tf_preds_8,data_feed)
        tf_weights_8 = tf_model_8.get_weights()
        plot_tf_weights(tf_weights_8[0]) 


        # %%
        tf_model_9 = create_model(hidden_size = 100,lambd = 1e-5)
        tf_model_9.fit(data_feed, data_feed,
                        epochs=5000,
                        batch_size=data_feed.shape[0])

        tf_history_9 = tf_model_9.history.history
        plt.plot(tf_history_9['loss'])
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Loss versus Epoch')

        tf_preds_9 = tf_model_9.predict(data_feed)
        autoencoder.display_outputs(tf_preds_9,data_feed)
        tf_weights_9 = tf_model_9.get_weights()
        plot_tf_weights(tf_weights_9[0]) 


        # %%
        #encoding_dim = 10
        rho,beta = 5e-1,1e-1
        inp_dim = 256
        #lamb = 0

        W_scaler = lambda L_pre,L_post : np.sqrt(6/(L_pre + L_post))

        def tf_weight_initializer(inp_dim,hidden_dim):
          initializer_1 = tf.keras.initializers.RandomUniform(minval=-W_scaler(inp_dim,hidden_dim), maxval=W_scaler(inp_dim,hidden_dim))
          values_2 = initializer_1(shape=(inp_dim,hidden_dim))

          initializer_2 = tf.keras.initializers.RandomUniform(minval=-W_scaler(hidden_dim,inp_dim), maxval=W_scaler(hidden_dim,inp_dim))
          values_2 = initializer_2(shape=(inp_dim,hidden_dim))

          initializer_3 = tf.keras.initializers.RandomUniform(minval=-W_scaler(inp_dim,hidden_dim), maxval=W_scaler(inp_dim,hidden_dim))
          values_3 = initializer_3(shape=(1,hidden_dim))

          initializer_4 = tf.keras.initializers.RandomUniform(minval=-W_scaler(hidden_dim,inp_dim), maxval=W_scaler(hidden_dim,inp_dim))
          values_4 = initializer_4(shape=(1,inp_dim))

          return {'W1':initializer_1,
                  'W2':initializer_2,
                  'B1':initializer_3,
                  'B2':initializer_4}

        tf_weights = tf_weight_initializer(inp_dim = inp_dim, hidden_dim = encoding_dim)


        # %%
        input_img = keras.Input(shape=(inp_dim,))

        encoded = layers.Dense(encoding_dim,activation='sigmoid',
                               kernel_regularizer=tf.keras.regularizers.l2(lamb),
                               activity_regularizer=KL_divergence(rho,beta),
                               kernel_initializer = tf_weights['W1'],
                               bias_initializer = tf_weights['B1'])(input_img)

        decoded = layers.Dense(inp_dim,activation='sigmoid',
                               activity_regularizer=tf.keras.regularizers.l2(lamb),
                               kernel_initializer = tf_weights['W2'],
                               bias_initializer = tf_weights['B2'])(encoded)

        tf_autoencoder = keras.Model(input_img,decoded)


        # %%
        tf_autoencoder.summary()


        # %%
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.9,momentum=0,nesterov=False)
        tf_autoencoder.compile(optimizer=optimizer,loss=MeanSquaredError())


        # %%
        tf_autoencoder.fit(data_feed, data_feed,
                        epochs=5000,
                        batch_size=data_feed.shape[0])


        # %%
        tf_history = tf_autoencoder.history.history
        plt.plot(tf_history['loss'])
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Loss versus Epoch')


        # %%
        tf_preds = tf_autoencoder.predict(data_feed)


        # %%
        autoencoder.display_outputs(tf_preds,data_feed)


        # %%
        tf_weights = tf_autoencoder.get_weights()


        # %%
        tf_weights = tf_autoencoder.get_weights()
        W1 = tf_weights[0]
        num_disp = W1.shape[1]
        fig = plt.figure(figsize = (9,8))
        for i in range(num_disp):
            plt.subplot(8,8,i+1)
            plt.imshow(W1.T[i].reshape(16,16),cmap = 'gray')
            plt.axis('off')
        fig.suptitle('Hidden Layer Feature Representation')    
        plt.show()


        # %%



        # %%
        input_img_optim = keras.Input(shape=(inp_dim,))

        encoded_optim = layers.Dense(encoding_dim,activation='sigmoid',
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                               activity_regularizer=KL_divergence(rho,beta))(input_img_optim)

        decoded_optim = layers.Dense(inp_dim,activation='sigmoid',
                               activity_regularizer=tf.keras.regularizers.l2(5e-4))(encoded_optim)

        tf_autoencoder_optim = keras.Model(input_img_optim,decoded_optim)

        tf_autoencoder.compile(optimizer='adam',loss=MeanSquaredError())
        tf_autoencoder.summary()


        # %%
        tf_autoencoder.fit(data_feed, data_feed,
                        epochs=5000,
                        batch_size=data_feed.shape[0])


        # %%
        tf_history_optim = tf_autoencoder.history.history
        plt.plot(tf_history_optim['loss'],color = 'green')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Loss versus Epoch')


        # %%
        tf_preds_optim = tf_autoencoder.predict(data_feed)
        autoencoder.display_outputs(tf_preds_optim,data_feed)


        # %%
        tf_weights_optim = tf_autoencoder_optim.get_weights()
        W1 = tf_weights[0]
        num_disp = W1.shape[1]
        fig = plt.figure(figsize = (9,8))
        for i in range(num_disp):
            plt.subplot(8,8,i+1)
            plt.imshow(W1.T[i].reshape(16,16),cmap = 'gray')
            plt.axis('off')
        fig.suptitle('Hidden Layer Feature Representation')    
        plt.show()


        # %%
    elif question == '3' :

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'




        # %%
        def sigmoid(x):
            c = np.clip(x,-700,700)
            return 1 / (1 + np.exp(-c))
        def dsigmoid(y):
            return y * (1 - y)
        def tanh(x):
            return np.tanh(x)
        def dtanh(y):
            return 1 - y * y


        # %%
        with h5py.File('assign3_data3.h5','r') as F:
        # Names variable contains the names of training and testing file 
            names = list(F.keys())
            X_train = np.array(F[names[0]][()])
            y_train = np.array(F[names[1]][()])
            X_test = np.array(F[names[2]][()])
            y_test = np.array(F[names[3]][()])


        # %%
        class Metrics: 
            """
            Necessary metrics to evaluate the model.
                Functions(labels,preds):
                --- confusion_matrix
                --- accuracy_score     
            """ 
            def confusion_matrix(self,labels,preds):
                """
                Takes desireds/labels and softmax predictions,
                return a confusion matrix.
                """
                label = pd.Series(labels,name='Actual')
                pred = pd.Series(preds,name='Predicted')
                return pd.crosstab(label,pred)

            def accuracy_score(self,labels,preds): 
                """
                Takes desireds/labels and softmax predictions,
                return a accuracy_score.
                """       
                count = 0
                size = labels.shape[0]
                for i in range(size):
                    if preds[i] == labels[i]:
                        count +=1
                return  100 * (count/size)

            def accuracy(self,labels,preds):
                """
                Takes desireds/labels and softmax predictions,
                return a accuracy.
                """
                return 100 * (labels == preds).mean()    


        # %%
        class Activations:
            """
            Necessary activation functions for recurrent neural network(RNN,LSTM,GRU).
            """
            def relu_alternative(self,X):
                """
                    Rectified linear unit activation(ReLU).
                """
                return np.maximum(X, 0)

            def ReLU(self,X):
                """
                    Rectified linear unit activation(ReLU).
                    Most time efficient version.
                """
                return (abs(X) + X) / 2

            def relu_another(self,X):
                """
                    Rectified linear unit activation(ReLU).
                """
                return X * (X > 0)

            def tanh(self,X):
                return np.tanh(X)

            def tanh_manuel(self,X):
                """
                    Hyperbolic tangent activation(tanh).
                """      
                return (np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X))

            def sigmoid(self,X):
                """
                    Sigmoidal activation.
                """
                c = np.clip(X,-700,700)
                return 1/(1 + np.exp(-c))

            def softmax(self,X):
                """
                    Stable version of softmax classifier, note that column sum is equal to 1.
                """
                e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
                return e_x / np.sum(e_x, axis=-1, keepdims=True)
                
            def softmax_stable(self,X):
                """
                Less stable version of softmax activation

                """
                e_x = np.exp(X - np.max(X))
                return e_x / np.sum(e_x)

            def ReLUDerivative(self,X): 
                """
                The derivative of the ReLU function w.r.t. given input.
                """
                return 1 * (X > 0)

            def ReLU_grad(self,X):
                """
                The derivative of the ReLU function w.r.t. given input.
                """
                X[X<=0] = 0
                X[X>1] = 1
                return X

            def dReLU(self,X):  
                """
                The derivative of the ReLU function w.r.t. given input.
                """     
                return np.where(X <= 0, 0, 1)

            def dtanh(self,X): 
                """
                The derivative of the tanh function w.r.t. given input.
                """       
                return  1-(np.tanh(X)**2)

            def dsigmoid(self,X):
                """
                The derivative of the sigmoid function w.r.t. given input.
                """
                return self.sigmoid(X) * (1-self.sigmoid(X))    
            
            def softmax_stable_gradient(self,soft_out):           
                return soft_out * (1 - soft_out)

            def softmax_grad(self,softmax):        
                s = softmax.reshape(-1,1)
                return np.diagflat(s) - np.dot(s, s.T)

            def softmax_gradient(self,Sz):
                """Computes the gradient of the softmax function.
                z: (T, 1) array of input values where the gradient is computed. T is the
                number of output classes.
                Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
                is DjSi - the partial derivative of Si w.r.t. input j.
                """
                
                # -SjSi can be computed using an outer product between Sz and itself. Then
                # we add back Si for the i=j cases by adding a diagonal matrix with the
                # values of Si on its diagonal.
                D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
                return D


        # %%
        class RNN(object):
            """
            Recurrent Neural Network for classifying human activity.
            RNN encapsulates all necessary logic for training the network.

            """
            def __init__(self,input_dim = 3,hidden_dim = 128, seq_len = 150, learning_rate = 1e-1, mom_coeff = 0.85, batch_size = 32, output_class = 6):

                """

                Initialization of weights/biases and other configurable parameters.
                
                """
                np.random.seed(150)
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim

                # Unfold case T = 150 :
                self.seq_len = seq_len
                self.output_class = output_class
                self.learning_rate = learning_rate
                self.batch_size = batch_size
                self.mom_coeff = mom_coeff

                # Xavier uniform scaler :
                Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

                lim_inp2hid = Xavier(self.input_dim,self.hidden_dim)
                self.W1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(self.input_dim,self.hidden_dim))
                self.B1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(1,self.hidden_dim))

                lim_hid2hid = Xavier(self.hidden_dim,self.hidden_dim)
                self.W1_rec= np.random.uniform(-lim_hid2hid,lim_hid2hid,(self.hidden_dim,self.hidden_dim))

                lim_hid2out = Xavier(self.hidden_dim,self.output_class)
                self.W2 = np.random.uniform(-lim_hid2out,lim_hid2out,(self.hidden_dim,self.output_class))
                self.B2 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(1,self.output_class))

                # To keep track loss and accuracy score :     
                self.train_loss,self.test_loss,self.train_acc,self.test_acc = [],[],[],[]
                
                # Storing previous momentum updates :
                self.prev_updates = {'W1'       : 0,
                                     'B1'       : 0,
                                     'W1_rec'   : 0,
                                     'W2'       : 0,
                                     'B2'       : 0}


            def forward(self,X) -> tuple:
                """ Forward propagation of the RNN through time.
                

                Inputs:
                --- X is the bacth.
                --- h_prev_state is the previous state of the hidden layer.
                
                Returns:
                --- (X_state,hidden_state,probs) as a tuple.       
                ------ 1) X_state is the input across all time steps
                ------ 2) hidden_state is the hidden stages across time
                ------ 3) probs is the probabilities of each outputs, i.e. outputs of softmax
                """ 
                X_state = dict()
                hidden_state = dict()
                output_state = dict()
                probs = dict()

                
                self.h_prev_state = np.zeros((1,self.hidden_dim))
                hidden_state[-1] = np.copy(self.h_prev_state)

                # Loop over time T = 150 :
                for t in range(self.seq_len):

                    # Selecting first record with 3 inputs, dimension = (batch_size,input_size)
                    X_state[t] = X[:,t]

                    # Recurrent hidden layer :
                    hidden_state[t] = np.tanh(np.dot(X_state[t],self.W1) + np.dot(hidden_state[t-1],self.W1_rec) + self.B1)
                    output_state[t] = np.dot(hidden_state[t],self.W2) + self.B2

                    # Per class probabilites :
                    probs[t] = activations.softmax(output_state[t])

                return (X_state,hidden_state,probs)
                

            def BPTT(self,cache,Y):
                """

                Back propagation through time algorihm.
                Inputs:
                -- Cache = (X_state,hidden_state,probs)
                -- Y = desired output

                Returns:
                -- Gradients w.r.t. all configurable elements
                """

                X_state,hidden_state,probs = cache

                # backward pass: compute gradients going backwards
                dW1, dW1_rec, dW2 = np.zeros_like(self.W1), np.zeros_like(self.W1_rec), np.zeros_like(self.W2)

                dB1, dB2 = np.zeros_like(self.B1), np.zeros_like(self.B2)

                dhnext = np.zeros_like(hidden_state[0])

                dy = np.copy(probs[149])      
                dy[np.arange(len(Y)),np.argmax(Y,1)] -= 1
                
                dB2 += np.sum(dy,axis = 0, keepdims = True)
                dW2 += np.dot(hidden_state[149].T,dy)

                for t in reversed(range(1,self.seq_len)):

                    
                
                    dh = np.dot(dy,self.W2.T) + dhnext
                
                    dhrec = (1 - (hidden_state[t] * hidden_state[t])) * dh

                    dB1 += np.sum(dhrec,axis = 0, keepdims = True)
                    
                    dW1 += np.dot(X_state[t].T,dhrec)
                    
                    dW1_rec += np.dot(hidden_state[t-1].T,dhrec)

                    dhnext = np.dot(dhrec,self.W1_rec.T)

                       
                for grad in [dW1,dB1,dW1_rec,dW2,dB2]:
                    np.clip(grad, -10, 10, out = grad)


                return [dW1,dB1,dW1_rec,dW2,dB2]    
                
            def earlyStopping(self,ce_train,ce_val,ce_threshold,acc_train,acc_val,acc_threshold):
                if ce_train - ce_val < ce_threshold or acc_train - acc_val > acc_threshold:
                    return True
                else:
                    return False
           

            def CategoricalCrossEntropy(self,labels,preds):
                """
                Computes cross entropy between labels and model's predictions
                """
                predictions = np.clip(preds, 1e-12, 1. - 1e-12)
                N = predictions.shape[0]         
                return -np.sum(labels * np.log(predictions + 1e-9)) / N

            def step(self,grads,momentum = True):
                """
                SGD on mini batches
                """

             
                #for config_param,grad in zip([self.W1,self.B1,self.W1_rec,self.W2,self.B2],grads):
                    #config_param -= self.learning_rate * grad

                if momentum:
                    
                    delta_W1 = -self.learning_rate * grads[0] +  self.mom_coeff * self.prev_updates['W1']
                    delta_B1 = -self.learning_rate * grads[1] +  self.mom_coeff * self.prev_updates['B1']  
                    delta_W1_rec = -self.learning_rate * grads[2] +  self.mom_coeff * self.prev_updates['W1_rec']
                    delta_W2 = -self.learning_rate * grads[3] +  self.mom_coeff * self.prev_updates['W2']              
                    delta_B2 = -self.learning_rate * grads[4] +  self.mom_coeff * self.prev_updates['B2']
                    
                       
                    self.W1 += delta_W1
                    self.W1_rec += delta_W1_rec
                    self.W2 += delta_W2
                    self.B1 += delta_B1
                    self.B2 += delta_B2     

                    
                    self.prev_updates['W1'] = delta_W1
                    self.prev_updates['W1_rec'] = delta_W1_rec
                    self.prev_updates['W2'] = delta_W2
                    self.prev_updates['B1'] = delta_B1
                    self.prev_updates['B2'] = delta_B2

                    self.learning_rate *= 0.9999

            def fit(self,X,Y,X_val,y_val,epochs = 50 ,verbose = True, earlystopping = False):
                """
                Given the traning dataset,their labels and number of epochs
                fitting the model, and measure the performance
                by validating training dataset.
                """
                        
                
                for epoch in range(epochs):
                    
                    print(f'Epoch : {epoch + 1}')

                    perm = np.random.permutation(3000)           
                    
                    for i in range(round(X.shape[0]/self.batch_size)): 

                        batch_start  =  i * self.batch_size
                        batch_finish = (i+1) * self.batch_size
                        index = perm[batch_start:batch_finish]
                        
                        X_feed = X[index]    
                        y_feed = Y[index]
                        
                        cache_train = self.forward(X_feed)                                                          
                        grads = self.BPTT(cache_train,y_feed)                
                        self.step(grads)
              

                    cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[2][149])
                    predictions_train = self.predict(X)
                    acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)

                    _,__,probs_test = self.forward(X_val)
                    cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[149])
                    predictions_val = np.argmax(probs_test[149],1)
                    acc_val = metrics.accuracy(np.argmax(y_val,1),predictions_val)

                    if earlystopping:                
                        if self.earlyStopping(ce_train = cross_loss_train,ce_val = cross_loss_val,ce_threshold = 3.0,acc_train = acc_train,acc_val = acc_val,acc_threshold = 15): 
                            break
                    


                    if verbose:

                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                        print('______________________________________________________________________________________\n')                         
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                        print('______________________________________________________________________________________\n')
                        
                    self.train_loss.append(cross_loss_train)              
                    self.test_loss.append(cross_loss_val) 
                    self.train_acc.append(acc_train)              
                    self.test_acc.append(acc_val)

            def predict(self,X):
                _,__,probs = self.forward(X)
                return np.argmax(probs[149],axis=1)

            def history(self):
                return {'TrainLoss' : self.train_loss,
                        'TrainAcc'  : self.train_acc,
                        'TestLoss'  : self.test_loss,
                        'TestAcc'   : self.test_acc}
         


        # %%
        input_dim = 3
        activations = Activations()
        metrics = Metrics()
        model = RNN(input_dim = input_dim,learning_rate = 1e-4, mom_coeff = 0.0, hidden_dim = 128)


        # %%
        model.fit(X_train,y_train,X_test,y_test,epochs = 35)


        # %%
        history = model.history()


        # %%
        plt.figure()
        plt.plot(history['TestLoss'],'-o')
        plt.plot(history['TrainLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Categorical Cross Entropy over epochs')
        plt.legend(['Test Loss','Train Loss'])
        plt.show()


        # %%
        plt.figure()
        plt.plot(history['TestAcc'],'-o')
        plt.plot(history['TrainAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Accuracy over epochs')
        plt.legend(['Test Acc','Train Acc'])
        plt.show()


        # %%
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)


        # %%
        confusion_mat_train = metrics.confusion_matrix(np.argmax(y_train,1),train_preds)
        confusion_mat_test = metrics.confusion_matrix(np.argmax(y_test,1),test_preds)


        # %%
        body_movements = ['downstairs','jogging','sitting','standing','upstairs','walking']
        confusion_mat_train_,confusion_mat_test_ = confusion_mat_train,confusion_mat_test
        confusion_mat_train.columns = body_movements
        confusion_mat_train.index = body_movements
        print(confusion_mat_train)


        # %%
        confusion_mat_test.columns = body_movements
        confusion_mat_test.index = body_movements
        print(confusion_mat_train)


        # %%
        sns.heatmap(confusion_mat_train/np.sum(confusion_mat_train), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()


        # %%
        print(confusion_mat_test)


        # %%
        sns.heatmap(confusion_mat_test/np.sum(confusion_mat_test), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()


        # %%
        plt.matshow(confusion_mat_test, cmap=plt.cm.gray_r)
        plt.title('Testing Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(confusion_mat_test.columns))
        plt.xticks(tick_marks, confusion_mat_test.columns, rotation=45)
        plt.yticks(tick_marks, confusion_mat_test.index)
        plt.tight_layout()
        plt.ylabel(confusion_mat_test.index.name)
        plt.xlabel(confusion_mat_test.columns.name)
        plt.show()


        # %%
        plt.matshow(confusion_mat_train, cmap=plt.cm.gray_r)
        plt.title('Training Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(confusion_mat_train.columns))
        plt.xticks(tick_marks, confusion_mat_train.columns, rotation=45)
        plt.yticks(tick_marks, confusion_mat_train.index)
        plt.tight_layout()
        plt.ylabel(confusion_mat_train.index.name)
        plt.xlabel(confusion_mat_train.columns.name)
        plt.show()


        # %%
        sns.heatmap(confusion_mat_test/np.sum(confusion_mat_test), annot=True, 
                    fmt='.2%',cmap = 'Greens')
        plt.show()


        # %%
        sns.heatmap(confusion_mat_test/np.sum(confusion_mat_test), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()


        # %%
        class Multi_Layer_RNN(object):
            """
            Recurrent Neural Network for classifying human activity.
            RNN encapsulates all necessary logic for training the network.

            """
            def __init__(self,input_dim = 3,hidden_dim_1 = 128, hidden_dim_2 = 64, seq_len = 150, learning_rate = 1e-1, mom_coeff = 0.85, batch_size = 32, output_class = 6):

                """

                Initialization of weights/biases and other configurable parameters.
                
                """
                np.random.seed(150)
                self.input_dim = input_dim
                self.hidden_dim_1 = hidden_dim_1
                self.hidden_dim_2 = hidden_dim_2
                # Unfold case T = 150 :
                self.seq_len = seq_len
                self.output_class = output_class
                self.learning_rate = learning_rate
                self.batch_size = batch_size
                self.mom_coeff = mom_coeff

                # Xavier uniform scaler :
                Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

                lim_inp2hid = Xavier(self.input_dim,self.hidden_dim_1)
                self.W1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(self.input_dim,self.hidden_dim_1))
                self.B1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(1,self.hidden_dim_1))

                lim_hid2hid = Xavier(self.hidden_dim_1,self.hidden_dim_1)
                self.W1_rec= np.random.uniform(-lim_hid2hid,lim_hid2hid,(self.hidden_dim_1,self.hidden_dim_1))


                lim_hid2hid2 = Xavier(self.hidden_dim_1,self.hidden_dim_2)
                self.W2 = np.random.uniform(-lim_hid2hid2,lim_hid2hid2,(self.hidden_dim_1,self.hidden_dim_2))
                self.B2 = np.random.uniform(-lim_hid2hid2,lim_hid2hid2,(1,self.hidden_dim_2))

                lim_hid2out = Xavier(self.hidden_dim_2,self.output_class)
                self.W3 = np.random.uniform(-lim_hid2out,lim_hid2out,(self.hidden_dim_2,self.output_class))
                self.B3 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(1,self.output_class))

                # To keep track loss and accuracy score :     
                self.train_loss,self.test_loss,self.train_acc,self.test_acc = [],[],[],[]
                
                # Storing previous momentum updates :
                self.prev_updates = {'W1'       : 0,
                                     'B1'       : 0,
                                     'W1_rec'   : 0,
                                     'W2'       : 0,
                                     'B2'       : 0,
                                     'W3'       : 0,
                                     'B3'       : 0}


            def forward(self,X) -> tuple:
                """
                Forward propagation of the RNN through time.
                __________________________________________________________

                Inputs:
                --- X is the bacth.
                --- h_prev_state is the previous state of the hidden layer.
                __________________________________________________________

                Returns:
                --- (X_state,hidden_state,probs) as a tuple.       
                ------ 1) X_state is the input across all time steps
                ------ 2) hidden_state is the hidden stages across time
                ------ 3) probs is the probabilities of each outputs, i.e. outputs of softmax
                __________________________________________________________
                """ 
                X_state = dict()
                hidden_state_1 = dict()
                hidden_state_mlp = dict()
                output_state = dict()
                probs = dict()
                mlp_linear = dict()
                
                self.h_prev_state = np.zeros((1,self.hidden_dim_1))
                hidden_state_1[-1] = np.copy(self.h_prev_state)

                # Loop over time T = 150 :
                for t in range(self.seq_len):

                    # Selecting first record with 3 inputs, dimension = (batch_size,input_size)
                    X_state[t] = X[:,t]

                    # Recurrent hidden layer :
                    hidden_state_1[t] = np.tanh(np.dot(X_state[t],self.W1) + np.dot(hidden_state_1[t-1],self.W1_rec) + self.B1)
                    mlp_linear[t] = np.dot(hidden_state_1[t],self.W2) + self.B2
                    hidden_state_mlp[t] = activations.ReLU(mlp_linear[t])
                    output_state[t] = np.dot(hidden_state_mlp[t],self.W3) + self.B3

                    # Per class probabilites :
                    probs[t] = activations.softmax(output_state[t])

                return (X_state,hidden_state_1,mlp_linear,hidden_state_mlp,probs)
                

            def BPTT(self,cache,Y):
                """

                Back propagation through time algorihm.
                Inputs:
                -- Cache = (X_state,hidden_state,probs)
                -- Y = desired output

                Returns:
                -- Gradients w.r.t. all configurable elements
                """

                X_state,hidden_state_1,mlp_linear,hidden_state_mlp,probs = cache

                # backward pass: compute gradients going backwards
                dW1, dW1_rec, dW2, dW3 = np.zeros_like(self.W1), np.zeros_like(self.W1_rec), np.zeros_like(self.W2),np.zeros_like(self.W3)

                dB1, dB2,dB3 = np.zeros_like(self.B1), np.zeros_like(self.B2),np.zeros_like(self.B3)

                dhnext = np.zeros_like(hidden_state_1[0])

                dy = np.copy(probs[149])        
                dy[np.arange(len(Y)),np.argmax(Y,1)] -= 1
                #dy = probs[0] - Y[0]

                dW3 += np.dot(hidden_state_mlp[149].T,dy)
                dB3 += np.sum(dy,axis = 0, keepdims = True)

                dy1 = np.dot(dy,self.W3.T) * activations.ReLU_grad(mlp_linear[149])

                dB2 += np.sum(dy1,axis = 0, keepdims = True)
                dW2 += np.dot(hidden_state_1[149].T,dy1)


                for t in reversed(range(1,self.seq_len)):

                    

                
                    dh = np.dot(dy1,self.W2.T) + dhnext        
                    dhrec = (1 - (hidden_state_1[t] * hidden_state_1[t])) * dh

                    dB1 += np.sum(dhrec,axis = 0, keepdims = True)            
                    dW1 += np.dot(X_state[t].T,dhrec)
                    
                    dW1_rec += np.dot(hidden_state_1[t-1].T,dhrec)

                    dhnext = np.dot(dhrec,self.W1_rec.T)

                       
                for grad in [dW1,dB1,dW1_rec,dW2,dB2,dW3,dB3]:
                    np.clip(grad, -10, 10, out = grad)


                return [dW1,dB1,dW1_rec,dW2,dB2,dW3,dB3]    
                
               

            def CategoricalCrossEntropy(self,labels,preds):
                """
                Computes cross entropy between labels and model's predictions
                """
                predictions = np.clip(preds, 1e-12, 1. - 1e-12)
                N = predictions.shape[0]         
                return -np.sum(labels * np.log(predictions + 1e-9)) / N

            def step(self,grads,momentum = True):

             
                #for config_param,grad in zip([self.W1,self.B1,self.W1_rec,self.W2,self.B2,self.W3,self.B3],grads):
                    #config_param -= self.learning_rate * grad

                if momentum:
                    
                    delta_W1 = -self.learning_rate * grads[0] -  self.mom_coeff * self.prev_updates['W1']
                    delta_B1 = -self.learning_rate * grads[1] -  self.mom_coeff * self.prev_updates['B1']  
                    delta_W1_rec = -self.learning_rate * grads[2] -  self.mom_coeff * self.prev_updates['W1_rec']
                    delta_W2 = -self.learning_rate * grads[3] - self.mom_coeff * self.prev_updates['W2']              
                    delta_B2 = -self.learning_rate * grads[4] -  self.mom_coeff * self.prev_updates['B2']
                    delta_W3 = -self.learning_rate * grads[5] -  self.mom_coeff * self.prev_updates['W3']
                    delta_B3 = -self.learning_rate * grads[6] -  self.mom_coeff * self.prev_updates['B3']
                    
                       
                    self.W1 += delta_W1
                    self.W1_rec += delta_W1_rec
                    self.W2 += delta_W2
                    self.B1 += delta_B1
                    self.B2 += delta_B2 
                    self.W3 += delta_W3
                    self.B3 += delta_B3    

                    
                    self.prev_updates['W1'] = delta_W1
                    self.prev_updates['W1_rec'] = delta_W1_rec
                    self.prev_updates['W2'] = delta_W2
                    self.prev_updates['B1'] = delta_B1
                    self.prev_updates['B2'] = delta_B2
                    self.prev_updates['W3'] = delta_W3
                    self.prev_updates['B3'] = delta_B3
                    
                    self.learning_rate *= 0.9999

            def fit(self,X,Y,X_val,y_val,epochs = 50 ,verbose = True, crossVal = False):
                """
                Given the traning dataset,their labels and number of epochs
                fitting the model, and measure the performance
                by validating training dataset.
                """
                        
                
                for epoch in range(epochs):
                    
                    print(f'Epoch : {epoch + 1}')

                    perm = np.random.permutation(3000)           
                    
                    for i in range(round(X.shape[0]/self.batch_size)): 

                        batch_start  =  i * self.batch_size
                        batch_finish = (i+1) * self.batch_size
                        index = perm[batch_start:batch_finish]
                        
                        X_feed = X[index]    
                        y_feed = Y[index]
                        
                        cache_train = self.forward(X_feed)                                                          
                        grads = self.BPTT(cache_train,y_feed)                
                        self.step(grads)
              
                        if crossVal:
                            stop = self.cross_validation(X,val_X,Y,val_Y,threshold = 5)
                            if stop: 
                                break
                    
                    cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[4][149])
                    predictions_train = self.predict(X)
                    acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)

                    _,__,___,____, probs_test = self.forward(X_val)
                    cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[149])
                    predictions_val = np.argmax(probs_test[149],1)
                    acc_val = metrics.accuracy(np.argmax(y_val,1),predictions_val)

                    if verbose:

                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                        print('______________________________________________________________________________________\n')                         
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                        print('______________________________________________________________________________________\n')
                        
                    self.train_loss.append(cross_loss_train)              
                    self.test_loss.append(cross_loss_val) 
                    self.train_acc.append(acc_train)              
                    self.test_acc.append(acc_val)

            def predict(self,X):
                _,__,___,____,probs = self.forward(X)
                return np.argmax(probs[149],axis=1)

            def history(self):
                return {'TrainLoss' : self.train_loss,
                        'TrainAcc'  : self.train_acc,
                        'TestLoss'  : self.test_loss,
                        'TestAcc'   : self.test_acc}
         


        # %%
        multilayer_rnn = Multi_Layer_RNN(learning_rate=1e-4,mom_coeff=0.0,hidden_dim_1 = 128, hidden_dim_2 = 64)


        # %%
        multilayer_rnn.fit(X_train,y_train,X_test,y_test,epochs = 35)


        # %%
        multilayer_rnn_history = multilayer_rnn.history()


        # %%
        plt.figure()
        plt.plot(multilayer_rnn_history['TestLoss'],'-o')
        plt.plot(multilayer_rnn_history['TrainLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Categorical Cross Entropy over epochs')
        plt.legend(['Test Loss','Train Loss'])
        plt.show()


        # %%
        plt.figure()
        plt.plot(multilayer_rnn_history['TestAcc'],'-o')
        plt.plot(multilayer_rnn_history['TrainAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Accuracy over epochs')
        plt.legend(['Test Acc','Train Acc'])
        plt.show()


        # %%
        plt.figure()
        plt.plot(multilayer_rnn_history['TrainAcc'],'-o')
        plt.plot(history['TrainAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Training Accuracy over epochs')
        plt.legend(['Multi Layer RNN','Vanilla RNN'])
        plt.show()


        # %%
        plt.plot(multilayer_rnn_history['TestAcc'],'-o')
        plt.plot(history['TestAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Accuracy over epochs')
        plt.legend(['Multi Layer RNN','Vanilla RNN'])
        plt.show()


        # %%
        train_preds_multilayer_rnn = multilayer_rnn.predict(X_train)
        test_preds_multilayer_rnn = multilayer_rnn.predict(X_test)
        confusion_mat_train_multilayer_rnn = metrics.confusion_matrix(np.argmax(y_train,1),train_preds_multilayer_rnn)
        confusion_mat_test_multilayer_rnn = metrics.confusion_matrix(np.argmax(y_test,1),test_preds_multilayer_rnn)

        body_movements = ['downstairs','jogging','sitting','standing','upstairs','walking']
        confusion_mat_train_multilayer_rnn.columns = body_movements
        confusion_mat_train_multilayer_rnn.index = body_movements
        confusion_mat_test_multilayer_rnn.columns = body_movements
        confusion_mat_test_multilayer_rnn.index = body_movements
        print(confusion_mat_train_multilayer_rnn)


        # %%
        print(confusion_mat_test_multilayer_rnn)


        # %%
        sns.heatmap(confusion_mat_test_multilayer_rnn/np.sum(confusion_mat_test_multilayer_rnn), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()


        # %%
        sns.heatmap(confusion_mat_train_multilayer_rnn/np.sum(confusion_mat_train_multilayer_rnn), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()


        # %%
        class Three_Hidden_Layer_RNN(object):
            """
            Recurrent Neural Network for classifying human activity.
            RNN encapsulates all necessary logic for training the network.

            """
            def __init__(self,input_dim = 3,hidden_dim_1 = 128, hidden_dim_2 = 64,hidden_dim_3 = 32, seq_len = 150, learning_rate = 1e-1, mom_coeff = 0.85, batch_size = 32, output_class = 6):

                """

                Initialization of weights/biases and other configurable parameters.
                
                """
                np.random.seed(150)
                self.input_dim = input_dim
                self.hidden_dim_1 = hidden_dim_1
                self.hidden_dim_2 = hidden_dim_2
                self.hidden_dim_3 = hidden_dim_3
                
                # Unfold case T = 150 :
                self.seq_len = seq_len
                self.output_class = output_class
                self.learning_rate = learning_rate
                self.batch_size = batch_size
                self.mom_coeff = mom_coeff

                # Xavier uniform scaler :
                Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

                lim_inp2hid = Xavier(self.input_dim,self.hidden_dim_1)
                self.W1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(self.input_dim,self.hidden_dim_1))
                self.B1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(1,self.hidden_dim_1))

                lim_hid2hid = Xavier(self.hidden_dim_1,self.hidden_dim_1)
                self.W1_rec= np.random.uniform(-lim_hid2hid,lim_hid2hid,(self.hidden_dim_1,self.hidden_dim_1))


                lim_hid2hid2 = Xavier(self.hidden_dim_1,self.hidden_dim_2)
                self.W2 = np.random.uniform(-lim_hid2hid2,lim_hid2hid2,(self.hidden_dim_1,self.hidden_dim_2))
                self.B2 = np.random.uniform(-lim_hid2hid2,lim_hid2hid2,(1,self.hidden_dim_2))

                lim_hid2hid3 = Xavier(self.hidden_dim_2,self.hidden_dim_3)
                self.W3 = np.random.uniform(-lim_hid2hid3,lim_hid2hid3,(self.hidden_dim_2,self.hidden_dim_3))
                self.B3 = np.random.uniform(-lim_hid2hid3,lim_hid2hid3,(1,self.hidden_dim_3))

                lim_hid2out = Xavier(self.hidden_dim_3,self.output_class)
                self.W4 = np.random.uniform(-lim_hid2out,lim_hid2out,(self.hidden_dim_3,self.output_class))
                self.B4 = np.random.uniform(-lim_hid2out,lim_hid2out,(1,self.output_class))

                # To keep track loss and accuracy score :     
                self.train_loss,self.test_loss,self.train_acc,self.test_acc = [],[],[],[]
                
                # Storing previous momentum updates :
                self.prev_updates = {'W1'       : 0,
                                     'B1'       : 0,
                                     'W1_rec'   : 0,
                                     'W2'       : 0,
                                     'B2'       : 0,
                                     'W3'       : 0,
                                     'W4'       : 0,
                                     'B3'       : 0,
                                     'B4'       : 0}


            def forward(self,X) -> tuple:
                """
                Forward propagation of the RNN through time.
                __________________________________________________________

                Inputs:
                --- X is the bacth.
                --- h_prev_state is the previous state of the hidden layer.
                __________________________________________________________

                Returns:
                --- (X_state,hidden_state,probs) as a tuple.       
                ------ 1) X_state is the input across all time steps
                ------ 2) hidden_state is the hidden stages across time
                ------ 3) probs is the probabilities of each outputs, i.e. outputs of softmax
                __________________________________________________________
                """ 
                X_state = dict()
                hidden_state_1 = dict()
                hidden_state_mlp = dict()
                hidden_state_mlp_2 = dict()
                output_state = dict()
                probs = dict()
                mlp_linear = dict()
                mlp_linear_2 = dict()
                
                self.h_prev_state = np.zeros((1,self.hidden_dim_1))
                hidden_state_1[-1] = np.copy(self.h_prev_state)

                # Loop over time T = 150 :
                for t in range(self.seq_len):

                    # Selecting first record with 3 inputs, dimension = (batch_size,input_size)
                    X_state[t] = X[:,t]

                    # Recurrent hidden layer :
                    hidden_state_1[t] = np.tanh(np.dot(X_state[t],self.W1) + np.dot(hidden_state_1[t-1],self.W1_rec) + self.B1)
                    mlp_linear[t] = np.dot(hidden_state_1[t],self.W2) + self.B2
                    hidden_state_mlp[t] = activations.ReLU(mlp_linear[t])
                    mlp_linear_2[t] = np.dot(hidden_state_mlp[t],self.W3) + self.B3
                    hidden_state_mlp_2[t] = activations.ReLU(mlp_linear_2[t])
                    output_state[t] = np.dot(hidden_state_mlp_2[t],self.W4) + self.B4

                    # Per class probabilites :
                    probs[t] = activations.softmax(output_state[t])

                return (X_state,hidden_state_1,mlp_linear,hidden_state_mlp,mlp_linear_2,hidden_state_mlp_2,probs)
                

            def BPTT(self,cache,Y):
                """

                Back propagation through time algorihm.
                Inputs:
                -- Cache = (X_state,hidden_state,probs)
                -- Y = desired output

                Returns:
                -- Gradients w.r.t. all configurable elements
                """

                X_state,hidden_state_1,mlp_linear,hidden_state_mlp,mlp_linear_2,hidden_state_mlp_2,probs = cache

                # backward pass: compute gradients going backwards
                dW1, dW1_rec, dW2, dW3, dW4 = np.zeros_like(self.W1), np.zeros_like(self.W1_rec), np.zeros_like(self.W2),np.zeros_like(self.W3),np.zeros_like(self.W4)

                dB1, dB2,dB3,dB4 = np.zeros_like(self.B1), np.zeros_like(self.B2),np.zeros_like(self.B3),np.zeros_like(self.B4)

                dhnext = np.zeros_like(hidden_state_1[0])

                for t in reversed(range(1,self.seq_len)):

                    dy = np.copy(probs[t])        
                    dy[np.arange(len(Y)),np.argmax(Y,1)] -= 1
                    #dy = probs[0] - Y[0]

                    dW4 += np.dot(hidden_state_mlp_2[t].T,dy)
                    dB4 += np.sum(dy,axis = 0, keepdims = True)

                    dy1 = np.dot(dy,self.W4.T) * activations.ReLU_grad(mlp_linear_2[t])

                    dW3 += np.dot(hidden_state_mlp[t].T,dy1)
                    dB3 += np.sum(dy1,axis = 0, keepdims = True)

                    dy2 = np.dot(dy1,self.W3.T) * activations.ReLU_grad(mlp_linear[t])

                    dB2 += np.sum(dy2,axis = 0, keepdims = True)
                    dW2 += np.dot(hidden_state_1[t].T,dy2)
                
                    dh = np.dot(dy2,self.W2.T) + dhnext        
                    dhrec = (1 - (hidden_state_1[t] * hidden_state_1[t])) * dh

                    dB1 += np.sum(dhrec,axis = 0, keepdims = True)            
                    dW1 += np.dot(X_state[t].T,dhrec)
                    
                    dW1_rec += np.dot(hidden_state_1[t-1].T,dhrec)

                    dhnext = np.dot(dhrec,self.W1_rec.T)

                       
                for grad in [dW1,dB1,dW1_rec,dW2,dB2,dW3,dB3,dW4,dB4]:
                    np.clip(grad, -10, 10, out = grad)


                return [dW1,dB1,dW1_rec,dW2,dB2,dW3,dB3,dW4,dB4]    
                
               

            def CategoricalCrossEntropy(self,labels,preds):
                """
                Computes cross entropy between labels and model's predictions
                """
                predictions = np.clip(preds, 1e-12, 1. - 1e-12)
                N = predictions.shape[0]         
                return -np.sum(labels * np.log(predictions + 1e-9)) / N

            def step(self,grads,momentum = True):

             
                #for config_param,grad in zip([self.W1,self.B1,self.W1_rec,self.W2,self.B2,self.W3,self.B3],grads):
                    #config_param -= self.learning_rate * grad

                if momentum:
                    
                    delta_W1 = -self.learning_rate * grads[0] +  self.mom_coeff * self.prev_updates['W1']
                    delta_B1 = -self.learning_rate * grads[1] +  self.mom_coeff * self.prev_updates['B1']  
                    delta_W1_rec = -self.learning_rate * grads[2] +  self.mom_coeff * self.prev_updates['W1_rec']
                    delta_W2 = -self.learning_rate * grads[3] +  self.mom_coeff * self.prev_updates['W2']              
                    delta_B2 = -self.learning_rate * grads[4] +  self.mom_coeff * self.prev_updates['B2']
                    delta_W3 = -self.learning_rate * grads[5] +  self.mom_coeff * self.prev_updates['W3']              
                    delta_B3 = -self.learning_rate * grads[6] +  self.mom_coeff * self.prev_updates['B3']
                    delta_W4 = -self.learning_rate * grads[7] +  self.mom_coeff * self.prev_updates['W4']              
                    delta_B4 = -self.learning_rate * grads[8] +  self.mom_coeff * self.prev_updates['B4']
                    
                       
                    self.W1 += delta_W1
                    self.W1_rec += delta_W1_rec
                    self.W2 += delta_W2
                    self.B1 += delta_B1
                    self.B2 += delta_B2 
                    self.W3 += delta_W3
                    self.B3 += delta_B3   
                    self.W4 += delta_W4
                    self.B4 += delta_B4

                    
                    self.prev_updates['W1'] = delta_W1
                    self.prev_updates['W1_rec'] = delta_W1_rec
                    self.prev_updates['W2'] = delta_W2
                    self.prev_updates['B1'] = delta_B1
                    self.prev_updates['B2'] = delta_B2
                    self.prev_updates['W3'] = delta_W3
                    self.prev_updates['B3'] = delta_B3
                    self.prev_updates['W4'] = delta_W4
                    self.prev_updates['B4'] = delta_B4

                    self.learning_rate *= 0.9999

            def fit(self,X,Y,X_val,y_val,epochs = 50 ,verbose = True, crossVal = False):
                """
                Given the traning dataset,their labels and number of epochs
                fitting the model, and measure the performance
                by validating training dataset.
                """
                        
                
                for epoch in range(epochs):
                    
                    print(f'Epoch : {epoch + 1}')

                    perm = np.random.permutation(3000)           
                    
                    for i in range(round(X.shape[0]/self.batch_size)): 

                        batch_start  =  i * self.batch_size
                        batch_finish = (i+1) * self.batch_size
                        index = perm[batch_start:batch_finish]
                        
                        X_feed = X[index]    
                        y_feed = Y[index]
                        
                        cache_train = self.forward(X_feed)                                                          
                        grads = self.BPTT(cache_train,y_feed)                
                        self.step(grads)
              
                        if crossVal:
                            stop = self.cross_validation(X,val_X,Y,val_Y,threshold = 5)
                            if stop: 
                                break
                    
                    cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[6][149])
                    predictions_train = self.predict(X)
                    acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)

                    _,__,___,____,_____,______, probs_test = self.forward(X_val)
                    cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[149])
                    predictions_val = np.argmax(probs_test[149],1)
                    acc_val = metrics.accuracy(np.argmax(y_val,1),predictions_val)

                    if verbose:

                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                        print('______________________________________________________________________________________\n')                         
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                        print('______________________________________________________________________________________\n')
                        
                    self.train_loss.append(cross_loss_train)              
                    self.test_loss.append(cross_loss_val) 
                    self.train_acc.append(acc_train)              
                    self.test_acc.append(acc_val)

            def predict(self,X):
                _,__,___,____,_____,______,probs = self.forward(X)
                return np.argmax(probs[149],axis=1)

            def history(self):
                return {'TrainLoss' : self.train_loss,
                        'TrainAcc'  : self.train_acc,
                        'TestLoss'  : self.test_loss,
                        'TestAcc'   : self.test_acc}
         


        # %%
        three_layer_rnn = Three_Hidden_Layer_RNN(hidden_dim_1 = 128, hidden_dim_2 = 64,hidden_dim_3 = 32, learning_rate = 1e-4, mom_coeff = 0.0, batch_size = 32, output_class = 6)


        # %%
        three_layer_rnn.fit(X_train,y_train,X_test,y_test,epochs=15)


        # %%
        three_layer_rnn_v1 = Three_Hidden_Layer_RNN(hidden_dim_1 = 128, hidden_dim_2 = 64,hidden_dim_3 = 32, learning_rate = 5e-5, mom_coeff = 0.0, batch_size = 32, output_class = 6)
        three_layer_rnn_v1.fit(X_train,y_train,X_test,y_test,epochs=15)


        # %%
        three_layer_rnn_v2 = Three_Hidden_Layer_RNN(hidden_dim_1 = 128, hidden_dim_2 = 64,hidden_dim_3 = 32, learning_rate = 1e-4, mom_coeff = 0.0, batch_size = 32, output_class = 6)
        three_layer_rnn_v2.fit(X_train,y_train,X_test,y_test,epochs=15)


        # %%
        three_layer_rnn_history = three_layer_rnn.history()
        plt.figure()
        plt.plot(three_layer_rnn_history['TestLoss'],'-o')
        plt.plot(three_layer_rnn_history['TrainLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Categorical Cross Entropy over epochs')
        plt.legend(['Test Loss','Train Loss'])
        plt.show()


        # %%
        plt.figure()
        plt.plot(three_layer_rnn_history['TestAcc'],'-o')
        plt.plot(three_layer_rnn_history['TrainAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Accuracy over epochs')
        plt.legend(['Test Acc','Train Acc'])
        plt.show()


        # %%
        plt.figure()
        plt.plot(three_layer_rnn_history['TrainAcc'],'-o')
        plt.plot(multilayer_rnn_history['TrainAcc'],'-o')
        plt.plot(history['TrainAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Training Accuracy over epochs')
        plt.legend(['3 hidden layer Rnn','Multi Layer RNN','Vanilla RNN'])
        plt.show()


        # %%
        plt.figure()
        plt.plot(three_layer_rnn_history['TestAcc'],'-o')
        plt.plot(multilayer_rnn_history['TestAcc'],'-o')
        plt.plot(history['TestAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Accuracy over epochs')
        plt.legend(['3 hidden layer Rnn','Multi Layer RNN','Vanilla RNN'])
        plt.show()


        # %%
        train_preds_three_layer_rnn_history = three_layer_rnn.predict(X_train)
        test_preds_three_layer_rnn_history = three_layer_rnn.predict(X_test)
        confusion_mat_train_three_layer_rnn_history = metrics.confusion_matrix(np.argmax(y_train,1),train_preds_three_layer_rnn_history)
        confusion_mat_test_three_layer_rnn_history = metrics.confusion_matrix(np.argmax(y_test,1),test_preds_three_layer_rnn_history)

        body_movements = ['downstairs','jogging','sitting','standing','upstairs','walking']
        confusion_mat_train_three_layer_rnn_history.columns = body_movements
        confusion_mat_train_three_layer_rnn_history.index = body_movements
        confusion_mat_test_three_layer_rnn_history.columns = body_movements
        confusion_mat_test_three_layer_rnn_history.index = body_movements
        print(confusion_mat_train_three_layer_rnn_history)


        # %%
        sns.heatmap(confusion_mat_test_three_layer_rnn_history/np.sum(confusion_mat_test_three_layer_rnn_history), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()


        # %%
        sns.heatmap(confusion_mat_train_three_layer_rnn_history/np.sum(confusion_mat_train_three_layer_rnn_history), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()


        # %%
        class Five_Hidden_Layer_RNN(object):
            """
            Recurrent Neural Network for classifying human activity.
            RNN encapsulates all necessary logic for training the network.

            """
            def __init__(self,input_dim = 3,hidden_dim_1 = 128, hidden_dim_2 = 64,hidden_dim_3 = 32,hidden_dim_4 = 16 ,hidden_dim_5 = 8, seq_len = 150, learning_rate = 1e-1, mom_coeff = 0.85, batch_size = 32, output_class = 6):

                """

                Initialization of weights/biases and other configurable parameters.
                
                """
                np.random.seed(150)
                self.input_dim = input_dim
                self.hidden_dim_1 = hidden_dim_1
                self.hidden_dim_2 = hidden_dim_2
                self.hidden_dim_3 = hidden_dim_3
                self.hidden_dim_4 = hidden_dim_4
                self.hidden_dim_5 = hidden_dim_5
                
                # Unfold case T = 150 :
                self.seq_len = seq_len
                self.output_class = output_class
                self.learning_rate = learning_rate
                self.batch_size = batch_size
                self.mom_coeff = mom_coeff

                # Xavier uniform scaler :
                Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

                lim_inp2hid = Xavier(self.input_dim,self.hidden_dim_1)
                self.W1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(self.input_dim,self.hidden_dim_1))
                self.B1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(1,self.hidden_dim_1))

                lim_hid2hid = Xavier(self.hidden_dim_1,self.hidden_dim_1)
                self.W1_rec= np.random.uniform(-lim_hid2hid,lim_hid2hid,(self.hidden_dim_1,self.hidden_dim_1))


                lim_hid2hid2 = Xavier(self.hidden_dim_1,self.hidden_dim_2)
                self.W2 = np.random.uniform(-lim_hid2hid2,lim_hid2hid2,(self.hidden_dim_1,self.hidden_dim_2))
                self.B2 = np.random.uniform(-lim_hid2hid2,lim_hid2hid2,(1,self.hidden_dim_2))

                lim_hid2hid3 = Xavier(self.hidden_dim_2,self.hidden_dim_3)
                self.W3 = np.random.uniform(-lim_hid2hid3,lim_hid2hid3,(self.hidden_dim_2,self.hidden_dim_3))
                self.B3 = np.random.uniform(-lim_hid2hid3,lim_hid2hid3,(1,self.hidden_dim_3))

                lim_hid2hid4 = Xavier(self.hidden_dim_3,self.hidden_dim_4)
                self.W4 = np.random.uniform(-lim_hid2hid4,lim_hid2hid4,(self.hidden_dim_3,self.hidden_dim_4))
                self.B4 = np.random.uniform(-lim_hid2hid4,lim_hid2hid4,(1,self.hidden_dim_4))

                lim_hid2hid5 = Xavier(self.hidden_dim_4,self.hidden_dim_5)
                self.W5 = np.random.uniform(-lim_hid2hid5,lim_hid2hid5,(self.hidden_dim_4,self.hidden_dim_5))
                self.B5 = np.random.uniform(-lim_hid2hid5,lim_hid2hid5,(1,self.hidden_dim_5))


                lim_hid2out = Xavier(self.hidden_dim_5,self.output_class)
                self.W6 = np.random.uniform(-lim_hid2out,lim_hid2out,(self.hidden_dim_5,self.output_class))
                self.B6 = np.random.uniform(-lim_hid2out,lim_hid2out,(1,self.output_class))

                # To keep track loss and accuracy score :     
                self.train_loss,self.test_loss,self.train_acc,self.test_acc = [],[],[],[]
                
                # Storing previous momentum updates :
                self.prev_updates = {'W1'       : 0,
                                     'B1'       : 0,
                                     'W1_rec'   : 0,
                                     'W2'       : 0,
                                     'B2'       : 0,
                                     'W3'       : 0,
                                     'W4'       : 0,
                                     'B3'       : 0,
                                     'B4'       : 0,
                                     'W5'       : 0,
                                     'W6'       : 0,
                                     'B5'       : 0,
                                     'B6'       : 0}


            def forward(self,X) -> tuple:
                """
                Forward propagation of the RNN through time.
                __________________________________________________________

                Inputs:
                --- X is the bacth.
                --- h_prev_state is the previous state of the hidden layer.
                __________________________________________________________

                Returns:
                --- (X_state,hidden_state,probs) as a tuple.       
                ------ 1) X_state is the input across all time steps
                ------ 2) hidden_state is the hidden stages across time
                ------ 3) probs is the probabilities of each outputs, i.e. outputs of softmax
                __________________________________________________________
                """ 
                X_state = dict()
                hidden_state_1 = dict()
                hidden_state_mlp = dict()
                hidden_state_mlp_2 = dict()
                hidden_state_mlp_3 = dict()
                hidden_state_mlp_4 = dict()
                output_state = dict()
                probs = dict()
                mlp_linear = dict()
                mlp_linear_2 = dict()
                mlp_linear_3 = dict()
                mlp_linear_4 = dict()
                
                self.h_prev_state = np.zeros((1,self.hidden_dim_1))
                hidden_state_1[-1] = np.copy(self.h_prev_state)

                # Loop over time T = 150 :
                for t in range(self.seq_len):

                    # Selecting first record with 3 inputs, dimension = (batch_size,input_size)
                    X_state[t] = X[:,t]

                    # Recurrent hidden layer :
                    hidden_state_1[t] = np.tanh(np.dot(X_state[t],self.W1) + np.dot(hidden_state_1[t-1],self.W1_rec) + self.B1)
                    mlp_linear[t] = np.dot(hidden_state_1[t],self.W2) + self.B2

                    hidden_state_mlp[t] = activations.ReLU(mlp_linear[t])

                    mlp_linear_2[t] = np.dot(hidden_state_mlp[t],self.W3) + self.B3
                    hidden_state_mlp_2[t] = activations.ReLU(mlp_linear_2[t])

                    mlp_linear_3[t] = np.dot(hidden_state_mlp_2[t],self.W4) + self.B4
                    hidden_state_mlp_3[t] = activations.ReLU(mlp_linear_3[t])

                    mlp_linear_4[t] = np.dot(hidden_state_mlp_3[t],self.W5) + self.B5
                    hidden_state_mlp_4[t] = activations.ReLU(mlp_linear_4[t])

                    output_state[t] = np.dot(hidden_state_mlp_4[t],self.W6) + self.B6

                    # Per class probabilites :
                    probs[t] = activations.softmax(output_state[t])

                return (X_state,hidden_state_1,mlp_linear,hidden_state_mlp,mlp_linear_2,hidden_state_mlp_2,mlp_linear_3,hidden_state_mlp_3,mlp_linear_4,hidden_state_mlp_4,probs)
                

            def BPTT(self,cache,Y):
                """

                Back propagation through time algorihm.
                Inputs:
                -- Cache = (X_state,hidden_state_1,mlp_linear,hidden_state_mlp,mlp_linear_2,hidden_state_mlp_2,mlp_linear_3,hidden_state_mlp_3,mlp_linear_4,hidden_state_mlp_4,probs)
                -- Y = desired output

                Returns:
                -- Gradients w.r.t. all configurable elements
                """

                X_state,hidden_state_1,mlp_linear,hidden_state_mlp,mlp_linear_2,hidden_state_mlp_2,mlp_linear_3,hidden_state_mlp_3,mlp_linear_4,hidden_state_mlp_4,probs = cache

                # backward pass: compute gradients going backwards
                dW1, dW1_rec, dW2, dW3, dW4, dW5, dW6 = np.zeros_like(self.W1), np.zeros_like(self.W1_rec), np.zeros_like(self.W2),np.zeros_like(self.W3),np.zeros_like(self.W4),np.zeros_like(self.W5),np.zeros_like(self.W6)

                dB1, dB2,dB3,dB4,dB5,dB6 = np.zeros_like(self.B1), np.zeros_like(self.B2),np.zeros_like(self.B3),np.zeros_like(self.B4),np.zeros_like(self.B5),np.zeros_like(self.B6)

                dhnext = np.zeros_like(hidden_state_1[0])


                for t in reversed(range(1,self.seq_len)):
                    
                    dy = np.copy(probs[149])        
                    dy[np.arange(len(Y)),np.argmax(Y,1)] -= 1
                    #dy = probs[0] - Y[0]

                    dW6 += np.dot(hidden_state_mlp_4[t].T,dy)
                    dB6 += np.sum(dy,axis = 0, keepdims = True)

                    dy1 = np.dot(dy,self.W6.T) * activations.ReLU_grad(mlp_linear_4[t])

                    dW5 += np.dot(hidden_state_mlp_3[t].T,dy1)
                    dB5 += np.sum(dy1,axis = 0, keepdims = True)

                    dy2 = np.dot(dy1,self.W5.T) * activations.ReLU_grad(mlp_linear_3[t])

                    dW4 += np.dot(hidden_state_mlp_2[t].T,dy2)
                    dB4 += np.sum(dy2,axis = 0, keepdims = True)

                    dy3 = np.dot(dy2,self.W4.T) * activations.ReLU_grad(mlp_linear_2[t])

                    dW3 += np.dot(hidden_state_mlp[t].T,dy3)
                    dB3 += np.sum(dy3,axis = 0, keepdims = True)

                    dy4 = np.dot(dy3,self.W3.T) * activations.ReLU_grad(mlp_linear[t])

                    dB2 += np.sum(dy4,axis = 0, keepdims = True)
                    dW2 += np.dot(hidden_state_1[t].T,dy4)            
                
                    dh = np.dot(dy4,self.W2.T) + dhnext        
                    dhrec = (1 - (hidden_state_1[t] * hidden_state_1[t])) * dh

                    dB1 += np.sum(dhrec,axis = 0, keepdims = True)            
                    dW1 += np.dot(X_state[t].T,dhrec)
                    
                    dW1_rec += np.dot(hidden_state_1[t-1].T,dhrec)

                    dhnext = np.dot(dhrec,self.W1_rec.T)

                       
                for grad in [dW1,dB1,dW1_rec,dW2,dB2,dW3,dB3,dW4,dB4,dW5,dB5,dW6,dB6]:
                    np.clip(grad, -10, 10, out = grad)


                return [dW1,dB1,dW1_rec,dW2,dB2,dW3,dB3,dW4,dB4,dW5,dB5,dW6,dB6]    
                
               

            def CategoricalCrossEntropy(self,labels,preds):
                """
                Computes cross entropy between labels and model's predictions
                """
                predictions = np.clip(preds, 1e-12, 1. - 1e-12)
                N = predictions.shape[0]         
                return -np.sum(labels * np.log(predictions + 1e-9)) / N

            def step(self,grads,momentum = True):

             
                #for config_param,grad in zip([self.W1,self.B1,self.W1_rec,self.W2,self.B2,self.W3,self.B3],grads):
                    #config_param -= self.learning_rate * grad

                if momentum:
                    
                    delta_W1 = -self.learning_rate * grads[0] +  self.mom_coeff * self.prev_updates['W1']
                    delta_B1 = -self.learning_rate * grads[1] +  self.mom_coeff * self.prev_updates['B1']  
                    delta_W1_rec = -self.learning_rate * grads[2] +  self.mom_coeff * self.prev_updates['W1_rec']
                    delta_W2 = -self.learning_rate * grads[3] +  self.mom_coeff * self.prev_updates['W2']              
                    delta_B2 = -self.learning_rate * grads[4] +  self.mom_coeff * self.prev_updates['B2']
                    delta_W3 = -self.learning_rate * grads[5] +  self.mom_coeff * self.prev_updates['W3']              
                    delta_B3 = -self.learning_rate * grads[6] +  self.mom_coeff * self.prev_updates['B3']
                    delta_W4 = -self.learning_rate * grads[7] +  self.mom_coeff * self.prev_updates['W4']              
                    delta_B4 = -self.learning_rate * grads[8] +  self.mom_coeff * self.prev_updates['B4']
                    delta_W5 = -self.learning_rate * grads[9] +  self.mom_coeff * self.prev_updates['W5']              
                    delta_B5 = -self.learning_rate * grads[10] +  self.mom_coeff * self.prev_updates['B5']
                    delta_W6 = -self.learning_rate * grads[11] +  self.mom_coeff * self.prev_updates['W6']              
                    delta_B6 = -self.learning_rate * grads[12] +  self.mom_coeff * self.prev_updates['B6']
                    
                       
                    self.W1 += delta_W1
                    self.W1_rec += delta_W1_rec
                    self.W2 += delta_W2
                    self.B1 += delta_B1
                    self.B2 += delta_B2 
                    self.W3 += delta_W3
                    self.B3 += delta_B3   
                    self.W4 += delta_W4
                    self.B4 += delta_B4
                    self.W5 += delta_W5
                    self.B5 += delta_B5
                    self.W6 += delta_W6
                    self.B6 += delta_B6

                    
                    self.prev_updates['W1'] = delta_W1
                    self.prev_updates['W1_rec'] = delta_W1_rec
                    self.prev_updates['W2'] = delta_W2
                    self.prev_updates['B1'] = delta_B1
                    self.prev_updates['B2'] = delta_B2
                    self.prev_updates['W3'] = delta_W3
                    self.prev_updates['B3'] = delta_B3
                    self.prev_updates['W4'] = delta_W4
                    self.prev_updates['B4'] = delta_B4
                    self.prev_updates['W5'] = delta_W5
                    self.prev_updates['B5'] = delta_B5
                    self.prev_updates['W6'] = delta_W6
                    self.prev_updates['B6'] = delta_B6

                    self.learning_rate *= 0.9999

            def fit(self,X,Y,X_val,y_val,epochs = 50 ,verbose = True, crossVal = False):
                """
                Given the traning dataset,their labels and number of epochs
                fitting the model, and measure the performance
                by validating training dataset.
                """
                        
                
                for epoch in range(epochs):
                    
                    print(f'Epoch : {epoch + 1}')

                    perm = np.random.permutation(3000)           
                    
                    for i in range(round(X.shape[0]/self.batch_size)): 

                        batch_start  =  i * self.batch_size
                        batch_finish = (i+1) * self.batch_size
                        index = perm[batch_start:batch_finish]
                        
                        X_feed = X[index]    
                        y_feed = Y[index]
                        
                        cache_train = self.forward(X_feed)                                                          
                        grads = self.BPTT(cache_train,y_feed)                
                        self.step(grads)
              
                        if crossVal:
                            stop = self.cross_validation(X,val_X,Y,val_Y,threshold = 5)
                            if stop: 
                                break
                    
                    cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[10][149])
                    predictions_train = self.predict(X)
                    acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)
                                                                          
                    _,__,___,____,_____,______,_______,________,__________,___________, probs_test = self.forward(X_val)
                    cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[149])
                    predictions_val = np.argmax(probs_test[149],1)
                    acc_val = metrics.accuracy(np.argmax(y_val,1),predictions_val)

                    if verbose:

                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                        print('______________________________________________________________________________________\n')                         
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                        print('______________________________________________________________________________________\n')
                        
                    self.train_loss.append(cross_loss_train)              
                    self.test_loss.append(cross_loss_val) 
                    self.train_acc.append(acc_train)              
                    self.test_acc.append(acc_val)

            def predict(self,X):
                _,__,___,____,_____,______,_______,________,__________,___________, probs = self.forward(X)
                return np.argmax(probs[149],axis=1)

            def history(self):
                return {'TrainLoss' : self.train_loss,
                        'TrainAcc'  : self.train_acc,
                        'TestLoss'  : self.test_loss,
                        'TestAcc'   : self.test_acc}
         


        # %%
        five_hidden_layer_rnn = Five_Hidden_Layer_RNN(hidden_dim_1 = 128, hidden_dim_2 = 64,hidden_dim_3 = 32,hidden_dim_4 = 16 ,hidden_dim_5 = 8,  learning_rate = 1e-4, mom_coeff = 0.0)


        # %%
        five_hidden_layer_rnn.fit(X_train,y_train,X_test,y_test,epochs = 35)


        # %%
        five_hidden_layer_rnn_history = five_hidden_layer_rnn.history()
        plt.figure()
        plt.plot(five_hidden_layer_rnn_history['TestLoss'],'-o')
        plt.plot(five_hidden_layer_rnn_history['TrainLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Categorical Cross Entropy over epochs')
        plt.legend(['Test Loss','Train Loss'])
        plt.show()


        # %%
        plt.figure()
        plt.plot(five_hidden_layer_rnn_history['TestAcc'],'-o')
        plt.plot(five_hidden_layer_rnn_history['TrainAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Accuracy over epochs')
        plt.legend(['Test Acc','Train Acc'])
        plt.show()


        # %%
        plt.figure()
        plt.plot(five_hidden_layer_rnn_history['TrainAcc'],'-o')
        plt.plot(three_layer_rnn_history['TrainAcc'],'-o')
        plt.plot(multilayer_rnn_history['TrainAcc'],'-o')
        plt.plot(history['TrainAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Training Accuracy over epochs')
        plt.legend(['Five hidden layer RNN','3 hidden layer RNN','Multi Layer RNN','Vanilla RNN'])
        plt.show()


        # %%
        plt.figure()
        plt.plot(five_hidden_layer_rnn_history['TestAcc'],'-o')
        plt.plot(three_layer_rnn_history['TestAcc'],'-o')
        plt.plot(multilayer_rnn_history['TestAcc'],'-o')
        plt.plot(history['TestAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Accuracy over epochs')
        plt.legend(['Five hidden layer RNN','3 hidden layer RNN','Multi Layer RNN','Vanilla RNN'])
        plt.show()


        # %%
        train_preds_five_hidden_layer_rnn = five_hidden_layer_rnn.predict(X_train)
        test_preds_five_hidden_layer_rnn = five_hidden_layer_rnn.predict(X_test)
        confusion_mat_train_five_hidden_layer_rnn = metrics.confusion_matrix(np.argmax(y_train,1),train_preds_five_hidden_layer_rnn)
        confusion_mat_test_five_hidden_layer_rnn = metrics.confusion_matrix(np.argmax(y_test,1),test_preds_five_hidden_layer_rnn)

        body_movements = ['downstairs','jogging','sitting','standing','upstairs','walking']
        confusion_mat_train_five_hidden_layer_rnn.columns = body_movements
        confusion_mat_train_five_hidden_layer_rnn.index = body_movements
        confusion_mat_test_five_hidden_layer_rnn.columns = body_movements
        confusion_mat_test_five_hidden_layer_rnn.index = body_movements
        print(confusion_mat_test_five_hidden_layer_rnn)


        # %%
        sns.heatmap(confusion_mat_test_five_hidden_layer_rnn/np.sum(confusion_mat_test_five_hidden_layer_rnn), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()


        # %%
        sns.heatmap(confusion_mat_train_five_hidden_layer_rnn/np.sum(confusion_mat_train_five_hidden_layer_rnn), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()

        # %% [markdown]
        # LSTM

        # %%
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))


        def dsigmoid(y):
            return y * (1 - y)


        def tanh(x):
            return np.tanh(x)


        def dtanh(y):
            return 1 - y * y


        # %%
        class LSTM(object):
            """

            Long-Short Term Memory Recurrent neural network, encapsulates all necessary logic for training, then built the hyperparameters and architecture of the network.
            """

            def __init__(self,input_dim = 3,hidden_dim = 100,output_class = 6,seq_len = 150,batch_size = 30,learning_rate = 1e-1,mom_coeff = 0.85):
                """

                Initialization of weights/biases and other configurable parameters.
                
                """
                np.random.seed(150)
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim

                # Unfold case T = 150 :
                self.seq_len = seq_len
                self.output_class = output_class
                self.learning_rate = learning_rate
                self.batch_size = batch_size
                self.mom_coeff = mom_coeff

                self.input_stack_dim = self.input_dim + self.hidden_dim
                
                # Xavier uniform scaler :
                Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

                lim1 = Xavier(self.input_dim,self.hidden_dim)
                self.W_f = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim))
                self.B_f = np.random.uniform(-lim1,lim1,(1,self.hidden_dim))

                self.W_i = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim))
                self.B_i = np.random.uniform(-lim1,lim1,(1,self.hidden_dim))

                self.W_c = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim))
                self.B_c = np.random.uniform(-lim1,lim1,(1,self.hidden_dim))

                self.W_o = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim))
                self.B_o = np.random.uniform(-lim1,lim1,(1,self.hidden_dim))
                
                lim2 = Xavier(self.hidden_dim,self.output_class)
                self.W = np.random.uniform(-lim2,lim2,(self.hidden_dim,self.output_class))
                self.B = np.random.uniform(-lim2,lim2,(1,self.output_class))

                # To keep track loss and accuracy score :     
                self.train_loss,self.test_loss,self.train_acc,self.test_acc = [],[],[],[]
                
                # To keep previous updates in momentum :
                self.previous_updates = [0] * 10
                
                # For AdaGrad:
                self.cache = [0] * 10     
                self.cache_rmsprop = [0] * 10
                self.m = [0] * 10
                self.v = [0] * 10
                self.t = 1

            def cell_forward(self,X,h_prev,C_prev):
                """

                Takes input, previous hidden state and previous cell state, compute:
                --- Forget gate + Input gate + New candidate input + New cell state + 
                    output gate + hidden state. Then, classify by softmax.
                """
                #print(X.shape,h_prev.shape)
                # Stacking previous hidden state vector with inputs:
                stack = np.column_stack([X,h_prev])

                # Forget gate:
                forget_gate = activations.sigmoid(np.dot(stack,self.W_f) + self.B_f)
               
                # İnput gate:
                input_gate = activations.sigmoid(np.dot(stack,self.W_i) + self.B_i)

                # New candidate:
                cell_bar = np.tanh(np.dot(stack,self.W_c) + self.B_c)

                # New Cell state:
                cell_state = forget_gate * C_prev + input_gate * cell_bar

                # Output fate:
                output_gate = activations.sigmoid(np.dot(stack,self.W_o) + self.B_o)

                # Hidden state:
                hidden_state = output_gate * np.tanh(cell_state)

                # Classifiers (Softmax) :
                dense = np.dot(hidden_state,self.W) + self.B
                probs = activations.softmax(dense)

                return (stack,forget_gate,input_gate,cell_bar,cell_state,output_gate,hidden_state,dense,probs)

                

            def forward(self,X,h_prev,C_prev):
                x_s,z_s,f_s,i_s = {},{},{},{}
                C_bar_s,C_s,o_s,h_s = {},{},{},{}
                v_s,y_s = {},{}


                h_s[-1] = np.copy(h_prev)
                C_s[-1] = np.copy(C_prev)

                for t in range(self.seq_len):
                    x_s[t] = X[:,t,:]
                    z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t],v_s[t], y_s[t] = self.cell_forward(x_s[t],h_s[t-1],C_s[t-1])

                return (z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s,v_s, y_s)
            
            def BPTT(self,outs,Y):

                z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s,v_s, y_s = outs

                dW_f, dW_i,dW_c, dW_o,dW = np.zeros_like(self.W_f), np.zeros_like(self.W_i), np.zeros_like(self.W_c),np.zeros_like(self.W_o),np.zeros_like(self.W)

                dB_f, dB_i,dB_c,dB_o,dB = np.zeros_like(self.B_f), np.zeros_like(self.B_i),np.zeros_like(self.B_c),np.zeros_like(self.B_o),np.zeros_like(self.B)

                dh_next = np.zeros_like(h_s[0]) 
                dC_next = np.zeros_like(C_s[0])   

                # w.r.t. softmax input
                ddense = np.copy(y_s[149])
                ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1
                #ddense[np.argmax(Y,1)] -=1
                #ddense = y_s[149] - Y
                # Softmax classifier's :
                dW = np.dot(h_s[149].T,ddense)
                dB = np.sum(ddense,axis = 0, keepdims = True)

                # Backprop through time:
                for t in reversed(range(1,self.seq_len)):           
                    
                    # Just equating more meaningful names
                    stack,forget_gate,input_gate,cell_bar,cell_state,output_gate,hidden_state,dense,probs = z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t],v_s[t], y_s[t]
                    C_prev = C_s[t-1]
                    
                    # w.r.t. softmax input
                    #ddense = np.copy(probs)
                    #ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1
                    #ddense[np.arange(len(Y)),np.argmax(Y,1)] -=1
                    # Softmax classifier's :
                    #dW += np.dot(hidden_state.T,ddense)
                    #dB += np.sum(ddense,axis = 0, keepdims = True)

                    # Output gate :
                    dh = np.dot(ddense,self.W.T) + dh_next            
                    do = dh * np.tanh(cell_state)
                    do = do * dsigmoid(output_gate)
                    dW_o += np.dot(stack.T,do)
                    dB_o += np.sum(do,axis = 0, keepdims = True)

                    # Cell state:
                    dC = np.copy(dC_next)
                    dC += dh * output_gate * activations.dtanh(cell_state)
                    dC_bar = dC * input_gate
                    dC_bar = dC_bar * dtanh(cell_bar) 
                    dW_c += np.dot(stack.T,dC_bar)
                    dB_c += np.sum(dC_bar,axis = 0, keepdims = True)
                    
                    # Input gate:
                    di = dC * cell_bar
                    di = dsigmoid(input_gate) * di
                    dW_i += np.dot(stack.T,di)
                    dB_i += np.sum(di,axis = 0,keepdims = True)

                    # Forget gate:
                    df = dC * C_prev
                    df = df * dsigmoid(forget_gate) 
                    dW_f += np.dot(stack.T,df)
                    dB_f += np.sum(df,axis = 0, keepdims = True)

                    dz = np.dot(df,self.W_f.T) + np.dot(di,self.W_i.T) + np.dot(dC_bar,self.W_c.T) + np.dot(do,self.W_o.T)

                    dh_next = dz[:,-self.hidden_dim:]
                    dC_next = forget_gate * dC
                
                # List of gradients :
                grads = [dW,dB,dW_o,dB_o,dW_c,dB_c,dW_i,dB_i,dW_f,dB_f]

                # Clipping gradients anyway
                for grad in grads:
                    np.clip(grad, -15, 15, out = grad)

                return h_s[self.seq_len - 1],C_s[self.seq_len -1 ],grads
            


            def fit(self,X,Y,X_val,y_val,epochs = 50 ,optimizer = 'SGD',verbose = True, crossVal = False):
                """
                Given the traning dataset,their labels and number of epochs
                fitting the model, and measure the performance
                by validating training dataset.
                """
                        
                
                for epoch in range(epochs):
                    
                    print(f'Epoch : {epoch + 1}')

                    perm = np.random.permutation(3000)           
                    h_prev,C_prev = np.zeros((self.batch_size,self.hidden_dim)),np.zeros((self.batch_size,self.hidden_dim))
                    for i in range(round(X.shape[0]/self.batch_size) - 1): 
                       
                        batch_start  =  i * self.batch_size
                        batch_finish = (i+1) * self.batch_size                
                        index = perm[batch_start:batch_finish]
                        
                        # Feeding random indexes:
                        X_feed = X[index]    
                        y_feed = Y[index]
                       
                        # Forward + BPTT + SGD:
                        cache_train = self.forward(X_feed,h_prev,C_prev)
                        h,c,grads = self.BPTT(cache_train,y_feed)

                        if optimizer == 'SGD':                                                                        
                          self.SGD(grads)

                        elif optimizer == 'AdaGrad' :
                          self.AdaGrad(grads)

                        elif optimizer == 'RMSprop':
                          self.RMSprop(grads)
                        
                        elif optimizer == 'VanillaAdam':
                          self.VanillaAdam(grads)
                        else:
                          self.Adam(grads)

                        # Hidden state -------> Previous hidden state
                        # Cell state ---------> Previous cell state
                        h_prev,C_prev = h,c

                    # Training metrics calculations:
                    cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[8][149])
                    predictions_train = self.predict(X)
                    acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)

                    # Validation metrics calculations:
                    test_prevs = np.zeros((X_val.shape[0],self.hidden_dim))
                    _,__,___,____,_____,______,_______,________,probs_test = self.forward(X_val,test_prevs,test_prevs)
                    cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[149])
                    predictions_val = np.argmax(probs_test[149],1)
                    acc_val = metrics.accuracy(np.argmax(y_val,1),predictions_val)

                    if verbose:

                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                        print('______________________________________________________________________________________\n')                         
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                        print('______________________________________________________________________________________\n')
                        
                    self.train_loss.append(cross_loss_train)              
                    self.test_loss.append(cross_loss_val) 
                    self.train_acc.append(acc_train)              
                    self.test_acc.append(acc_val)
              
            
            def params(self):
                """
                Return all weights/biases in sequential order starting from end in list form.

                """        
                return [self.W,self.B,self.W_o,self.B_o,self.W_c,self.B_c,self.W_i,self.B_i,self.W_f,self.B_f]


            def SGD(self,grads):
              """

              Stochastic gradient descent with momentum on mini-batches.
              """
              prevs = []
              for param,grad,prev_update in zip(self.params(),grads,self.previous_updates):            
                  delta = self.learning_rate * grad - self.mom_coeff * prev_update
                  param -= delta 
                  prevs.append(delta)

              self.previous_updates = prevs       

              self.learning_rate *= 0.99999   

            
            def AdaGrad(self,grads):
              """
              AdaGrad adaptive optimization algorithm.
              """         

              i = 0
              for param,grad in zip(self.params(),grads):

                self.cache[i] += grad **2
                param += -self.learning_rate * grad / (np.sqrt(self.cache[i]) + 1e-6)

                i += 1


            def RMSprop(self,grads,decay_rate = 0.9):
              """

              RMSprop adaptive optimization algorithm
              """


              i = 0
              for param,grad in zip(self.params(),grads):
                self.cache_rmsprop[i] = decay_rate * self.cache_rmsprop[i] + (1-decay_rate) * grad **2
                param += - self.learning_rate * grad / (np.sqrt(self.cache_rmsprop[i])+ 1e-6)
                i += 1


            def VanillaAdam(self,grads,beta1 = 0.9,beta2 = 0.999):
                """
                Adam optimizer, but bias correction is not implemented
                """
                i = 0

                for param,grad  in zip(self.params(),grads):

                  self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
                  self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2  
                  param += -self.learning_rate * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)
                  i += 1


            def Adam(self,grads,beta1 = 0.9,beta2 = 0.999):
                """

                Adam optimizer, bias correction is implemented.
                """
              
                i = 0

                for param,grad  in zip(self.params(),grads):
                  
                  self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
                  self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2
                  m_corrected = self.m[i] / (1-beta1**self.t)
                  v_corrected = self.v[i] / (1-beta2**self.t)
                  param += -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)
                  i += 1
                  
                self.t +=1
            
            
            def CategoricalCrossEntropy(self,labels,preds):
                """
                Computes cross entropy between labels and model's predictions
                """
                predictions = np.clip(preds, 1e-12, 1. - 1e-12)
                N = predictions.shape[0]         
                return -np.sum(labels * np.log(predictions + 1e-9)) / N
            
            def predict(self,X):
                """
                Return predictions, (not one hot encoded format)
                """

                # Give zeros to hidden/cell states:
                pasts = np.zeros((X.shape[0],self.hidden_dim))
                _,__,___,____,_____,______,_______,_______,probs = self.forward(X,pasts,pasts)
                return np.argmax(probs[149],axis=1)

            def history(self):
                return {'TrainLoss' : self.train_loss,
                        'TrainAcc'  : self.train_acc,
                        'TestLoss'  : self.test_loss,
                        'TestAcc'   : self.test_acc}      


        # %%
        lstm = LSTM(learning_rate = 5e-4,mom_coeff = 0.0,batch_size = 32,hidden_dim=128)


        # %%
        lstm.fit(X_train,y_train,X_test,y_test,epochs = 15,optimizer='SGD')


        # %%
        lstm_history = lstm.history()


        # %%
        train_preds_lstm = lstm.predict(X_train)
        test_preds_lstm = lstm.predict(X_test)
        confusion_mat_train_lstm = metrics.confusion_matrix(np.argmax(y_train,1),train_preds_lstm)
        confusion_mat_test_lstm = metrics.confusion_matrix(np.argmax(y_test,1),test_preds_lstm)

        body_movements = ['downstairs','jogging','sitting','standing','upstairs','walking']
        confusion_mat_train_lstm.columns = body_movements
        confusion_mat_train_lstm.index = body_movements
        confusion_mat_test_lstm.columns = body_movements
        confusion_mat_test_lstm.index = body_movements

        sns.heatmap(confusion_mat_train_lstm/np.sum(confusion_mat_train_lstm), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()
        sns.heatmap(confusion_mat_test_lstm/np.sum(confusion_mat_test_lstm), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()


        # %%
        lstm2 = LSTM(learning_rate = 2e-3,mom_coeff = 0.0,batch_size = 32,hidden_dim=128)
        lstm2.fit(X_train,y_train,X_test,y_test,epochs = 15,optimizer='RMSprop')


        # %%
        lstm2_history = lstm2.history()


        # %%
        lstm3 = LSTM(learning_rate = 3e-3,mom_coeff = 0.0,batch_size = 32,hidden_dim=128)
        lstm3.fit(X_train,y_train,X_test,y_test,epochs = 15,optimizer='Adam')


        # %%
        lstm4 = LSTM(learning_rate = 1e-3,mom_coeff = 0.0,batch_size = 32,hidden_dim=128)
        lstm4.fit(X_train,y_train,X_test,y_test,epochs = 15,optimizer='AdaGrad')


        # %%
        lstm5 = LSTM(learning_rate = 1e-3,mom_coeff = 0.0,batch_size = 32,hidden_dim=128)
        lstm5.fit(X_train,y_train,X_test,y_test,epochs = 15,optimizer='VanillaAdam')


        # %%
        lstm3_history = lstm3.history()
        lstm4_history = lstm4.history()
        lstm5_history = lstm5.history()
        plt.figure()
        plt.plot(lstm_history['TrainAcc'],'-o')
        plt.plot(lstm2_history['TrainAcc'],'-o')
        plt.plot(lstm3_history['TrainAcc'],'-o')
        plt.plot(lstm4_history['TrainAcc'],'-o')
        plt.plot(lstm5_history['TrainAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Training Accuracy over epochs')
        plt.legend(['SGD','RMSprop','Adam','AdaGrad','Vanilla Adam'])
        plt.show()

        plt.figure()
        plt.plot(lstm_history['TestAcc'],'-o')
        plt.plot(lstm2_history['TestAcc'],'-o')
        plt.plot(lstm3_history['TestAcc'],'-o')
        plt.plot(lstm4_history['TestAcc'],'-o')
        plt.plot(lstm5_history['TestAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Accuracy over epochs')
        plt.legend(['SGD','RMSprop','Adam','AdaGrad','Vanilla Adam'])
        plt.show()

        plt.figure()
        plt.plot(lstm_history['TrainLoss'],'-o')
        plt.plot(lstm2_history['TrainLoss'],'-o')
        plt.plot(lstm3_history['TrainLoss'],'-o')
        plt.plot(lstm4_history['TrainLoss'],'-o')
        plt.plot(lstm5_history['TrainLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over epochs')
        plt.legend(['SGD','RMSprop','Adam','AdaGrad','Vanilla Adam'])
        plt.show()

        plt.figure()
        plt.plot(lstm_history['TestLoss'],'-o')
        plt.plot(lstm2_history['TestLoss'],'-o')
        plt.plot(lstm3_history['TestLoss'],'-o')
        plt.plot(lstm4_history['TestLoss'],'-o')
        plt.plot(lstm5_history['TestLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Loss over epochs')
        plt.legend(['SGD','RMSprop','Adam','AdaGrad','Vanilla Adam'])
        plt.show()


        # %%
        three_layer_rnn_v2_history = three_layer_rnn_v2.history()
        plt.figure()
        plt.plot(three_layer_rnn_v2_history['TrainAcc'],'-o')
        plt.plot(lstm_history['TrainAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Training Accuracy over epochs')
        plt.legend(['Best RNN','Best LSTM'])
        plt.show()


        plt.figure()
        plt.plot(three_layer_rnn_v2_history['TestAcc'],'-o')
        plt.plot(lstm_history['TestAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Accuracy over epochs')
        plt.legend(['Best RNN','Best LSTM'])
        plt.show()

        plt.figure()
        plt.plot(three_layer_rnn_v2_history['TrainLoss'],'-o')
        plt.plot(lstm_history['TrainLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over epochs')
        plt.legend(['Best RNN','Best LSTM'])
        plt.show()


        plt.figure()
        plt.plot(three_layer_rnn_v2_history['TestLoss'],'-o')
        plt.plot(lstm_history['TestLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Loss over epochs')
        plt.legend(['Best RNN','Best LSTM'])
        plt.show()


        # %%
        train_preds_lstm = lstm3.predict(X_train)
        test_preds_lstm = lstm3.predict(X_test)
        confusion_mat_train_lstm = metrics.confusion_matrix(np.argmax(y_train,1),train_preds_lstm)
        confusion_mat_test_lstm = metrics.confusion_matrix(np.argmax(y_test,1),test_preds_lstm)

        body_movements = ['downstairs','jogging','sitting','standing','upstairs','walking']
        confusion_mat_train_lstm.columns = body_movements
        confusion_mat_train_lstm.index = body_movements
        confusion_mat_test_lstm.columns = body_movements
        confusion_mat_test_lstm.index = body_movements

        sns.heatmap(confusion_mat_train_lstm/np.sum(confusion_mat_train_lstm), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()
        sns.heatmap(confusion_mat_test_lstm/np.sum(confusion_mat_test_lstm), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()


        # %%



        # %%
        class Multi_Layer_LSTM(object):
            """

            Long-Short Term Memory Recurrent neural network, encapsulates all necessary logic for training, then built the hyperparameters and architecture of the network.
            """

            def __init__(self,input_dim = 3,hidden_dim_1 = 128,hidden_dim_2 =64,output_class = 6,seq_len = 150,batch_size = 30,learning_rate = 1e-1,mom_coeff = 0.85):
                """

                Initialization of weights/biases and other configurable parameters.
                
                """
                np.random.seed(150)
                self.input_dim = input_dim
                self.hidden_dim_1 = hidden_dim_1
                self.hidden_dim_2 = hidden_dim_2

                # Unfold case T = 150 :
                self.seq_len = seq_len
                self.output_class = output_class
                self.learning_rate = learning_rate
                self.batch_size = batch_size
                self.mom_coeff = mom_coeff

                self.input_stack_dim = self.input_dim + self.hidden_dim_1
                
                # Xavier uniform scaler :
                Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

                lim1 = Xavier(self.input_dim,self.hidden_dim_1)
                self.W_f = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim_1))
                self.B_f = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

                self.W_i = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim_1))
                self.B_i = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

                self.W_c = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim_1))
                self.B_c = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

                self.W_o = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim_1))
                self.B_o = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))
                
                lim2 = Xavier(self.hidden_dim_1,self.hidden_dim_2)
                self.W_hid = np.random.uniform(-lim2,lim2,(self.hidden_dim_1,self.hidden_dim_2))
                self.B_hid = np.random.uniform(-lim2,lim2,(1,self.hidden_dim_2))

                lim3 = Xavier(self.hidden_dim_2,self.output_class)
                self.W = np.random.uniform(-lim3,lim3,(self.hidden_dim_2,self.output_class))
                self.B = np.random.uniform(-lim3,lim3,(1,self.output_class))

                # To keep track loss and accuracy score :     
                self.train_loss,self.test_loss,self.train_acc,self.test_acc = [],[],[],[]
                
                # To keep previous updates in momentum :
                self.previous_updates = [0] * 13
                
                # For AdaGrad:
                self.cache = [0] * 13     
                self.cache_rmsprop = [0] * 13
                self.m = [0] * 13
                self.v = [0] * 13
                self.t = 1

            def cell_forward(self,X,h_prev,C_prev):
                """

                Takes input, previous hidden state and previous cell state, compute:
                --- Forget gate + Input gate + New candidate input + New cell state + 
                    output gate + hidden state. Then, classify by softmax.
                """
                #print(X.shape,h_prev.shape)
                # Stacking previous hidden state vector with inputs:
                stack = np.column_stack([X,h_prev])

                # Forget gate:
                forget_gate = activations.sigmoid(np.dot(stack,self.W_f) + self.B_f)
               
                # İnput gate:
                input_gate = activations.sigmoid(np.dot(stack,self.W_i) + self.B_i)

                # New candidate:
                cell_bar = np.tanh(np.dot(stack,self.W_c) + self.B_c)

                # New Cell state:
                cell_state = forget_gate * C_prev + input_gate * cell_bar

                # Output fate:
                output_gate = activations.sigmoid(np.dot(stack,self.W_o) + self.B_o)

                # Hidden state:
                hidden_state = output_gate * np.tanh(cell_state)

                # Classifiers (Softmax) :
                dense_hid = np.dot(hidden_state,self.W_hid) + self.B_hid
                act = activations.ReLU(dense_hid)

                dense = np.dot(act,self.W) + self.B
                probs = activations.softmax(dense)

                return (stack,forget_gate,input_gate,cell_bar,cell_state,output_gate,hidden_state,dense,probs,dense_hid,act)

                

            def forward(self,X,h_prev,C_prev):
                x_s,z_s,f_s,i_s = {},{},{},{}
                C_bar_s,C_s,o_s,h_s = {},{},{},{}
                v_s,y_s,v_1s,y_1s = {},{},{},{}


                h_s[-1] = np.copy(h_prev)
                C_s[-1] = np.copy(C_prev)

                for t in range(self.seq_len):
                    x_s[t] = X[:,t,:]
                    z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t],v_s[t], y_s[t],v_1s[t],y_1s[t] = self.cell_forward(x_s[t],h_s[t-1],C_s[t-1])

                return (z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s,v_s, y_s,v_1s,y_1s)
            
            def BPTT(self,outs,Y):

                z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s,v_s, y_s,v_1s,y_1s = outs

                dW_f, dW_i,dW_c, dW_o,dW,dW_hid = np.zeros_like(self.W_f), np.zeros_like(self.W_i), np.zeros_like(self.W_c),np.zeros_like(self.W_o),np.zeros_like(self.W),np.zeros_like(self.W_hid)

                dB_f, dB_i,dB_c,dB_o,dB,dB_hid  = np.zeros_like(self.B_f), np.zeros_like(self.B_i),np.zeros_like(self.B_c),np.zeros_like(self.B_o),np.zeros_like(self.B),np.zeros_like(self.B_hid)

                dh_next = np.zeros_like(h_s[0]) 
                dC_next = np.zeros_like(C_s[0])   

                # w.r.t. softmax input
                ddense = np.copy(y_s[149])
                ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1
                #ddense[np.argmax(Y,1)] -=1
                #ddense = y_s[149] - Y
                # Softmax classifier's :
                dW = np.dot(v_1s[149].T,ddense)
                dB = np.sum(ddense,axis = 0, keepdims = True)

                ddense_hid = np.dot(ddense,self.W.T) * activations.dReLU(v_1s[149])
                dW_hid = np.dot(h_s[149].T,ddense_hid)
                dB_hid = np.sum(ddense_hid,axis = 0, keepdims = True)


                # Backprop through time:
                for t in reversed(range(1,self.seq_len)):           
                    
                    # Just equating more meaningful names
                    stack,forget_gate,input_gate,cell_bar,cell_state,output_gate,hidden_state,dense,probs = z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t],v_s[t], y_s[t]
                    C_prev = C_s[t-1]
                    
                    # w.r.t. softmax input
                    #ddense = np.copy(probs)
                    #ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1
                    #ddense[np.arange(len(Y)),np.argmax(Y,1)] -=1
                    # Softmax classifier's :
                    #dW += np.dot(hidden_state.T,ddense)
                    #dB += np.sum(ddense,axis = 0, keepdims = True)

                    # Output gate :
                    dh = np.dot(ddense_hid,self.W_hid.T) + dh_next            
                    do = dh * np.tanh(cell_state)
                    do = do * dsigmoid(output_gate)
                    dW_o += np.dot(stack.T,do)
                    dB_o += np.sum(do,axis = 0, keepdims = True)

                    # Cell state:
                    dC = np.copy(dC_next)
                    dC += dh * output_gate * activations.dtanh(cell_state)
                    dC_bar = dC * input_gate
                    dC_bar = dC_bar * dtanh(cell_bar) 
                    dW_c += np.dot(stack.T,dC_bar)
                    dB_c += np.sum(dC_bar,axis = 0, keepdims = True)
                    
                    # Input gate:
                    di = dC * cell_bar
                    di = dsigmoid(input_gate) * di
                    dW_i += np.dot(stack.T,di)
                    dB_i += np.sum(di,axis = 0,keepdims = True)

                    # Forget gate:
                    df = dC * C_prev
                    df = df * dsigmoid(forget_gate) 
                    dW_f += np.dot(stack.T,df)
                    dB_f += np.sum(df,axis = 0, keepdims = True)

                    dz = np.dot(df,self.W_f.T) + np.dot(di,self.W_i.T) + np.dot(dC_bar,self.W_c.T) + np.dot(do,self.W_o.T)

                    dh_next = dz[:,-self.hidden_dim_1:]
                    dC_next = forget_gate * dC
                
                # List of gradients :
                grads = [dW,dB,dW_hid,dB_hid,dW_o,dB_o,dW_c,dB_c,dW_i,dB_i,dW_f,dB_f]

                # Clipping gradients anyway
                for grad in grads:
                    np.clip(grad, -15, 15, out = grad)

                return h_s[self.seq_len - 1],C_s[self.seq_len -1 ],grads
            


            def fit(self,X,Y,X_val,y_val,epochs = 50 ,optimizer = 'SGD',verbose = True, crossVal = False):
                """
                Given the traning dataset,their labels and number of epochs
                fitting the model, and measure the performance
                by validating training dataset.
                """
                        
                
                for epoch in range(epochs):
                    
                    print(f'Epoch : {epoch + 1}')

                    perm = np.random.permutation(3000)           
                    h_prev,C_prev = np.zeros((self.batch_size,self.hidden_dim_1)),np.zeros((self.batch_size,self.hidden_dim_1))
                    for i in range(round(X.shape[0]/self.batch_size) - 1): 
                       
                        batch_start  =  i * self.batch_size
                        batch_finish = (i+1) * self.batch_size                
                        index = perm[batch_start:batch_finish]
                        
                        # Feeding random indexes:
                        X_feed = X[index]    
                        y_feed = Y[index]
                       
                        # Forward + BPTT + SGD:
                        cache_train = self.forward(X_feed,h_prev,C_prev)
                        h,c,grads = self.BPTT(cache_train,y_feed)

                        if optimizer == 'SGD':                                                           
                          self.SGD(grads)

                        elif optimizer == 'AdaGrad' :
                          self.AdaGrad(grads)

                        elif optimizer == 'RMSprop':
                          self.RMSprop(grads)
                        
                        elif optimizer == 'VanillaAdam':
                          self.VanillaAdam(grads)
                        else:
                          self.Adam(grads)

                        # Hidden state -------> Previous hidden state
                        # Cell state ---------> Previous cell state
                        h_prev,C_prev = h,c

                    # Training metrics calculations:
                    cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[8][149])
                    predictions_train = self.predict(X)
                    acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)

                    # Validation metrics calculations:
                    test_prevs = np.zeros((X_val.shape[0],self.hidden_dim_1))
                    _,__,___,____,_____,______,_______,________,probs_test,a,b = self.forward(X_val,test_prevs,test_prevs)
                    cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[149])
                    predictions_val = np.argmax(probs_test[149],1)
                    acc_val = metrics.accuracy(np.argmax(y_val,1),predictions_val)

                    if verbose:

                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                        print('______________________________________________________________________________________\n')                         
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                        print('______________________________________________________________________________________\n')
                        
                    self.train_loss.append(cross_loss_train)              
                    self.test_loss.append(cross_loss_val) 
                    self.train_acc.append(acc_train)              
                    self.test_acc.append(acc_val)
              
            
            def params(self):
                """
                Return all weights/biases in sequential order starting from end in list form.

                """        
                return [self.W,self.B,self.W_hid,self.B_hid,self.W_o,self.B_o,self.W_c,self.B_c,self.W_i,self.B_i,self.W_f,self.B_f]


            def SGD(self,grads):
              """

              Stochastic gradient descent with momentum on mini-batches.
              """
              prevs = []
             
              for param,grad,prev_update in zip(self.params(),grads,self.previous_updates):            
                  delta = self.learning_rate * grad - self.mom_coeff * prev_update
                  param -= delta 
                  prevs.append(delta)
             
                 

              self.previous_updates = prevs     

              self.learning_rate *= 0.99999   

            
            def AdaGrad(self,grads):
              """
              AdaGrad adaptive optimization algorithm.
              """         
              i = 0
              for param,grad in zip(self.params(),grads):
                self.cache[i] += grad **2
                param += -self.learning_rate * grad / (np.sqrt(self.cache[i]) + 1e-6)
                i += 1


            def RMSprop(self,grads,decay_rate = 0.9):
              """

              RMSprop adaptive optimization algorithm
              """


              i = 0
              for param,grad in zip(self.params(),grads):
                self.cache_rmsprop[i] = decay_rate * self.cache_rmsprop[i] + (1-decay_rate) * grad **2
                param += - self.learning_rate * grad / (np.sqrt(self.cache_rmsprop[i])+ 1e-6)
                i += 1


            def VanillaAdam(self,grads,beta1 = 0.9,beta2 = 0.999):
                """
                Adam optimizer, but bias correction is not implemented
                """
                i = 0

                for param,grad  in zip(self.params(),grads):

                  self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
                  self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2  
                  param += -self.learning_rate * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)
                  i += 1


            def Adam(self,grads,beta1 = 0.9,beta2 = 0.999):
                """

                Adam optimizer, bias correction is implemented.
                """
              
                i = 0

                for param,grad  in zip(self.params(),grads):
                  
                  self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
                  self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2
                  m_corrected = self.m[i] / (1-beta1**self.t)
                  v_corrected = self.v[i] / (1-beta2**self.t)
                  param += -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)
                  i += 1
                  
                self.t +=1
            
            
            def CategoricalCrossEntropy(self,labels,preds):
                """
                Computes cross entropy between labels and model's predictions
                """
                predictions = np.clip(preds, 1e-12, 1. - 1e-12)
                N = predictions.shape[0]         
                return -np.sum(labels * np.log(predictions + 1e-9)) / N
            
            def predict(self,X):
                """
                Return predictions, (not one hot encoded format)
                """

                # Give zeros to hidden/cell states:
                pasts = np.zeros((X.shape[0],self.hidden_dim_1))
                _,__,___,____,_____,______,_______,_______,probs,a,b = self.forward(X,pasts,pasts)
                return np.argmax(probs[149],axis=1)

            def history(self):
                return {'TrainLoss' : self.train_loss,
                        'TrainAcc'  : self.train_acc,
                        'TestLoss'  : self.test_loss,
                        'TestAcc'   : self.test_acc}      


        # %%
        mutl_layer_lstm = Multi_Layer_LSTM(learning_rate=1e-3,batch_size=32,hidden_dim_1 = 128,hidden_dim_2=64,mom_coeff=0.0)
        mutl_layer_lstm.fit(X_train,y_train,X_test,y_test,epochs=15,optimizer='Adam')


        # %%
        mutl_layer_lstm_history = mutl_layer_lstm.history()
        plt.figure()
        plt.plot(mutl_layer_lstm_history['TrainAcc'],'-o')
        plt.plot(lstm_history['TrainAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Training Accuracy over epochs')
        plt.legend(['Multi Layer LSTM','LSTM'])
        plt.show()


        plt.figure()
        plt.plot(mutl_layer_lstm_history['TestAcc'],'-o')
        plt.plot(lstm_history['TestAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Accuracy over epochs')
        plt.legend(['Multi Layer LSTM','LSTM'])
        plt.show()

        plt.figure()
        plt.plot(mutl_layer_lstm_history['TrainLoss'],'-o')
        plt.plot(lstm_history['TrainLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over epochs')
        plt.legend(['Multi Layer LSTM','LSTM'])
        plt.show()


        plt.figure()
        plt.plot(mutl_layer_lstm_history['TestLoss'],'-o')
        plt.plot(lstm_history['TestLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Loss over epochs')
        plt.legend(['Multi Layer LSTM','LSTM'])
        plt.show()


        # %%
        mutl_layer_lstm.fit(X_train,y_train,X_test,y_test,epochs=15,optimizer = 'Vanilla')


        # %%
        mutl_layer_lstm_history = mutl_layer_lstm.history()


        # %%
        plt.figure()
        plt.plot(mutl_layer_lstm_history['TestLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Loss over epochs')
        plt.show()

        plt.figure()
        plt.plot(mutl_layer_lstm_history['TrainLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Loss over epochs')
        plt.show()


        plt.figure()
        plt.plot(mutl_layer_lstm_history['TestAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Loss over epochs')
        plt.show()


        plt.figure()
        plt.plot(mutl_layer_lstm_history['TrainAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Loss over epochs')
        plt.show()


        # %%
        class GRU(object):
            """

            Gater recurrent unit, encapsulates all necessary logic for training, then built the hyperparameters and architecture of the network.
            """

            def __init__(self,input_dim = 3,hidden_dim = 128,output_class = 6,seq_len = 150,batch_size = 32,learning_rate = 1e-1,mom_coeff = 0.85):
                """

                Initialization of weights/biases and other configurable parameters.
                
                """
                np.random.seed(32)
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim

                # Unfold case T = 150 :
                self.seq_len = seq_len
                self.output_class = output_class
                self.learning_rate = learning_rate
                self.batch_size = batch_size
                self.mom_coeff = mom_coeff

               
                
                # Xavier uniform scaler :
                Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

                lim1 = Xavier(self.input_dim,self.hidden_dim)
                lim1_hid = Xavier(self.hidden_dim,self.hidden_dim)
                self.W_z = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim))
                self.U_z = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim,self.hidden_dim))
                self.B_z = np.random.uniform(-lim1,lim1,(1,self.hidden_dim))

                self.W_r = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim))
                self.U_r = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim,self.hidden_dim))
                self.B_r = np.random.uniform(-lim1,lim1,(1,self.hidden_dim))

                self.W_h = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim))
                self.U_h = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim,self.hidden_dim))
                self.B_h = np.random.uniform(-lim1,lim1,(1,self.hidden_dim))

                
                lim2 = Xavier(self.hidden_dim,self.output_class)
                self.W = np.random.uniform(-lim2,lim2,(self.hidden_dim,self.output_class))
                self.B = np.random.uniform(-lim2,lim2,(1,self.output_class))

                # To keep track loss and accuracy score :     
                self.train_loss,self.test_loss,self.train_acc,self.test_acc = [],[],[],[]
                
                # To keep previous updates in momentum :
                self.previous_updates = [0] * 10
                
                # For AdaGrad:
                self.cache = [0] * 11   
                self.cache_rmsprop = [0] * 11
                self.m = [0] * 11
                self.v = [0] * 11
                self.t = 1

            def cell_forward(self,X,h_prev):
                """

                Takes input, previous hidden state and previous cell state, compute:
                --- Forget gate + Input gate + New candidate input + New cell state + 
                    output gate + hidden state. Then, classify by softmax.
                """
                              

                # Update gate:
                update_gate = activations.sigmoid(np.dot(X,self.W_z) + np.dot(h_prev,self.U_z) + self.B_z)
               
                # Reset gate:
                reset_gate = activations.sigmoid(np.dot(X,self.W_r) + np.dot(h_prev,self.U_r) + self.B_r)

                # Current memory content:
                h_hat = np.tanh(np.dot(X,self.W_h) + np.dot(np.multiply(reset_gate,h_prev),self.U_h) + self.B_h)

                # Hidden state:
                hidden_state = np.multiply(update_gate,h_prev) + np.multiply((1-update_gate),h_hat)


                # Classifiers (Softmax) :
                dense = np.dot(hidden_state,self.W) + self.B
                probs = activations.softmax(dense)

                return (update_gate,reset_gate,h_hat,hidden_state,dense,probs)

                

            def forward(self,X,h_prev):
                x_s,z_s,r_s,h_hat = {},{},{},{}
                h_s = {}
                y_s,p_s = {},{}        

                h_s[-1] = np.copy(h_prev)
                

                for t in range(self.seq_len):
                    x_s[t] = X[:,t,:]
                    z_s[t], r_s[t], h_hat[t], h_s[t], y_s[t], p_s[t] = self.cell_forward(x_s[t],h_s[t-1])

                return (x_s,z_s, r_s, h_hat, h_s, y_s, p_s)
            
            def BPTT(self,outs,Y):

                x_s,z_s, r_s, h_hat, h_s, y_s, p_s = outs

                dW_z, dW_r,dW_h, dW = np.zeros_like(self.W_z), np.zeros_like(self.W_r), np.zeros_like(self.W_h),np.zeros_like(self.W)

                dU_z, dU_r,dU_h, = np.zeros_like(self.U_z), np.zeros_like(self.U_r), np.zeros_like(self.U_h)


                dB_z, dB_r,dB_h,dB = np.zeros_like(self.B_z), np.zeros_like(self.B_r),np.zeros_like(self.B_h),np.zeros_like(self.B)

                dh_next = np.zeros_like(h_s[0]) 
                   

                # w.r.t. softmax input
                ddense = np.copy(p_s[149])
                ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1
                #ddense[np.argmax(Y,1)] -=1
                #ddense = y_s[149] - Y
                # Softmax classifier's :
                dW = np.dot(h_s[149].T,ddense)
                dB = np.sum(ddense,axis = 0, keepdims = True)

                # Backprop through time:
                for t in reversed(range(1,self.seq_len)):           
                                
                    # w.r.t. softmax input
                    #ddense = np.copy(probs)
                    #ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1
                    #ddense[np.arange(len(Y)),np.argmax(Y,1)] -=1
                    # Softmax classifier's :
                    #dW += np.dot(hidden_state.T,ddense)
                    #dB += np.sum(ddense,axis = 0, keepdims = True)


                    # Curernt memort state :
                    dh = np.dot(ddense,self.W.T) + dh_next            
                    dh_hat = dh * (1-z_s[t])
                    dh_hat = dh_hat * dtanh(h_hat[t])
                    dW_h += np.dot(x_s[t].T,dh_hat)
                    dU_h += np.dot((r_s[t] * h_s[t-1]).T,dh_hat)
                    dB_h += np.sum(dh_hat,axis = 0, keepdims = True)

                    # Reset gate:
                    dr_1 = np.dot(dh_hat,self.U_h.T)
                    dr = dr_1  * h_s[t-1]
                    dr = dr * dsigmoid(r_s[t])
                    dW_r += np.dot(x_s[t].T,dr)
                    dU_r += np.dot(h_s[t-1].T,dr)
                    dB_r += np.sum(dr,axis = 0, keepdims = True)

                    # Forget gate:
                    dz = dh * (h_s[t-1] - h_hat[t])
                    dz = dz * dsigmoid(z_s[t])
                    dW_z += np.dot(x_s[t].T,dz)
                    dU_z += np.dot(h_s[t-1].T,dz)
                    dB_z += np.sum(dz,axis = 0, keepdims = True)


                    # Nexts:
                    dh_next = np.dot(dz,self.U_z.T) + (dh * z_s[t]) + (dr_1 * r_s[t]) + np.dot(dr,self.U_r.T)


                # List of gradients :
                grads = [dW,dB,dW_z,dU_z,dB_z,dW_r,dU_r,dB_r,dW_h,dU_h,dB_h]

                # Clipping gradients anyway
                for grad in grads:
                    np.clip(grad, -15, 15, out = grad)

                return h_s[self.seq_len - 1],grads
            


            def fit(self,X,Y,X_val,y_val,epochs = 50 ,optimizer = 'SGD',verbose = True, crossVal = False):
                """
                Given the traning dataset,their labels and number of epochs
                fitting the model, and measure the performance
                by validating training dataset.
                """
                        
                
                for epoch in range(epochs):
                    
                    print(f'Epoch : {epoch + 1}')

                    perm = np.random.permutation(3000)           
                    h_prev = np.zeros((self.batch_size,self.hidden_dim))
                    for i in range(round(X.shape[0]/self.batch_size) - 1): 
                       
                        batch_start  =  i * self.batch_size
                        batch_finish = (i+1) * self.batch_size                
                        index = perm[batch_start:batch_finish]
                        
                        # Feeding random indexes:
                        X_feed = X[index]    
                        y_feed = Y[index]
                       
                        # Forward + BPTT + SGD:
                        cache_train = self.forward(X_feed,h_prev)
                        h,grads = self.BPTT(cache_train,y_feed)

                        if optimizer == 'SGD':                                                                
                          self.SGD(grads)

                        elif optimizer == 'AdaGrad' :
                          self.AdaGrad(grads)

                        elif optimizer == 'RMSprop':
                          self.RMSprop(grads)
                        
                        elif optimizer == 'VanillaAdam':
                          self.VanillaAdam(grads)
                        else:
                          self.Adam(grads)

                        # Hidden state -------> Previous hidden state
                        h_prev= h

                    # Training metrics calculations:
                    cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[6][149])
                    predictions_train = self.predict(X)
                    acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)

                    # Validation metrics calculations:
                    test_prevs = np.zeros((X_val.shape[0],self.hidden_dim))
                    _,__,___,____,_____,______,probs_test = self.forward(X_val,test_prevs)
                    cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[149])
                    predictions_val = np.argmax(probs_test[149],1)
                    acc_val = metrics.accuracy(np.argmax(y_val,1),predictions_val)

                    if verbose:

                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                        print('______________________________________________________________________________________\n')                         
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                        print('______________________________________________________________________________________\n')
                        
                    self.train_loss.append(cross_loss_train)              
                    self.test_loss.append(cross_loss_val) 
                    self.train_acc.append(acc_train)              
                    self.test_acc.append(acc_val)
              
            
            def params(self):
                """
                Return all weights/biases in sequential order starting from end in list form.

                """        
                return [self.W,self.B,self.W_z,self.U_z,self.B_z,self.W_r,self.U_r,self.B_r,self.W_h,self.U_h,self.B_h]

            def SGD(self,grads):
              """

              Stochastic gradient descent with momentum on mini-batches.
              """
              prevs = []
              for param,grad,prev_update in zip(self.params(),grads,self.previous_updates):            
                  delta = self.learning_rate * grad - self.mom_coeff * prev_update
                  param -= delta 
                  prevs.append(delta)

              self.previous_updates = prevs       

              self.learning_rate *= 0.99999   

            
            def AdaGrad(self,grads):
              """
              AdaGrad adaptive optimization algorithm.
              """         

              i = 0
              for param,grad in zip(self.params(),grads):

                self.cache[i] += grad **2
                param += -self.learning_rate * grad / (np.sqrt(self.cache[i]) + 1e-6)

                i += 1


            def RMSprop(self,grads,decay_rate = 0.9):
              """

              RMSprop adaptive optimization algorithm
              """


              i = 0
              for param,grad in zip(self.params(),grads):
                self.cache_rmsprop[i] = decay_rate * self.cache_rmsprop[i] + (1-decay_rate) * grad **2
                param += - self.learning_rate * grad / (np.sqrt(self.cache_rmsprop[i])+ 1e-6)
                i += 1


            def VanillaAdam(self,grads,beta1 = 0.9,beta2 = 0.999):
                """
                Adam optimizer, but bias correction is not implemented
                """
                i = 0

                for param,grad  in zip(self.params(),grads):

                  self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
                  self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2  
                  param += -self.learning_rate * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)
                  i += 1


            def Adam(self,grads,beta1 = 0.9,beta2 = 0.999):
                """

                Adam optimizer, bias correction is implemented.
                """
              
                i = 0

                for param,grad  in zip(self.params(),grads):
                  
                  self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
                  self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2
                  m_corrected = self.m[i] / (1-beta1**self.t)
                  v_corrected = self.v[i] / (1-beta2**self.t)
                  param += -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)
                  i += 1
                  
                self.t +=1
            
            
            def CategoricalCrossEntropy(self,labels,preds):
                """
                Computes cross entropy between labels and model's predictions
                """
                predictions = np.clip(preds, 1e-12, 1. - 1e-12)
                N = predictions.shape[0]         
                return -np.sum(labels * np.log(predictions + 1e-9)) / N
            
            def predict(self,X):
                """
                Return predictions, (not one hot encoded format)
                """

                # Give zeros to hidden/cell states:
                pasts = np.zeros((X.shape[0],self.hidden_dim))
                _,__,___,____,_____,______,probs = self.forward(X,pasts)
                return np.argmax(probs[149],axis=1)

            def history(self):
                return {'TrainLoss' : self.train_loss,
                        'TrainAcc'  : self.train_acc,
                        'TestLoss'  : self.test_loss,
                        'TestAcc'   : self.test_acc}      


        # %%
        gru = GRU(hidden_dim=128,learning_rate=1e-3,batch_size=32,mom_coeff=0.0)


        # %%
        gru.fit(X_train,y_train,X_test,y_test,epochs = 15,optimizer = 'RMSprop')


        # %%
        gru_history = gru.history()


        # %%
        # For figure 97:

        plt.figure()
        plt.plot(gru_history['TrainLoss'],'-o')
        plt.plot(gru_history['TestLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.legend(['Train Loss','Test Loss'])
        plt.show()


        plt.figure()
        plt.plot(gru_history['TrainAcc'],'-o')
        plt.plot(gru_history['TestAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Accuracy over epochs')
        plt.legend(['Train Acc','Test Acc'])
        plt.show()


        # %%
        # For figure 98:
        multi_layer_gru_history = multi_layer_gru.history()
        plt.figure()
        plt.plot(multi_layer_gru_history['TrainAcc'],'-o')
        plt.plot(gru_history['TrainAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Training Accuracy over epochs')
        plt.legend(['Multi Layer GRU','GRU'])
        plt.show()


        plt.figure()
        plt.plot(multi_layer_gru_history['TestAcc'],'-o')
        plt.plot(gru_history['TestAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Accuracy over epochs')
        plt.legend(['Multi Layer GRU','GRU'])
        plt.show()

        plt.figure()
        plt.plot(multi_layer_gru_history['TrainLoss'],'-o')
        plt.plot(gru_history['TrainLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over epochs')
        plt.legend(['Multi Layer GRU','GRU'])
        plt.show()


        plt.figure()
        plt.plot(multi_layer_gru_history['TestLoss'],'-o')
        plt.plot(gru_history['TestLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Loss over epochs')
        plt.legend(['Multi Layer GRU','GRU'])
        plt.show()


        # %%
        # For figure 99:
        three_layer_rnn_history = three_layer_rnn.history()
        plt.figure()
        plt.plot(gru_history['TrainAcc'],'-o')
        plt.plot(lstm_history['TrainAcc'],'-o')
        plt.plot(three_layer_rnn_history['TrainAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Training Accuracy over epochs')
        plt.legend(['GRU','LSTM','RNN'])
        plt.show()


        plt.figure()
        plt.plot(gru_history['TestAcc'],'-o')
        plt.plot(lstm_history['TestAcc'],'-o')
        plt.plot(three_layer_rnn_history['TestAcc'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Accuracy over epochs')
        plt.legend(['GRU','LSTM','RNN'])
        plt.show()

        plt.figure()
        plt.plot(gru_history['TrainLoss'],'-o')
        plt.plot(lstm_history['TrainLoss'],'-o')
        plt.plot(three_layer_rnn_history['TrainLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over epochs')
        plt.legend(['GRU','LSTM','RNN'])
        plt.show()


        plt.figure()
        plt.plot(gru_history['TestLoss'],'-o')
        plt.plot(lstm_history['TestLoss'],'-o')
        plt.plot(three_layer_rnn_history['TestLoss'],'-o')
        plt.xlabel('# of epochs')
        plt.ylabel('Loss')
        plt.title('Testing Loss over epochs')
        plt.legend(['GRU','LSTM','RNN'])
        plt.show()


        # %%
        train_preds_gru = gru.predict(X_train)
        test_preds_gru = gru.predict(X_test)
        confusion_mat_train_gru = metrics.confusion_matrix(np.argmax(y_train,1),train_preds_gru)
        confusion_mat_test_gru = metrics.confusion_matrix(np.argmax(y_test,1),test_preds_gru)

        body_movements = ['downstairs','jogging','sitting','standing','upstairs','walking']
        confusion_mat_train_gru.columns = body_movements
        confusion_mat_train_gru.index = body_movements
        confusion_mat_test_gru.columns = body_movements
        confusion_mat_test_gru.index = body_movements

        sns.heatmap(confusion_mat_train_gru/np.sum(confusion_mat_train_gru), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()
        sns.heatmap(confusion_mat_test_gru/np.sum(confusion_mat_test_gru), annot=True, 
                    fmt='.2%',cmap = 'Blues')
        plt.show()


        # %%
        class Multi_layer_GRU(object):
            """

            Gater recurrent unit, encapsulates all necessary logic for training, then built the hyperparameters and architecture of the network.
            """

            def __init__(self,input_dim = 3,hidden_dim_1 = 128,hidden_dim_2 = 64,output_class = 6,seq_len = 150,batch_size = 32,learning_rate = 1e-1,mom_coeff = 0.85):
                """

                Initialization of weights/biases and other configurable parameters.
                
                """
                np.random.seed(150)
                self.input_dim = input_dim
                self.hidden_dim_1 = hidden_dim_1
                self.hidden_dim_2 = hidden_dim_2

                # Unfold case T = 150 :
                self.seq_len = seq_len
                self.output_class = output_class
                self.learning_rate = learning_rate
                self.batch_size = batch_size
                self.mom_coeff = mom_coeff

               
                
                # Xavier uniform scaler :
                Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

                lim1 = Xavier(self.input_dim,self.hidden_dim_1)
                lim1_hid = Xavier(self.hidden_dim_1,self.hidden_dim_1)
                self.W_z = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim_1))
                self.U_z = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim_1,self.hidden_dim_1))
                self.B_z = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

                self.W_r = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim_1))
                self.U_r = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim_1,self.hidden_dim_1))
                self.B_r = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

                self.W_h = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim_1))
                self.U_h = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim_1,self.hidden_dim_1))
                self.B_h = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

                lim2_hid = Xavier(self.hidden_dim_1,self.hidden_dim_2)
                self.W_hid = np.random.uniform(-lim2_hid,lim2_hid,(self.hidden_dim_1,self.hidden_dim_2))
                self.B_hid = np.random.uniform(-lim2_hid,lim2_hid,(1,self.hidden_dim_2))
                
                lim2 = Xavier(self.hidden_dim_2,self.output_class)
                self.W = np.random.uniform(-lim2,lim2,(self.hidden_dim_2,self.output_class))
                self.B = np.random.uniform(-lim2,lim2,(1,self.output_class))

                # To keep track loss and accuracy score :     
                self.train_loss,self.test_loss,self.train_acc,self.test_acc = [],[],[],[]
                
                # To keep previous updates in momentum :
                self.previous_updates = [0] * 13
                
                # For AdaGrad:
                self.cache = [0] * 13  
                self.cache_rmsprop = [0] * 13
                self.m = [0] * 13
                self.v = [0] * 13
                self.t = 1

            def cell_forward(self,X,h_prev):
                

                # Update gate:
                update_gate = activations.sigmoid(np.dot(X,self.W_z) + np.dot(h_prev,self.U_z) + self.B_z)
               
                # Reset gate:
                reset_gate = activations.sigmoid(np.dot(X,self.W_r) + np.dot(h_prev,self.U_r) + self.B_r)

                # Current memory content:
                h_hat = np.tanh(np.dot(X,self.W_h) + np.dot(np.multiply(reset_gate,h_prev),self.U_h) + self.B_h)

                # Hidden state:
                hidden_state = np.multiply(update_gate,h_prev) + np.multiply((1-update_gate),h_hat)

                # Hidden MLP:
                hid_dense = np.dot(hidden_state,self.W_hid) + self.B_hid
                relu = activations.ReLU(hid_dense)

                # Classifiers (Softmax) :
                dense = np.dot(relu,self.W) + self.B
                probs = activations.softmax(dense)

                return (update_gate,reset_gate,h_hat,hidden_state,hid_dense,relu,dense,probs)

                

            def forward(self,X,h_prev):
                x_s,z_s,r_s,h_hat = {},{},{},{}
                h_s = {}
                hd_s,relu_s = {},{}
                y_s,p_s = {},{}        

                h_s[-1] = np.copy(h_prev)
                

                for t in range(self.seq_len):
                    x_s[t] = X[:,t,:]
                    z_s[t], r_s[t], h_hat[t], h_s[t],hd_s[t],relu_s[t], y_s[t], p_s[t] = self.cell_forward(x_s[t],h_s[t-1])

                return (x_s,z_s, r_s, h_hat, h_s, hd_s,relu_s, y_s, p_s)
            
            def BPTT(self,outs,Y):

                x_s,z_s, r_s, h_hat, h_s, hd_s,relu_s, y_s, p_s = outs

                dW_z, dW_r,dW_h, dW = np.zeros_like(self.W_z), np.zeros_like(self.W_r), np.zeros_like(self.W_h),np.zeros_like(self.W)
                dW_hid = np.zeros_like(self.W_hid)
                dU_z, dU_r,dU_h = np.zeros_like(self.U_z), np.zeros_like(self.U_r), np.zeros_like(self.U_h)


                dB_z, dB_r,dB_h,dB = np.zeros_like(self.B_z), np.zeros_like(self.B_r),np.zeros_like(self.B_h),np.zeros_like(self.B)
                dB_hid = np.zeros_like(self.B_hid)
                dh_next = np.zeros_like(h_s[0]) 
                   

                # w.r.t. softmax input
                ddense = np.copy(p_s[149])
                ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1
                #ddense[np.argmax(Y,1)] -=1
                #ddense = y_s[149] - Y
                # Softmax classifier's :
                dW = np.dot(relu_s[149].T,ddense)
                dB = np.sum(ddense,axis = 0, keepdims = True)

                ddense_hid = np.dot(ddense,self.W.T) * activations.dReLU(hd_s[149])
                dW_hid = np.dot(h_s[149].T,ddense_hid)
                dB_hid = np.sum(ddense_hid,axis = 0, keepdims = True)

           
                # Backprop through time:
                for t in reversed(range(1,self.seq_len)):           

                    # Curernt memort state :
                    dh = np.dot(ddense_hid,self.W_hid.T) + dh_next            
                    dh_hat = dh * (1-z_s[t])
                    dh_hat = dh_hat * dtanh(h_hat[t])
                    dW_h += np.dot(x_s[t].T,dh_hat)
                    dU_h += np.dot((r_s[t] * h_s[t-1]).T,dh_hat)
                    dB_h += np.sum(dh_hat,axis = 0, keepdims = True)

                    # Reset gate:
                    dr_1 = np.dot(dh_hat,self.U_h.T)
                    dr = dr_1  * h_s[t-1]
                    dr = dr * dsigmoid(r_s[t])
                    dW_r += np.dot(x_s[t].T,dr)
                    dU_r += np.dot(h_s[t-1].T,dr)
                    dB_r += np.sum(dr,axis = 0, keepdims = True)

                    # Forget gate:
                    dz = dh * (h_s[t-1] - h_hat[t])
                    dz = dz * dsigmoid(z_s[t])
                    dW_z += np.dot(x_s[t].T,dz)
                    dU_z += np.dot(h_s[t-1].T,dz)
                    dB_z += np.sum(dz,axis = 0, keepdims = True)


                    # Nexts:
                    dh_next = np.dot(dz,self.U_z.T) + (dh * z_s[t]) + (dr_1 * r_s[t]) + np.dot(dr,self.U_r.T)


                # List of gradients :
                grads = [dW,dB,dW_hid,dB_hid,dW_z,dU_z,dB_z,dW_r,dU_r,dB_r,dW_h,dU_h,dB_h]
                      
                # Clipping gradients anyway
                for grad in grads:
                    np.clip(grad, -15, 15, out = grad)

                return h_s[self.seq_len - 1],grads
            


            def fit(self,X,Y,X_val,y_val,epochs = 50 ,optimizer = 'SGD',verbose = True, crossVal = False):
                """
                Given the traning dataset,their labels and number of epochs
                fitting the model, and measure the performance
                by validating training dataset.
                """
                        
                
                for epoch in range(epochs):
                    
                    print(f'Epoch : {epoch + 1}')

                    perm = np.random.permutation(3000)

                    # Equate 0 in every epoch:           
                    h_prev = np.zeros((self.batch_size,self.hidden_dim_1))

                    for i in range(round(X.shape[0]/self.batch_size) - 1): 
                       
                        batch_start  =  i * self.batch_size
                        batch_finish = (i+1) * self.batch_size                
                        index = perm[batch_start:batch_finish]
                        
                        # Feeding random indexes:
                        X_feed = X[index]    
                        y_feed = Y[index]
                       
                        # Forward + BPTT + Optimization:
                        cache_train = self.forward(X_feed,h_prev)
                        h,grads = self.BPTT(cache_train,y_feed)

                        if optimizer == 'SGD':                                                                
                          self.SGD(grads)

                        elif optimizer == 'AdaGrad' :
                          self.AdaGrad(grads)

                        elif optimizer == 'RMSprop':
                          self.RMSprop(grads)
                        
                        elif optimizer == 'VanillaAdam':
                          self.VanillaAdam(grads)
                        else:
                          self.Adam(grads)

                        # Hidden state -------> Previous hidden state
                        h_prev = h

                    # Training metrics calculations:
                    cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[8][149])
                    predictions_train = self.predict(X)
                    acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)

                    # Validation metrics calculations:
                    test_prevs = np.zeros((X_val.shape[0],self.hidden_dim_1))
                    _,__,___,____,_____,______,_______,________,probs_test = self.forward(X_val,test_prevs)
                    cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[149])
                    predictions_val = np.argmax(probs_test[149],1)
                    acc_val = metrics.accuracy(np.argmax(y_val,1),predictions_val)

                    if verbose:

                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                        print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                        print('______________________________________________________________________________________\n')                         
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                        print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                        print('______________________________________________________________________________________\n')
                        
                    self.train_loss.append(cross_loss_train)              
                    self.test_loss.append(cross_loss_val) 
                    self.train_acc.append(acc_train)              
                    self.test_acc.append(acc_val)
              
            
            def params(self):
                """
                Return all weights/biases in sequential order starting from end in list form.

                """        
                return [self.W,self.B,self.W_hid,self.B_hid,self.W_z,self.U_z,self.B_z,self.W_r,self.U_r,self.B_r,self.W_h,self.U_h,self.B_h]

            def SGD(self,grads):
              """

              Stochastic gradient descent with momentum on mini-batches.
              """
              prevs = []
              
              for param,grad,prev_update in zip(self.params(),grads,self.previous_updates): 
                             
                  delta = self.learning_rate * grad + self.mom_coeff * prev_update
                  param -= delta 
                  prevs.append(delta)
                

              self.previous_updates = prevs     
              self.learning_rate *= 0.99999   

            
            def AdaGrad(self,grads):
              """
              AdaGrad adaptive optimization algorithm.
              """      

              i = 0
              for param,grad in zip(self.params(),grads):

                self.cache[i] += grad **2
                param += -self.learning_rate * grad / (np.sqrt(self.cache[i]) + 1e-6)

                i += 1


            def RMSprop(self,grads,decay_rate = 0.9):
              """
              RMSprop adaptive optimization algorithm
              """
              i = 0
              for param,grad in zip(self.params(),grads):
                self.cache_rmsprop[i] = decay_rate * self.cache_rmsprop[i] + (1-decay_rate) * grad **2
                param += - self.learning_rate * grad / (np.sqrt(self.cache_rmsprop[i])+ 1e-6)
                i += 1


            def VanillaAdam(self,grads,beta1 = 0.9,beta2 = 0.999):
                """
                Adam optimizer, but bias correction is not implemented
                """
                i = 0

                for param,grad  in zip(self.params(),grads):

                  self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
                  self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2  
                  param += -self.learning_rate * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)
                  i += 1


            def Adam(self,grads,beta1 = 0.9,beta2 = 0.999):
                """

                Adam optimizer, bias correction is implemented.
                """
              
                i = 0

                for param,grad  in zip(self.params(),grads):
                  
                  self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
                  self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2
                  m_corrected = self.m[i] / (1-beta1**self.t)
                  v_corrected = self.v[i] / (1-beta2**self.t)
                  param += -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)
                  i += 1
                  
                self.t +=1
            
            
            def CategoricalCrossEntropy(self,labels,preds):
                """
                Computes cross entropy between labels and model's predictions
                """
                predictions = np.clip(preds, 1e-12, 1. - 1e-12)
                N = predictions.shape[0]         
                return -np.sum(labels * np.log(predictions + 1e-9)) / N
            
            def predict(self,X):
                """
                Return predictions, (not one hot encoded format)
                """

                # Give zeros to hidden states:
                pasts = np.zeros((X.shape[0],self.hidden_dim_1))
                _,__,___,____,_____,______,_______,________,probs = self.forward(X,pasts)
                return np.argmax(probs[149],axis=1)

            def history(self):
                return {'TrainLoss' : self.train_loss,
                        'TrainAcc'  : self.train_acc,
                        'TestLoss'  : self.test_loss,
                        'TestAcc'   : self.test_acc}      


        # %%
        multi_layer_gru = Multi_layer_GRU(hidden_dim_1=128,hidden_dim_2=64,learning_rate=1e-3,mom_coeff=0.0,batch_size=32)


        # %%
        multi_layer_gru.fit(X_train,y_train,X_test,y_test,epochs = 15,optimizer = 'RMSprop')


can_kocagil_21602218_hw3(question)

