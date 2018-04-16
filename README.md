# ConvolutionalNeuralNets-FacialExpressionRecognition

Repository files consists of different ConvolutionalNeuralNets models used to train the system for recognizing expressions on the face dynamically. <br/>
DataSet is taken from Kaggle Challenges:<br />
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

new_model.py:<br />
            Its a new CNN model was created for training the system.The layers used are:<br />
            INPUT- >[CONV64+RELU]- >MAX-POOL- >[CONV128+RELU]- >MAX-POOL- >FC1+RELU- >FC2+RELU- >Softmax Regression- >classification<br />
            
CNN_Layers.py: <br />
              This file consists of CNN Layers API's from tensorflow library.Appropriate layers with required hyperparameters and parameters were defined.<br/>
              new_model.py uses layers from this file.<br/>
resnets_model.py: <br/>
                 Residual Blocks are used as basic building blocks for creating the CNN model.<br />
                 Tflearn API's are used for building the layers and residual blocks.Appropriate hyperparameters and parameters were used.<br />
                 
save_model_android.py: <br />
                      We freeze the graph model we created so that it can be deployed in android in production.<br />
                      Graph ,nodes and parameters(Weights and biases) are freezed and stored in a .pb file extenstion ( Protocol Buffer) <br />
