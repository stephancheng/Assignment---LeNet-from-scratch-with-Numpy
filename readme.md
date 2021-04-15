# Image classification with Lenet from scratch

## Introduction
This assignment will build **2 versions of Lenet** to classify **mini ImageNet data set** that have 50 classes. For the Lenet, **no deep learning packages** is used in this assignment, only packages of numpy, pandas, math and time are used.

### Data preprocessing
As the entire training data set is bigger than 10 GB, using the entire set during training plus the trainable parameters and propagations will lead to the computer running out of memory. Therefore, I have split the training set into 126 minibatch of 500 samples each before all of the works, with the use of *"Preread_img.py"*:

You will need txt files with first column as link to the image and second as its class value.
The txt files are named *train.txt*, *val.txt*, *test.txt*.
The programe will 1. shuffle train.txt and make it array of 500 minibatches; 2. read the images according to the shuffled array, resize and save the RBG pixel values and class values in H5 format. The files are saved in folder *"data"*, named *"0.h5,1.h5,2.h5...256.h5"* 3. read the validation and testing set and save in *data.h5*

### Parameter calculation
Use *"parameter_calculation_model1.py"* and *"parameter_calculation_models.py"* to calculate the parameters for the models.
It will show how diffent hyper-parameters (i.e. filter size, stride, zero pad size etc. ) affect the output size and number of parameters.

### Models
In *Lenet5.py*, the class *LeNet5*/ *LeNet5_update* have built the layers according to the parameter calculated above.
Model 1: 7 layers basic model
![Model 1](/pics/model1_archi.png)
Model2: 9 layers modified model
![Model 2](/pics/model2_archi.png)

For ease of comparison, following changes are applied to both models:
1.	**Average pooling** instead of max pooling is applied in the subsampling layers
2.	**SoftMax and Cross entropy loss** is used instead of RBF layer and mean square error
3.	**Special mapping** from S1 to C2 is **not applied**, every S1 feature map is connected to C2 maps
4.	**Momentum optimizer** of 0.95 is applied
5.	**Learning rate** is adjusted though validation tests
6.	Weights are initialized by **Xavier**
7.  **10 epochs** of training are run

The modified Lenet mainly implement the following changes:
1.	Convolutional layers use filters of 3 X 3 X channels
2.	One more convolutional layer and pooling layer is added, and the stride of some of the layers can be smaller
3.	Activation function of Exponential Linear Unit (ELU) is used

#### training and testing
Lenet5.py is created for the modified version:
1. read and normalize the validation and training set
2. Create the models
3. Start each epoch:
    1. Set learning rates
    2. read one batch of data:
        1. Forward propagation and return loss
        2. back propagation and update weights
    3. Evaluate accuracies in term of training dataset and validation dataset
    4. After all batches are trained, print progress
    5. Save feature map of a selected image at epoch 1, 6  
4. Evaluate accuracies in term of testing dataset
5. Create accuracy curve for training and testing
6. Create and save the feature map of a selected image of the trained model

For different model architecture(e.g. model1), build a diffent model, update in line 281 number of layer and 282 the model class crated.

#### Result
The rusult of the 2 models is available in *"Lenet_model1.ipynb"* and *"Lenet_model2.ipynb"*
Model	Top 1 accuracy (%)

Model	|Train accuracy	| Vali-dation accuracy | Test accuracy |	Training time per epoch
----- | ------------- | -------------------- | ------------- | ------------------------
1	| 20.57%	| 16.88% |	17.33%	| 54.7 mins
2	| 23.09%	| 20.66%	| 22.67%	| 25.9 mins

## Technologies
The code is tested on Python 3.8

## Sources
My codes take reference from below:
* LeNet-from-Scratch. GitHub. https://github.com/mattwang44/LeNet-from-Scratch
