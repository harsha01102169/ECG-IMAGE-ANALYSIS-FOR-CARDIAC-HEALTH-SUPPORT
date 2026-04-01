This is a major project . 
Worked on 3 datasets
dataset 1 consists of 2 classes of normal ecg images and images havinbg the history of MI
LINK : https://drive.google.com/drive/folders/16xHagKiC3moIW-ZN1LBAotYEUP9OFjT1?usp=sharing
dataset 2 consists of 2 classes of normal ecg images and the images with abnormal heart beats
LINK : https://drive.google.com/drive/folders/1d00onbtdEOsYpBtQnsdnp8xMrGRLfUL7?usp=sharing
dataset 3 consists of 3 classes of normal ecg images, images havinbg the history of MI, and the images with abnormal heart beats
LINK : https://drive.google.com/drive/folders/1WlH7Xy2ziUOihpjf2wTTb6ZgmRmVl8NI?usp=sharing
We have implemented 3 models and compared . the models are resnet+attention model, resnet 50 model and also densenet201 model

==> RESNET+ATTENTION MODEL GAVE THE GOOD EVALUSTION METRICS WITH GOOD ACCURACY. WE HAVE TAKEN THOSE MODEL WEIGHTS AND ALSO MODEL ARCHITECTURE, 
AND IMPLEMENTED A HUGGING FACE PLATFORM ON A LOCAL SERVER USING GRADIO. WHERE THIS PLATFORM HELPS TO UPLOAD AN IMGE FROM THE DATASET AND 
IT PREDICTS THAT PARTICULAR IMAGE IS OF FROM WHICH THAT THAT IS WHETHER IT IS OF image havinbg the history of MI, image with abnormal heart beats AND
normal ecg image.
