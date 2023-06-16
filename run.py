
from networks import *
from sklearn.metrics import confusion_matrix
import joblib
import os
from sklearn.decomposition import PCA
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import pandas as pd
import spectral




def AA(confusion_matrix):
  counter = confusion_matrix.shape[0]
  list_diag = np.diag(confusion_matrix)
  list_raw_sum = np.sum(confusion_matrix, axis=1)
  each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
  average_acc = np.mean(each_acc)
  return each_acc, average_acc


def reports (X_test,y_test,name,model):

  y_pred = model.predict(X_test)
  y_pred = y_pred.reshape(y_pred.shape[0],y_pred.shape[-1])
  y_test = y_test.reshape(y_test.shape[0],y_test.shape[-1])

  if name == 'Indian_Pines':
      target_names = ['Alfalfa', 'Corn Notill', 'Corn Mintill', 'Corn'
                      ,'Grass Pasture', 'Grass Trees', 'Grass Pasture Mowed', 
                      'Hay Windrowed', 'Oats', 'Soybean Notill', 'Soybean Mintill',
                      'Soybean Clean', 'Wheat', 'Woods', 'Buildings Grass Trees Drives',
                      'Stone Steel Towers']

  elif name == 'Pavia_University':
      target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                      'Self-Blocking Bricks','Shadows']


  classification = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=target_names)
  oa = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
  confusion = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
  confusion = pd.DataFrame(confusion, index=target_names, columns=target_names)
  each_acc, aa = AA(confusion)
  kappa = cohen_kappa_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
  y_test = y_test.reshape(y_test.shape[0],1,y_test.shape[-1])
  score = model.evaluate(X_test, y_test, batch_size=512)
  Test_Loss =  score[0]*100
  Test_accuracy = score[1]*100
  
  return classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100


dataset = ["Indian_Pines","Pavia_University"]
ds = 0

if ds == 0:
    X, Y, cls, en_type, clusters = import_IP()
elif ds == 1:
    X, Y, cls, en_type, clusters = import_PavU()


patch_size = 10
print(dataset[ds])
print(X.shape, Y.shape)
feature_count = 30
x,pca_model = PCA(X, feature_count,return_model = True)
y=Y
print(x.shape,y.shape)

UNet_acc,best_x_train,best_x_test,best_y_train,best_y_test,best_model = mymodel(x,y, num_epochs = 100, class_num = cls,return_all = True,ws = patch_size, folds = 5)

classification, confusion, Test_Loss, Test_accuracy, oa, each_acc, aa, kappa = reports(best_x_test,best_y_test,dataset[ds],best_model)

print("Testing Results")
print(classification)
print("Overall Accuracy: ",oa)
print("Average Accuracy: ", aa)
print("Kappa Score: ",kappa)
print("confusion:",confusion)
print("test_loss:",Test_Loss)
print("test_acc:",Test_accuracy)
print("each_acc:",each_acc)



class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = 16
        self.confusion_matrix = np.zeros((self.numClass,) * 2)


    def pixelAccuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def CPA(self):
        classAcc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.CPA()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def miou(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(
            self.confusion_matrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, y_pred, y_test):

        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[-1])
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[-1])
        confusionMatrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        confusionMatrix = pd.DataFrame(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):

        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def dice(self,y_pred, y_test):
        """
        calculate the Dice score of two N-d volumes.
        s: the segmentation volume of numpy array
        g: the ground truth volume of numpy array
        """
        assert (len(y_pred.shape) , len(y_test.shape))
        prod = np.multiply(y_pred, y_test)
        s0 = prod.sum()
        dice = (2.0 * s0 + 1e-10) / (y_pred.sum() + y_test.sum() + 1e-10)
        return dice

    def addBatch(self, y_pred, y_test):
        assert y_pred.shape == y_test.shape
        self.confusion_matrix += self.genConfusionMatrix(y_pred, y_test)

    def reset(self):
        self.confusion_matrix = np.zeros((self.numClass, self.numClass))




y_pred = best_model.predict(best_x_test)
y_pred = y_pred.reshape(y_pred.shape[0],y_pred.shape[-1])
y_test = best_y_test.reshape(best_y_test.shape[0],best_y_test.shape[-1])
metric = SegmentationMetric(16)
metric.addBatch(y_pred, y_test)
pa = metric.pixelAccuracy()
cpa = metric.CPA()
mpa = metric.meanPixelAccuracy()
mIoU = metric.miou()
FWIoU=metric.Frequency_Weighted_Intersection_over_Union()
dice=metric.dice(y_pred,y_test)
print('pa is :')
print(pa)
print('cpa is :')
print(cpa)
print('mpa is :')
print(mpa)
print('mIoU is :')
print(mIoU)
print('fwmIoU is :')
print(FWIoU)
print(dice)


height = y.shape[0]
width = y.shape[1]
patch_size = 10
X = PCA(X, numComponents= 30)
X = padWithZeros(X, patch_size//2)


dirs = 'testModel'
if not os.path.exists(dirs):
    os.makedirs(dirs)


clf=joblib.dump(Model, dirs + '/trained_model.m')
clf=joblib.dump(Model, dirs + '/trained_model.ckpt')


outputs = np.zeros((height,width))

for i in range(height):
    for j in range(width):
        target = int(y[i,j])
        if target == 0 :
            continue
        else :
            image_patch = X[i:i + patch_size, j:j + patch_size, :]
            X_test_image = image_patch.reshape(1,  image_patch.shape[0],image_patch.shape[1],image_patch.shape[2],).astype('float32')
            prediction = (best_model.predict(X_test_image))
            prediction = np.argmax(prediction, axis=-1)
            outputs[i][j] = prediction+1


predicted_labels = spectral.imshow(classes = outputs.astype(int),figsize =(20,20))
plt.show()



