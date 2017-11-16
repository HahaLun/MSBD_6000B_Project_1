

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from numpy import genfromtxt

work_path = "D:/_Lun/GoogleDrive/Information/HKUST/MSBD/MSBD 6000B Deep Learning/HW 1/"

X = genfromtxt(work_path + "traindata.csv", delimiter=',')
y = genfromtxt(work_path + "trainlabel.csv", delimiter=',')

X_test = genfromtxt(work_path + "testdata.csv", delimiter=',')

param = {'max_depth':21 , 'min_samples_leaf': 1,'max_features': 11,'min_samples_split':2 ,'bootstrap': False, 'criterion': "entropy", "n_estimators": 91}

rf = RandomForestClassifier( random_state=10 ).set_params(**param)
rf.fit(X,y)
y_test = rf.predict(X_test)


np.savetxt("project1_20039916.csv", y_test , fmt = "%i" , delimiter=",")
