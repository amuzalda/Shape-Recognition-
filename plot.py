import sys
import cv2
import numpy as np
import pylab as pl
from sklearn.metrics import precision_recall_curve

recall=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
precision=[1,1,1,1,1,1,1,1,1,1,1]
def main(recall,precision,pr):
	pl.clf()

    #for precision, recall, label in lines:
	f=pl.plot(recall, precision)
	pl.xlabel('Recall')
	pl.ylabel('Precision')
	pl.ylim([0.0, 2.5])
	pl.xlim([0.0, 1.0])
	pl.title('Precision-Recall')
	pl.legend(loc="upper right")
	#pl.show()
	pl.savefig(pr)
	img=cv2.imread(pr)
	cv2.imshow('pr',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


main(recall,precision,'pr_curve.tiff')