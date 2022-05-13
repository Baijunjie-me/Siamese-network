#confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
classes = ['sailboat', 'cargo', 'tanker']
confusion_matrix = np.array([(37,0,1),(0,30,0),(0,0,26)],dtype=np.float64)

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  #按照像素显示出矩阵
plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=-45)
plt.yticks(tick_marks, classes)

thresh = confusion_matrix.max() / 2.
#iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
#ij配对，遍历矩阵迭代器
iters = np.reshape([[[i,j] for j in range(3)] for i in range(3)],(confusion_matrix.size,2))
for i, j in iters:
    plt.text(j, i, format(confusion_matrix[i, j]))   #显示对应的数字

plt.ylabel('Real label')
plt.xlabel('Prediction')
plt.tight_layout()
plt.show()

#iter:https://blog.csdn.net/zlf19910726/article/details/79172333
#np:https://blog.csdn.net/a486259/article/details/85238726
#array manipulate:https://blog.csdn.net/zhangchuang601/article/details/79626511
#color:https://matplotlib.org/users/colormaps.html


