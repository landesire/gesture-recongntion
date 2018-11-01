import  numpy as np

file = open("S_feature_all.txt")
# for line in file.readlines():
#     current_str_array=line.split(",")
#     print(current_str_array[0])
#     print(current_str_array[1])
#     print(len(current_str_array))

data = np.loadtxt(file, delimiter = ',')
print(data.shape)
#生成标注类别1-10类
for i in range(len(data)):
    label = (i /800)
    label_list.append(label)
        #print(label)
y_label = np.array(label_list)
y_label = y_label.reshape(-1, 1)
# print y_train2
print(len(y_label))
all_data = np.hstack((y_label, data))
print("this is all data")
print(all_data[255][125])
print("all data shape"+str(all_data.shape))