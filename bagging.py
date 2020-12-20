import numpy as np


def vote2(all_pred, reshape_dim1, reshape_dim2):
    result = np.zeros((reshape_dim1, reshape_dim2))
    for i in range(len(all_pred)):
        for j in range(len(all_pred[i])):
            tmp = np.bincount(np.array(all_pred[i][j].astype(int)))
            result[i][j] = np.argmax(tmp)
    print(result)
    return result

models_path = ['path1', 'path2', 'path3']

path1 = [[1, 2, 3], [4, 5, 6]]
path2 = [[10, 11, 12], [13, 14, 15]]
path3 = [[19, 20, 21], [22, 23, 24]]
mapping = {'path1': path1, 'path2': path2, 'path3': path3}

all_pred = np.array([path1, path2, path3])
print(all_pred.shape)

combined_pred = np.zeros((2, 3, 3))

for i, row in enumerate(all_pred):
    print(i)
    for j, col in enumerate(row):
    
        # print(combined_pred[i][j])
        combined_pred[j][i] = [all_pred[m][j][i] for m in range(len(models_path))]

#print(combined_pred)
vote2(combined_pred, 2, 3)
exit()












def vote(all_pred):
    count = []
    for i in range(len(all_pred)):
        count.append(np.bincount(all_pred[i]))
    
    result = []
    for i in range(len(count)):
        result.append(np.argmax(count[i]))
    
    return result

all_pred_flatten = []
for path_num in range(len(models_path)):

    '''這邊GET每個MODEL的PREDICTION'''
    # model = Predictor(torch.load(path, map_location=device))
    # pred = model.predict(
    #     datasets['aicup_dev'],
    #     seq_len_field_name='seq_len',
    # )
    # pred = pred['pred']
    
    pred = mapping[models_path[path_num]]
    pred = np.array(pred)
    reshape_dim1 = len(pred)
    '''所有模型都flatten'''
    pred = pred.reshape(1, -1)[0]
    all_pred_flatten.append(pred)

all_pred = np.zeros(3,3,len(models_path))
len_of_ele = len(all_pred_flatten[0])
for i in range(len_of_ele):
    tmp = []
    for j in range(len(all_pred_flatten)):
        tmp.append(all_pred_flatten[j][i])
    all_pred.append(tmp)

all_pred = np.array(vote(all_pred)).reshape(reshape_dim1, -1)

print(all_pred)









