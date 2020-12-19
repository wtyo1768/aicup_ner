import numpy as np


def vote(all_pred):

    staked_pred = np.column_stack(all_pred)
    print('stacked pred_shape:', staked_pred.shape)

    result = []
    for i in range(len(staked_pred)):
        max_idx = np.argmax(np.bincount(staked_pred[i]))
        result.append(max_idx)
    return result


if __name__ == "__main__":
    
    pred1 = np.load('./V1/pred/pred0.npy', allow_pickle=True)
    # pred2 = np.load('./V1/pred/pred1.npy', allow_pickle=True)
    # pred3 = np.load('./V1/pred/pred2.npy', allow_pickle=True)
    # print(pred1.shape)
    # a = [1,2,3]
    # print(np.array([a,a,a]))
    # print(np.ones((3,3)))
    
    voted_res = vote([pred1, pred1, pred1])
    # print(voted_res)
    # print(np.stack([np.ones(3), np.zeros(3) ], axis=-1))