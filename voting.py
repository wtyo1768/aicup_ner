import numpy as np
from src.dataset  import romove_redundant_str
from src.predict import load_dev, split_to_pred_per_article, write_result, count_article_length
from aicup_dataset import get_label_vocab

def vote(all_pred):
    print(all_pred[0][0:10])
    print(all_pred[1][0:10])
    print(all_pred[2][0:10])

    staked_pred = np.column_stack(all_pred)
    print('stacked pred_shape:', staked_pred.shape)

    result = []
    for i in range(len(staked_pred)):
        max_idx = np.argmax(np.bincount(staked_pred[i]))
        result.append(max_idx)
    print(result[0:10])
    return result


if __name__ == "__main__":

    total_pred = []
    
    for m in ['flat_lk']:
        for i in range(10):
            total_pred.append(np.load(f'./pred/{m}/{i}.npy'))
    for m in ['flat']:
        for i in range(10):
            total_pred.append(np.load(f'./pred/{m}/{i}.npy'))
    
    vote_result = vote(total_pred)
    dev_data = load_dev()
    origin_data = load_dev(simplify=False)

    offset_map = []
    for idx in range(len(dev_data)):
        dev_data[idx], map_arr = romove_redundant_str(dev_data[idx], dev_mode=True)
        offset_map.append(map_arr)

    label_vocab = get_label_vocab(data_type='default')
    pred = [label_vocab.to_word(ele) for ele in vote_result]
    pred_per_article = split_to_pred_per_article([pred], count_article_length(dev_data))
    print('writing file...')
    write_result(
        dev_data, 
        pred_per_article, 
        offset_map, 
        origin_data, 
        output_path='./pred/bagging.tsv'
    )
    # print(voted_res)
    # print(np.stack([np.ones(3), np.zeros(3) ], axis=-1))