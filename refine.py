import sys
from src.predict import load_dev
import re
import pandas as pd
import numpy as np


def drop_tokens(read_path):
    data = pd.read_csv(read_path, sep='\t').to_numpy()

    output="article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
    mark = [':', '，', ',', '!', '。', '.', ' ']
    for i in range(len(data)):
        # 去除頭部符號
        if data[i][3][0] in mark: 
            data[i][1] = str(int(data[i][1])+1)
            data[i][3] = data[i][3][1:]
            output += f'{data[i][0]}\t{data[i][1]}\t{data[i][2]}\t{data[i][3]}\t{data[i][4]}\n'
        
        #去除尾部符號
        elif data[i][3][-1] in mark: 
            data[i][2] = str(int(data[i][2])-1)
            data[i][3] = data[i][3][:-1]
            output += f'{data[i][0]}\t{data[i][1]}\t{data[i][2]}\t{data[i][3]}\t{data[i][4]}\n'

        #其他的不改變
        else:
            pass
            output += f'{data[i][0]}\t{data[i][1]}\t{data[i][2]}\t{data[i][3]}\t{data[i][4]}\n'
    
    return output


def write_res(output, outpath):
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(output)
    return


def get_pred_output(path):
    origin_text = load_dev(simplify=False)
    out = ''
    pred_result = {}

    with open(path) as f:
        text = f.read()

    text = text.split('\n')
    out += text[0]
    for line in text[1:]:
        res = line.split('\t')
        if len(res) != 5:
            continue
        if not res[0] in pred_result.keys():
            pred_result[res[0]] = []
            pred_result[res[0]].append(res[1:])
            #pred_result['1'] = []
        else:
            pred_result[res[0]].append(res[1:])
    for k in pred_result.keys():
        pred_result[k] = np.array(pred_result[k])
    return pred_result


def is_entity_predicted(pred_result, arti_id, position, offset):
    delete_list = []
    for j in range(len(pred_result[str(arti_id)])): # 進到有ID的那格裡面
        if (int(pred_result[str(arti_id)][j][0]) >= position and int(pred_result[str(arti_id)][j][0]) <= position+offset) or (int(pred_result[str(arti_id)][j][0]) <= position and int(pred_result[str(arti_id)][j][1])-1 >= position):
            delete_list.append(j)

    pred_result[str(arti_id)] = np.delete(pred_result[str(arti_id)], delete_list, axis=0)
    #print(pred_result[str(arti_id)])
    return pred_result


def write_refined(refined):
    output="article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
    for k in refined.keys():
        for ele in refined[k]:
            line = f'{k}\t{ele[0]}\t{ele[1]}\t{ele[2]}\t{ele[3]}\n'
            output+=line
    with open('./V1/refined.tsv','w',encoding='utf-8') as f:
        f.write(output)


def rule(origin_text, pred_result, r, length, entity, replace, ignorecase=True):
    for arti_id in range(len(origin_text)):
        if ignorecase:
            pattern = re.compile(r, re.I)
        else:
            pattern = re.compile(r)
        match = pattern.findall(origin_text[arti_id])
        print(match)
        beg_tmp = -1
        for i in range(len(match)): 
            position = origin_text[arti_id].find(match[i], beg_tmp+1)
            beg_tmp = position
            pred_result = is_entity_predicted(pred_result, arti_id, position, length-1)

            if replace:
                arr = np.array([position, position+length, match[i], entity])
                pred_result[str(arti_id)] = np.append(pred_result[str(arti_id)], arr).reshape(-1,4)
    return pred_result


def refine_output(pred_result):
    # ex = extractor()
    origin_text = load_dev(simplify=False)

    pred_result = rule(origin_text, pred_result, "[A-Z]{1}[1-2]{1}[0-9]{8}", 10, "ID", True)
    pred_result = rule(origin_text, pred_result, "09[0-9]{8}", 10, "contact", True)
    pred_result = rule(origin_text, pred_result, "google", 6, "profession", True)
    pred_result = rule(origin_text, pred_result, "line", 4, "contact", True)
    pred_result = rule(origin_text, pred_result, "cd4", 3, "", False)
    pred_result = rule(origin_text, pred_result, "N95", 3, "", False)
    pred_result = rule(origin_text, pred_result, "h1n1", 4, "clinical_event", True)
    pred_result = rule(origin_text, pred_result, "sars", 4, "clinical_event", True)
    pred_result = rule(origin_text, pred_result, "PrEP", 4, "", False, ignorecase = False)
    
    return pred_result    


if __name__ == "__main__":
    # 先將錯誤的符號去除, 寫在refine.tsv中

    output = drop_tokens('./pred/bagging.tsv')
    write_res(output, './pred/refined.tsv')

    # 生成dict {artical id:[[ ],[ ],[ ]]}
    path = './pred/refined.tsv'
    pred_res = get_pred_output(path) 

    # 用RULES抓取一些東西
    refined = refine_output(pred_res) #別人寫好的抓資料
    write_refined(refined)