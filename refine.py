
import sys
sys.path.append('/home/dy/aicup/src')
from predict import load_dev
from cocoNLP.extractor import extractor
import re
import pandas as pd
import re
import numpy as np

def drop_tokens(path):
    data = pd.read_csv(path, sep='\t').to_numpy()

    output="article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
    mark = [':', '，', ',', '!', '。', '.']
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

def write_res(output):
    with open('./V1/refine.tsv','w',encoding='utf-8') as f:
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
# def check_phone():

# def check_email():

# def check_name():
def write_refined(refined):
    output="article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
    for k in refined.keys():
        for ele in refined[k]:
            print(ele)
            line = f'{k}\t{ele[0]}\t{ele[1]}\t{ele[2]}\t{ele[3]}\n'
            output+=line
    with open('./V1/fix_id.tsv','w',encoding='utf-8') as f:
        f.write(output)
    
    

def refine_output(pred_result):
    ex = extractor()
    origin_text = load_dev(simplify=False)

    # 一條一條的看TEST DATA
    for arti_id in range(len(origin_text)):
        pattern = re.compile("[A-Z]{1}[1-2]{1}[0-9]{8}", re.I) 
        id_match = pattern.findall(origin_text[arti_id])
        for i in range(len(id_match)):
            position = origin_text[arti_id].find(id_match[i])

            pred_result = is_entity_predicted(pred_result, arti_id, position, 9)
            arr = np.array([position, position+10, id_match[i], 'ID'])
            pred_result[str(arti_id)] = np.append(pred_result[str(arti_id)], arr).reshape(-1,4)
    return pred_result    

                
            

        # if not str(arti_id) in pred_result.keys():
        #     continue
        # arti_pred = pred_result[str(arti_id)]
    
        # mail = ex.extract_email(origin_text[arti_id])
        # cellphones = ex.extract_cellphone(origin_text[arti_id], nation='CHN')
        # locations = ex.extract_locations(origin_text[arti_id])
        # times = ex.extract_time(origin_text[arti_id])
        # name = ex.extract_name(origin_text[arti_id])

        # if times:
        #     print(times)
        # if name:
        #     print(name)

        # if mail:
        #     print(mail)
        # if cellphones:
        #     print(cellphones)
        # if locations:
        #     print(locations)

if __name__ == "__main__":
    # 先將錯誤的符號去除, 寫在refine.tsv中
    path = './V1/output.tsv'
    output = drop_tokens(path)
    write_res(output)

    # 生成dict {artical id:[[ ],[ ],[ ]]}
    path = './V1/refine.tsv'
    pred_res = get_pred_output(path) 

    # 用RULES抓取一些東西
    refined = refine_output(pred_res) #別人寫好的抓資料
    write_refined(refined)
