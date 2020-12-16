
import sys
sys.path.append('/home/dy/aicup/src')
from predict import load_dev
from cocoNLP.extractor import extractor
import re

def drop_tokens():

    return

def check_id(text):



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
        else:
            pred_result[res[0]].append(res[1:])
    return pred_result



# def is_entity_predicted(arti_id, position):


# def check_phone():

# def check_email():

# def check_name():


def refine_output(pred_result):
    ex = extractor()
    origin_text = load_dev(simplify=True)
    # print(len(origin_text))
    for arti_id in range(len(origin_text)):
        if not str(arti_id) in pred_result.keys():
            continue
        arti_pred = pred_result[str(arti_id)]
    
        mail = ex.extract_email(origin_text[arti_id])
        cellphones = ex.extract_cellphone(origin_text[arti_id], nation='CHN')
        locations = ex.extract_locations(origin_text[arti_id])
        # times = ex.extract_time(origin_text[arti_id])
        name = ex.extract_name(origin_text[arti_id])

        # if times:
        #     print(times)
        if name:
            print(name)

        # if mail:
        #     print(mail)
        # if cellphones:
        #     print(cellphones)
        # if locations:
        #     print(locations)

if __name__ == "__main__":
    path = './V1/output.tsv'

    pred_res = get_pred_output(path)
    refine_output(pred_res)