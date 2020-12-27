import numpy as np
from src.predict import load_dev
from src.dataset import loadInputFile


def create_pseudo_data(path, outpath):
    with open(path, 'r') as f:
        text = f.read()
    test_doc = load_dev(simplify=False)
    text = text.split('\n')[1:]
    previous_done = -1
    out = ''
    for line in text:
        entity_attr = line.split('\t')
        if (len(entity_attr) != 5):
            continue
        artid, sp, ep, entity, entype = entity_attr
        artid = int(artid)
        if artid != previous_done:
            if not artid == 0:
                out+='\n--------------------\n\n'
            out+=test_doc[artid]
            out+='\narticle_id\tstart_position\tend_position\tentity_text\tentity_type\n'
            previous_done = artid 
            
        out += f'{artid}\t{sp}\t{ep}\t{entity}\t{entype}\n'

    out+='\n--------------------\n\n\n'
    with open(outpath, 'w+') as f:
        print('writing', outpath, '...')
        f.write(out)
    
    
if __name__ == "__main__":
    # fpath = './pred/0.7693305_Fix.tsv'
    fpath = './pred/bagging.tsv'
    pseudo_data_path = './data/pseudo_data.txt'
    create_pseudo_data(fpath, pseudo_data_path)
    # fpath = '/home/dy/flat-chinese-ner/data/pseudo_data.txt'
    # t, m, _ = loadInputFile(fpath)
    # print(m[0:50])



