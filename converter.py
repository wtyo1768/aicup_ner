import opencc
import os 

def conver2s(prefix, fname):
    path = os.path.join(prefix, fname)
    with open(path, 'r+') as f:

        converter = opencc.OpenCC('t2s.json')

        lines = f.read()
        out = converter.convert(lines)

        fout = open(os.path.join(prefix, fname+'_s'), 'w+')
        fout.write(out)
        # f.write(out)


if __name__ == "__main__":
    print('converting...')
    
    prefix = './data/'
    conver2s('./data/train_2.txt')
    # conver2s(prefix+'train')
    # conver2s(prefix+'dev')
    # conver2s(prefix+'test')

    