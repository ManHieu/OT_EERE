import os
import random


def resampling(dir_name, file_name, new_dir, sample_rate=1.0):
    
    new_lines = []
    for line in open(dir_name + file_name, encoding='UTF-8'):
        _line = line.split('\t')
        if _line[0] == 'Relation':
            rel = _line[3]
            print(rel)
            if rel == 'NoRel':
                if random.random() < sample_rate:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
        
    with open(new_dir + file_name, 'w', encoding='UTF-8') as f:
        f.writelines(new_lines)


if __name__ == '__main__':

    data_type = 'subevent'
    langs = ['da', 'en', 'es', 'tr', 'ur']
    splits = ['train', 'dev', 'test']
    sample_rate = 0.5
    for lang in langs:
        for split in splits:
            dir_name = f'datasets/mulerx/{data_type}-{lang}-10/{split}/'
            onlyfiles = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f)) and f.endswith('.tsvx')]
            new_dir = f'datasets/mulerx/{data_type}-{lang}-20/{split}/'
            try:
                os.mkdir(f'datasets/mulerx/{data_type}-{lang}-20/')
            except FileExistsError:
                pass
            try:
                os.mkdir(new_dir)
            except FileExistsError:
                pass
            for file in onlyfiles:
                resampling(dir_name=dir_name, file_name=file, new_dir=new_dir, sample_rate=sample_rate)


