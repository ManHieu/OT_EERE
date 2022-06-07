import argparse
from collections import defaultdict
import json
import os
from sklearn.model_selection import KFold, train_test_split
import tqdm
from datapoint_formats import get_datapoint
from readers import cat_xml_reader, mulerx_tsvx_reader, mulerx_tsvx_reader_v2, tsvx_reader
import random
import numpy as np


class Preprocessor(object):
    def __init__(self, dataset, datapoint, intra=True, inter=False):
        self.dataset = dataset
        self.intra = intra
        self.inter = inter
        self.datapoint = datapoint
        self.register_reader(self.dataset)

    def register_reader(self, dataset):
        if self.dataset == 'HiEve':
            self.reader = tsvx_reader
        elif dataset == 'ESL':
            self.reader = cat_xml_reader
        elif 'mulerx' in dataset:
            self.reader = mulerx_tsvx_reader_v2
        else:
            raise ValueError("We have not supported this dataset {} yet!".format(self.dataset))

    def load_dataset(self, dir_name):
        corpus = []
        if self.dataset == 'ESL':
            topic_folders = [t for t in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, t))]
            for topic in tqdm.tqdm(topic_folders):
                topic_folder = os.path.join(dir_name, topic)
                onlyfiles = [f for f in os.listdir(topic_folder) if os.path.isfile(os.path.join(topic_folder, f))]
                for file_name in onlyfiles:
                    file_name = os.path.join(topic, file_name)
                    if file_name.endswith('.xml'):
                        my_dict = self.reader(dir_name, file_name, inter=self.inter, intra=self.intra)
                        if my_dict != None:
                            corpus.append(my_dict)
                            # print(my_dict)
        else:
            onlyfiles = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
            i = 0
            for file_name in tqdm.tqdm(onlyfiles):
                # if i == 11:
                #     break
                # i = i + 1
                my_dict = self.reader(dir_name, file_name)
                if my_dict != None:
                    corpus.append(my_dict)
        
        return corpus
    
    def process_and_save(self, corpus, save_path=None):
        if type(corpus) == list:
            processed_corpus = []
            for my_dict in tqdm.tqdm(corpus):
                len_doc = sum([len(sentence['tokens']) for sentence in my_dict['sentences']])
                doc_info = True
                processed_corpus.extend(get_datapoint(self.datapoint, my_dict, doc_info))
            if save_path != None:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_corpus, f, indent=6)
        else:
            processed_corpus = defaultdict(list)
            for key, topic in corpus.items():
                for my_dict in tqdm.tqdm(topic):
                    len_doc = sum([len(sentence['tokens']) for sentence in my_dict['sentences']])
                    doc_info = True
                    processed_corpus[key].extend(get_datapoint(self.datapoint, my_dict, doc_info))
            if save_path != None:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_corpus, f, indent=6)

        return processed_corpus


if __name__ == '__main__':
    seed = 7890
    import torch
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='mulerx_en', help='Language')
    # parser.add_argument('-la', '--lang', type=str, default='en', help='Language')
    args, remaining_args = parser.parse_known_args()

    dataset = f'{args.data}'

    if dataset == 'HiEve':
        datapoint = 'hieve_datapoint_v3'
        corpus_dir = 'datasets/hievents_v2/processed/'
        processor = Preprocessor(dataset, datapoint)
        corpus = processor.load_dataset(corpus_dir)
        corpus = list(sorted(corpus, key=lambda x: x['doc_id']))
        train, test = train_test_split(corpus, train_size=100.0/120, test_size=20.0/120, random_state=seed)
        train, validate = train_test_split(train, train_size=80.0/100., test_size=20.0/100, random_state=seed)

        processed_path = 'datasets/hievents_v2/train.json'
        train = processor.process_and_save(train, processed_path)

        processed_path = 'datasets/hievents_v2/val.json'
        val = processor.process_and_save(validate, processed_path)

        processed_path = 'datasets/hievents_v2/test.json'
        test = processor.process_and_save(test, processed_path)
    
    elif dataset == 'ESL':
        datapoint = 'ESL_datapoint'
        kfold = KFold(n_splits=5)
        processor = Preprocessor(dataset, datapoint, intra=True, inter=False)
        corpus_dir = './datasets/EventStoryLine/annotated_data/v0.9/'
        corpus = processor.load_dataset(corpus_dir)

        _train, test = [], []
        data = defaultdict(list)
        for my_dict in corpus:
            topic = my_dict['doc_id'].split('/')[0]
            data[topic].append(my_dict)

            if '37/' in my_dict['doc_id'] or '41/' in my_dict['doc_id']:
                test.append(my_dict)
            else:
                _train.append(my_dict)

        # print()
        # processed_path = f"./datasets/EventStoryLine/intra_data.json"
        # processed_data = processor.process_and_save(processed_path, data)

        random.shuffle(_train)
        for fold, (train_ids, valid_ids) in enumerate(kfold.split(_train)):
            try:
                os.mkdir(f"./datasets/EventStoryLine/{fold}")
            except FileExistsError:
                pass

            train = [_train[id] for id in train_ids]
            # print(train[0])
            validate = [_train[id] for id in valid_ids]
        
            processed_path = f"./datasets/EventStoryLine/{fold}/train.json"
            processed_train = processor.process_and_save(train, processed_path)

            processed_path = f"./datasets/EventStoryLine/{fold}/test.json"
            processed_validate = processor.process_and_save(validate, processed_path)
            
            processed_path = f"./datasets/EventStoryLine/{fold}/val.json"
            processed_test = processor.process_and_save(test, processed_path)
    
    elif dataset=='mulerx_en':
        datapoint = 'mulerx_datapoint'
        train_dir = 'datasets/mulerx/subevent-en-20/train/'
        test_dir = 'datasets/mulerx/subevent-en-20/test/'
        val_dir = 'datasets/mulerx/subevent-en-20/dev/'

        processor = Preprocessor(dataset, datapoint)
        train = processor.load_dataset(train_dir)
        test = processor.load_dataset(test_dir)
        validate = processor.load_dataset(val_dir)

        processed_path = 'datasets/mulerx/subevent-en-20/train.json'
        train = processor.process_and_save(train, processed_path)

        processed_path = 'datasets/mulerx/subevent-en-20/val.json'
        val = processor.process_and_save(validate, processed_path)

        processed_path = 'datasets/mulerx/subevent-en-20/test.json'
        test = processor.process_and_save(test, processed_path)

        print("Statistic in HiEve")
        print("Number datapoints in dataset: {}".format(len(train + val + test)))
        print("Number training points: {}".format(len(train)))
        print("Number validate points: {}".format(len(val)))
        print("Number test points: {}".format(len(test)))
    
    elif dataset=='mulerx_da':
        datapoint = 'mulerx_datapoint'
        train_dir = 'datasets/mulerx/subevent-da-20/train/'
        test_dir = 'datasets/mulerx/subevent-da-20/test/'
        val_dir = 'datasets/mulerx/subevent-da-20/dev/'

        processor = Preprocessor(dataset, datapoint)
        train = processor.load_dataset(train_dir)
        test = processor.load_dataset(test_dir)
        validate = processor.load_dataset(val_dir)

        processed_path = 'datasets/mulerx/subevent-da-20/train.json'
        train = processor.process_and_save(train, processed_path)

        processed_path = 'datasets/mulerx/subevent-da-20/val.json'
        val = processor.process_and_save(validate, processed_path)

        processed_path = 'datasets/mulerx/subevent-da-20/test.json'
        test = processor.process_and_save(test, processed_path)

        print("Statistic in HiEve")
        print("Number datapoints in dataset: {}".format(len(train + val + test)))
        print("Number training points: {}".format(len(train)))
        print("Number validate points: {}".format(len(val)))
        print("Number test points: {}".format(len(test)))
    
    elif dataset=='mulerx_es':
        datapoint = 'mulerx_datapoint'
        train_dir = 'datasets/mulerx/subevent-es-20/train/'
        test_dir = 'datasets/mulerx/subevent-es-20/test/'
        val_dir = 'datasets/mulerx/subevent-es-20/dev/'

        processor = Preprocessor(dataset, datapoint)
        
        train = processor.load_dataset(train_dir)
        processed_path = 'datasets/mulerx/subevent-es-20/train.json'
        train = processor.process_and_save(train, processed_path)

        test = processor.load_dataset(test_dir)
        processed_path = 'datasets/mulerx/subevent-es-20/test.json'
        test = processor.process_and_save(test, processed_path)

        validate = processor.load_dataset(val_dir)
        processed_path = 'datasets/mulerx/subevent-es-20/val.json'
        val = processor.process_and_save(validate, processed_path)

        print("Statistic in HiEve")
        print("Number datapoints in dataset: {}".format(len(train + val + test)))
        print("Number training points: {}".format(len(train)))
        print("Number validate points: {}".format(len(val)))
        print("Number test points: {}".format(len(test)))
    
    elif dataset=='mulerx_tr':
        datapoint = 'mulerx_datapoint'
        train_dir = 'datasets/mulerx/subevent-tr-20/train/'
        test_dir = 'datasets/mulerx/subevent-tr-20/test/'
        val_dir = 'datasets/mulerx/subevent-tr-20/dev/'

        processor = Preprocessor(dataset, datapoint)
        train = processor.load_dataset(train_dir)
        test = processor.load_dataset(test_dir)
        validate = processor.load_dataset(val_dir)

        processed_path = 'datasets/mulerx/subevent-tr-20/train.json'
        train = processor.process_and_save(train, processed_path)

        processed_path = 'datasets/mulerx/subevent-tr-20/val.json'
        val = processor.process_and_save(validate, processed_path)

        processed_path = 'datasets/mulerx/subevent-tr-20/test.json'
        test = processor.process_and_save(test, processed_path)

        print("Statistic in HiEve")
        print("Number datapoints in dataset: {}".format(len(train + val + test)))
        print("Number training points: {}".format(len(train)))
        print("Number validate points: {}".format(len(val)))
        print("Number test points: {}".format(len(test)))
    
    elif dataset=='mulerx_ur':
        datapoint = 'mulerx_datapoint'
        train_dir = 'datasets/mulerx/subevent-ur-20/train/'
        test_dir = 'datasets/mulerx/subevent-ur-20/test/'
        val_dir = 'datasets/mulerx/subevent-ur-20/dev/'

        processor = Preprocessor(dataset, datapoint)
        train = processor.load_dataset(train_dir)
        test = processor.load_dataset(test_dir)
        validate = processor.load_dataset(val_dir)

        processed_path = 'datasets/mulerx/subevent-ur-20/train.json'
        train = processor.process_and_save(train, processed_path)

        processed_path = 'datasets/mulerx/subevent-ur-20/val.json'
        val = processor.process_and_save(validate, processed_path)

        processed_path = 'datasets/mulerx/subevent-ur-20/test.json'
        test = processor.process_and_save(test, processed_path)

        print("Statistic in HiEve")
        print("Number datapoints in dataset: {}".format(len(train + val + test)))
        print("Number training points: {}".format(len(train)))
        print("Number validate points: {}".format(len(val)))
        print("Number test points: {}".format(len(test)))
    
    elif dataset=='mulerx_causal_ur':
        datapoint = 'mulerx_datapoint'
        train_dir = 'datasets/mulerx/causal-ur-3/train/'
        test_dir = 'datasets/mulerx/causal-ur-3/test/'
        val_dir = 'datasets/mulerx/causal-ur-3/dev/'

        processor = Preprocessor(dataset, datapoint)
        train = processor.load_dataset(train_dir)
        test = processor.load_dataset(test_dir)
        validate = processor.load_dataset(val_dir)

        processed_path = 'datasets/mulerx/causal-ur-3/train.json'
        train = processor.process_and_save(train, processed_path)

        processed_path = 'datasets/mulerx/causal-ur-3/val.json'
        val = processor.process_and_save(validate, processed_path)

        processed_path = 'datasets/mulerx/causal-ur-3/test.json'
        test = processor.process_and_save(test, processed_path)

        print("Statistic in HiEve")
        print("Number datapoints in dataset: {}".format(len(train + val + test)))
        print("Number training points: {}".format(len(train)))
        print("Number validate points: {}".format(len(val)))
        print("Number test points: {}".format(len(test)))
    
    elif dataset=='mulerx_causal_en':
        datapoint = 'mulerx_datapoint'
        train_dir = 'datasets/mulerx/causal-en-10/train/'
        test_dir = 'datasets/mulerx/causal-en-10/test/'
        val_dir = 'datasets/mulerx/causal-en-10/dev/'

        processor = Preprocessor(dataset, datapoint)
        train = processor.load_dataset(train_dir)
        test = processor.load_dataset(test_dir)
        validate = processor.load_dataset(val_dir)

        processed_path = 'datasets/mulerx/causal-en-10/train.json'
        train = processor.process_and_save(train, processed_path)

        processed_path = 'datasets/mulerx/causal-en-10/val.json'
        val = processor.process_and_save(validate, processed_path)

        processed_path = 'datasets/mulerx/causal-en-10/test.json'
        test = processor.process_and_save(test, processed_path)

        print("Statistic in HiEve")
        print("Number datapoints in dataset: {}".format(len(train + val + test)))
        print("Number training points: {}".format(len(train)))
        print("Number validate points: {}".format(len(val)))
        print("Number test points: {}".format(len(test)))


    