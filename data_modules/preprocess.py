from collections import defaultdict
import json
import os
from sklearn.model_selection import train_test_split

import tqdm
from datapoint_formats import get_datapoint
from readers import tsvx_reader
import random
random.seed(1741)
import numpy as np
np.random.seed(1741)


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
        # elif dataset == 'ESL':
        #     self.reader = cat_xml_reader
        # elif dataset == 'Causal-TB':
        #     self.reader = ctb_cat_reader
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
    
    def process_and_save(self, save_path, corpus):
        if type(corpus) == list:
            processed_corpus = []
            for my_dict in tqdm.tqdm(corpus):
                len_doc = sum([len(sentence['tokens']) for sentence in my_dict['sentences']])
                if len_doc < 400:
                    doc_info = True
                else:
                    doc_info = False
                processed_corpus.extend(get_datapoint(self.datapoint, my_dict, doc_info))
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(processed_corpus, f, indent=6)
        else:
            processed_corpus = defaultdict(list)
            for key, topic in corpus.items():
                for my_dict in tqdm.tqdm(topic):
                    len_doc = sum([len(sentence['tokens']) for sentence in my_dict['sentences']])
                    if len_doc < 400:
                        doc_info = True
                    else:
                        doc_info = False
                    processed_corpus[key].extend(get_datapoint(self.datapoint, my_dict, doc_info))
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(processed_corpus, f, indent=6)

        return processed_corpus


if __name__ == '__main__':
    dataset = 'HiEve'

    if dataset == 'HiEve':
        datapoint = 'hieve_datapoint_v2'
        corpus_dir = 'datasets/hievents_v2/processed/'
        processor = Preprocessor(dataset, datapoint)
        corpus = processor.load_dataset(corpus_dir)
        corpus = list(sorted(corpus, key=lambda x: x['doc_id']))
        train, test = train_test_split(corpus, train_size=0.8, test_size=0.2)
        train, validate = train_test_split(train, train_size=0.75, test_size=0.25)

        processed_path = 'datasets/hievents_v2/train.json'
        train = processor.process_and_save(processed_path, train)

        processed_path = 'datasets/hievents_v2/val.json'
        val = processor.process_and_save(processed_path, validate)

        processed_path = 'datasets/hievents_v2/test.json'
        test = processor.process_and_save(processed_path, test)

        print("Statistic in HiEve")
        print("Number datapoints in dataset: {}".format(len(train + val + test)))
        print("Number training points: {}".format(len(train)))
        print("Number validate points: {}".format(len(val)))
        print("Number test points: {}".format(len(test)))
    