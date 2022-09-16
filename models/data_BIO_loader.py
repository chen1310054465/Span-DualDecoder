
import torch
import numpy as np
import random
import json
from transformers import BertTokenizer

sentiment2id = {'none': 0, 'positive': 1, 'negative': 2, 'neutral': 3}


def get_spans(tags):
    '''for BIO tag'''
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def get_subject_labels(tags):
    '''for BIO tag'''

    label = {}
    subject_span = get_spans(tags)[0]
    tags = tags.strip().split()
    sentence = []
    for tag in tags:
        sentence.append(tag.strip().split('\\')[0])
    word = ' '.join(sentence[subject_span[0]:subject_span[1] + 1])
    label[word] = subject_span
    return label


def get_object_labels(tags):
    '''for BIO tag'''
    label = {}
    object_spans = get_spans(tags)
    tags = tags.strip().split()
    sentence = []
    for tag in tags:
        sentence.append(tag.strip().split('\\')[0])
    for object_span in object_spans:
        word = ' '.join(sentence[object_span[0]:object_span[1] + 1])
        label[word] = object_span
    return label


class InputExample(object):
    def __init__(self, id, text_a, aspect_num, triple_num, all_label=None, text_b=None):
        """Build a InputExample"""
        self.id = id
        self.text_a = text_a
        self.text_b = text_b
        self.all_label = all_label
        self.aspect_num = aspect_num
        self.triple_num = triple_num


class Instance(object):
    def __init__(self, sentence_pack, args):
        triple_dict = {}
        id = sentence_pack['id']
        aspect_num = 0
        for triple in sentence_pack['triples']:
            aspect = triple['target_tags']
            opinion = triple['opinion_tags']
            sentiment = triple['sentiment']
            subject_label = get_subject_labels(aspect)
            object_label = get_object_labels(opinion)
            objects = list(object_label.keys())
            subject = list(subject_label.keys())[0]
            aspect_num += len(subject_label)
            for i, object in enumerate(objects):
                # 由于数据集的每个triples中aspect只有一个，而opinion可能有多个  需要分开构建
                word = str(subject) + '|' + str(object)
                if word not in triple_dict:
                    triple_dict[word] = []
                triple_dict[word] = (subject_label[subject], object_label[object], sentiment)
        examples = InputExample(id=id, text_a=sentence_pack['sentence'], text_b=None, all_label=triple_dict,
                                aspect_num=aspect_num, triple_num=len(triple_dict))
        self.examples = examples
        self.triple_num = len(triple_dict)
        self.aspect_num = aspect_num


def load_data_instances(sentence_packs, args):
    instances = list()
    triples_num = 0
    aspects_num = 0
    for i, sentence_pack in enumerate(sentence_packs):
        instance = Instance(sentence_pack, args)
        instances.append(instance.examples)
        triples_num += instance.triple_num
        aspects_num += instance.aspect_num
    return instances


def convert_examples_to_features(args, train_instances, max_span_length=8):

    features = []
    num_aspect = 0
    num_triple = 0
    num_opinion = 0
    for ex_index, example in enumerate(train_instances):
        sample = {'id': example.id}
        sample['tokens'] = example.text_a.split(' ')
        sample['text_length'] = len(sample['tokens'])
        sample['triples'] = example.all_label
        sample['sentence'] = example.text_a
        aspect = {}
        opinion = {}
        for triple_name in sample['triples']:
            aspect_span, opinion_span, sentiment = tuple(sample['triples'][triple_name][0]), tuple(
                sample['triples'][triple_name][1]), sample['triples'][triple_name][2]
            num_triple += 1
            if aspect_span not in aspect:
                aspect[aspect_span] = sentiment
                opinion[aspect_span] = [(opinion_span, sentiment)]
            else:
                assert aspect[aspect_span] == sentiment
                opinion[aspect_span].append((opinion_span, sentiment))
        num_aspect += len(aspect)
        num_opinion += len(opinion)

        # if len(aspect) != example.aspect_num:
        #     print('有不同三元组使用重复了aspect:', example.id)

        spans = []
        span_tokens = []
        spans_aspect_label = []
        spans_aspect2opinion_label =[]
        spans_opinion_label = []
        if args.order_shuffle:
            for i in range(max_span_length):
                if sample['text_length'] < i:
                    continue
                for j in range(sample['text_length'] - i):
                    spans.append((j, i + j, i + 1))
                    span_token = ' '.join(sample['tokens'][j:i + j + 1])
                    span_tokens.append(span_token)
                    if (j, i + j) not in aspect:
                        spans_aspect_label.append(0)
                    else:
                        spans_aspect_label.append(sentiment2id[aspect[(j, i + j)]])
        else:
            for i in range(sample['text_length']):
                for j in range(i, min(sample['text_length'], i + max_span_length)):
                    spans.append((i, j, j - i + 1))
                    # sample['spans'].append((i, j, j-i+1))
                    span_token = ' '.join(sample['tokens'][i:j + 1])
                    # sample['span tokens'].append(span_tokens)
                    span_tokens.append(span_token)
                    if (i, j) not in aspect:
                        # sample['spans_aspect_label'].append(0)
                        spans_aspect_label.append(0)
                    else:
                        # sample['spans_aspect_label'].append(sentiment2id[aspect[(i, j)]])
                        spans_aspect_label.append(sentiment2id[aspect[(i, j)]])


        # assert len(sample['span tokens']) == len(sample['spans'])
        assert len(span_tokens) == len(spans)
        for key_aspect in opinion:
            opinion_list = []
            sentiment_opinion = []
            # sample['spans_aspect2opinion_label'].append(key_aspect)
            spans_aspect2opinion_label.append(key_aspect)
            for opinion_span_2_aspect in opinion[key_aspect]:
                opinion_list.append(opinion_span_2_aspect[0])
                sentiment_opinion.append(opinion_span_2_aspect[1])
            assert len(set(sentiment_opinion)) == 1
            opinion_label2triple = []
            # for i in range(sample['text_length']):
            #     for j in range(i, min(sample['text_length'], i + max_span_length)):
            #         if (i, j) not in opinion_list:
            #             opinion_label2triple.append(0)
            #         else:
            #             opinion_label2triple.append(sentiment2id[sentiment_opinion[0]])
            for i in spans:
                if (i[0], i[1]) not in opinion_list:
                    opinion_label2triple.append(0)
                else:
                    opinion_label2triple.append(sentiment2id[sentiment_opinion[0]])
            # sample['spans_opinion_label'].append(opinion_label2triple)
            spans_opinion_label.append(opinion_label2triple)
        # sample['aspect_num'] = len(sample['spans_opinion_label'])
        sample['aspect_num'] = len(spans_opinion_label)
        sample['spans_aspect2opinion_label'] = spans_aspect2opinion_label
        if args.shuffle_data != 0:
            np.random.seed(args.shuffle_data)
            shuffle_ix = np.random.permutation(np.arange(len(spans)))
            spans_np = np.array(spans)[shuffle_ix]
            span_tokens_np = np.array(span_tokens)[shuffle_ix]
            spans_aspect_label_np = np.array(spans_aspect_label)[shuffle_ix]
            spans_opinion_label_shuffle = []
            for spans_opinion_label_split in spans_opinion_label:
                spans_opinion_label_split_np = np.array(spans_opinion_label_split)[shuffle_ix]
                spans_opinion_label_shuffle.append(spans_opinion_label_split_np.tolist())
            spans, span_tokens, spans_aspect_label = spans_np.tolist(), span_tokens_np.tolist(), spans_aspect_label_np.tolist()
            spans_opinion_label = spans_opinion_label_shuffle
        sample['spans'], sample['span tokens'], sample['spans_aspect_label'], sample[
            'spans_opinion_label'] = spans, span_tokens, spans_aspect_label, spans_opinion_label
        features.append(sample)
    return features, num_aspect, num_opinion


def load_data(args, path, if_train=False):
    # sentence_packs = json.load(open(path))
    # # datasets process
    # if if_train:
    #     random.seed(args.RANDOM_SEED)
    #     random.shuffle(sentence_packs)
    # instances = load_data_instances(sentence_packs, args)
    # tokenizer = BertTokenizer.from_pretrained(args.init_vocab, do_lower_case=args.do_lower_case)
    # data_instances, aspect_num, num_opinion = convert_examples_to_features(args, train_instances=instances,
    #                                                                        max_seq_length=args.max_seq_length,
    #                                                                        tokenizer=tokenizer,
    #                                                                        max_span_length=args.max_span_length)
    # list_instance_batch = []
    # for i in range(0, len(data_instances), args.train_batch_size):
    #     list_instance_batch.append(data_instances[i:i + args.train_batch_size])
    # return list_instance_batch

    with open(path, 'r') as f:
        lines = f.readlines()
    if if_train:
        random.seed(args.RANDOM_SEED)
        random.shuffle(lines)
    instances = load_data_instances_txt(lines, args)
    data_instances, aspect_num, num_opinion = convert_examples_to_features(args, train_instances=instances,
                                                                           max_span_length=args.max_span_length)
    list_instance_batch = []
    for i in range(0, len(data_instances), args.train_batch_size):
        list_instance_batch.append(data_instances[i:i + args.train_batch_size])
    return list_instance_batch


def load_data_instances_txt(lines, args):
    sentiment2sentiment = {'NEG': 'negative', 'POS': 'positive', 'NEU': 'neutral'}

    instances = list()
    triples_num = 0
    aspects_num = 0
    opinions_num = 0
    for ex_index, line in enumerate(lines):
        id = str(ex_index) # id
        line = line.strip()
        line = line.split('####')
        sentence = line[0].split()  # sentence
        raw_pairs = eval(line[1])  # triplets
        
        triple_dict = {}
        aspect_num = 0
        opinion_num = 0
        for triple in raw_pairs:
            raw_aspect = triple[0]
            raw_opinion = triple[1]

            if len(raw_aspect) > 2:
                print(1)
            if len(raw_opinion) > 2:
                print(2)

            sentiment = sentiment2sentiment[triple[2]]
            
            if len(raw_aspect) == 1:
                aspect_word = sentence[raw_aspect[0]]
                raw_aspect = [raw_aspect[0], raw_aspect[0]]
            else:
                aspect_word = ' '.join(sentence[raw_aspect[0] : raw_aspect[-1] + 1])
            aspect_label = {}
            aspect_label[aspect_word] = raw_aspect
            aspect_num += len(aspect_label)
            
            if len(raw_opinion) == 1:
                opinion_word = sentence[raw_opinion[0]]
                raw_opinion = [raw_opinion[0], raw_opinion[0]]
            else:
                opinion_word = ' '.join(sentence[raw_opinion[0] : raw_opinion[-1] + 1])
            opinion_label = {}
            opinion_label[opinion_word] = raw_opinion
            opinion_num += len(opinion_label)
            
            word = str(aspect_word) + '|' + str(opinion_word)
            if word not in triple_dict:
                triple_dict[word] = []
                triple_dict[word] = (raw_aspect, raw_opinion, sentiment)
            else:
                print('单句'+id+'中三元组重复出现！')
        examples = InputExample(id=id, text_a=line[0], text_b=None, all_label=triple_dict, aspect_num=aspect_num, 
                                triple_num=len(triple_dict))
        
        instances.append(examples)
        triples_num += len(triple_dict)
        aspects_num += aspect_num
        opinions_num += opinion_num

    return instances


class DataTterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = len(instances)
        self.tokenizer = BertTokenizer.from_pretrained(args.init_vocab, do_lower_case=args.do_lower_case)

    def get_batch(self, batch_num):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        spans_opinion_label_tensor_list = []
        spans_aspect_tensor_list = []
        sentence_length = []


        max_tokens = self.args.max_seq_length
        max_spans = 0
        for i, sample in enumerate(self.instances[batch_num]):
            tokens = sample['tokens']
            spans = sample['spans']
            spans_ner_label = sample['spans_aspect_label']
            spans_aspect2opinion_labels = sample['spans_aspect2opinion_label']
            spans_aspect_labels = []
            for spans_aspect2opinion_label in spans_aspect2opinion_labels:
                spans_aspect_labels.append((i, spans_aspect2opinion_label[0], spans_aspect2opinion_label[1]))
            spans_opinion_label = sample['spans_opinion_label']
            tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_aspect_labels_tensor, spans_opinion_tensor = self.get_input_tensors(
                self.tokenizer, tokens, spans,
                spans_ner_label, spans_aspect_labels, spans_opinion_label)
            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            spans_aspect_tensor_list.append(spans_aspect_labels_tensor)
            spans_opinion_label_tensor_list.append(spans_opinion_tensor)
            assert (bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1])
            # tokens和spans的最大个数被设定为固定值
            # if (tokens_tensor.shape[1] > max_tokens):
            #     max_tokens = tokens_tensor.shape[1]
            if (bert_spans_tensor.shape[1] > max_spans):
                max_spans = bert_spans_tensor.shape[1]
            sentence_length.append(sample['text_length'])
        sentence_length = torch.Tensor(sentence_length)

        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_aspect_tensor = None
        final_spans_opinion_label_tensor = None
        final_spans_mask_tensor = None
        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_aspect_tensor, spans_opinion_label_tensor in zip(
                tokens_tensor_list, bert_spans_tensor_list,
                spans_ner_label_tensor_list, spans_aspect_tensor_list, spans_opinion_label_tensor_list):
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = torch.full([1, num_tokens], 1, dtype=torch.long)
            if tokens_pad_length > 0:
                pad = torch.full([1, tokens_pad_length], self.tokenizer.pad_token_id, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_pad = torch.full([1, tokens_pad_length], 0, dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)

            # padding for spans
            num_spans = bert_spans_tensor.shape[1]
            num_aspect = spans_aspect_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1, num_spans], 1, dtype=torch.long)
            if spans_pad_length > 0:
                pad = torch.full([1, spans_pad_length, bert_spans_tensor.shape[2]], 0, dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)
                mask_pad = torch.full([1, spans_pad_length], 0, dtype=torch.long)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=1)
                spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=1)
                opinion_mask_pad = torch.full([1, num_aspect, spans_pad_length], 0, dtype=torch.long)
                spans_opinion_label_tensor = torch.cat((spans_opinion_label_tensor, opinion_mask_pad), dim=-1)

            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_mask_tensor = spans_mask_tensor
                final_spans_aspect_tensor = spans_aspect_tensor.squeeze(0)
                final_spans_opinion_label_tensor = spans_opinion_label_tensor.squeeze(0)
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor, tokens_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat((final_bert_spans_tensor, bert_spans_tensor), dim=0)
                final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
                final_spans_mask_tensor = torch.cat((final_spans_mask_tensor, spans_mask_tensor), dim=0)
                final_spans_aspect_tensor = torch.cat(
                    (final_spans_aspect_tensor, spans_aspect_tensor.squeeze(0)), dim=0)
                final_spans_opinion_label_tensor = torch.cat(
                    (final_spans_opinion_label_tensor, spans_opinion_label_tensor.squeeze(0)), dim=0)

        # 注意，特征中最大span间隔不一定为设置的max_span_length，这是因为bert分词之后造成的span扩大了。
        final_tokens_tensor = final_tokens_tensor.to(self.args.device)
        final_attention_mask = final_attention_mask.to(self.args.device)
        final_bert_spans_tensor = final_bert_spans_tensor.to(self.args.device)
        final_spans_mask_tensor = final_spans_mask_tensor.to(self.args.device)
        final_spans_ner_label_tensor = final_spans_ner_label_tensor.to(self.args.device)
        final_spans_aspect_tensor = final_spans_aspect_tensor.to(self.args.device)
        final_spans_opinion_label_tensor = final_spans_opinion_label_tensor.to(self.args.device)
        return final_tokens_tensor, final_attention_mask, final_bert_spans_tensor, final_spans_mask_tensor, \
               final_spans_ner_label_tensor, final_spans_aspect_tensor, final_spans_opinion_label_tensor, \
               sentence_length

    def get_input_tensors(self, tokenizer, tokens, spans, spans_label, spans_aspect_label, spans_opinion_label):
        start2idx = []
        end2idx = []
        bert_tokens = []
        bert_tokens.append(tokenizer.cls_token)
        for token in tokens:
            start2idx.append(len(bert_tokens))
            sub_tokens = tokenizer.tokenize(token)
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens) - 1)
        bert_tokens.append(tokenizer.sep_token)
        indexed_tokens = tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
        # 在bert分出subword之后  需要对原有的aspect span进行补充
        spans_aspect_label = [[aspect_span[0], start2idx[aspect_span[1]], end2idx[aspect_span[2]]] for aspect_span in spans_aspect_label]
        bert_spans_tensor = torch.tensor([bert_spans])
        spans_ner_label_tensor = torch.tensor([spans_label])
        spans_aspect_label = torch.tensor([spans_aspect_label])
        spans_opinion_tensor = torch.tensor([spans_opinion_label])
        return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_aspect_label, spans_opinion_tensor
