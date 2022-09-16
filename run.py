import os
import argparse
import torch
import torch.nn.functional as F
from models.data_BIO_loader import load_data, DataTterator
from models.model import opinion_stage_features, Aspect_Extractor, Opinion_Extractor, Loss
from models.Metric import Metric
import tqdm
from transformers import AdamW, BertModel
from thop import profile, clever_format
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

def train(args):
    if args.dataset_path == './datasets/BIO_form/':
        train_path = args.dataset_path + args.dataset + "/train.json"
        dev_path = args.dataset_path + args.dataset + "/dev.json"
        test_path = args.dataset_path + args.dataset + "/test.json"
    else:
        train_path = args.dataset_path + args.dataset + "/train_triplets.txt"
        dev_path = args.dataset_path + args.dataset + "/dev_triplets.txt"
        test_path = args.dataset_path + args.dataset + "/test_triplets.txt"

    print('-------------------------------')
    print('开始加载测试集')
    test_datasets = load_data(args, test_path, if_train=False)
    testset = DataTterator(test_datasets, args)
    print('测试集加载完成')
    print('-------------------------------')

    Bert = BertModel.from_pretrained(args.init_model)
    bert_config = Bert.config
    Bert.to(args.device)
    bert_param_optimizer = list(Bert.named_parameters())
    aspect_model = Aspect_Extractor(args, bert_config)
    aspect_model.to(args.device)
    aspect_param_optimizer = list(aspect_model.named_parameters())
    if args.task == 'triples':
        opinion_model = Opinion_Extractor(args, bert_config)
        opinion_model.to(args.device)
        opinion_param_optimizer = list(opinion_model.named_parameters())
        training_param_optimizer = [
            {'params': [p for n, p in bert_param_optimizer]},
            {'params': [p for n, p in aspect_param_optimizer], 'lr': args.task_learning_rate},
            {'params': [p for n, p in opinion_param_optimizer], 'lr': args.task_learning_rate}]
    else:
        training_param_optimizer = [
            {'params': [p for n, p in bert_param_optimizer]},
            {'params': [p for n, p in aspect_param_optimizer], 'lr': args.task_learning_rate}]
    optimizer = AdamW(training_param_optimizer, lr=args.learning_rate)

    if args.muti_gpu:
        Bert = torch.nn.DataParallel(Bert)
        aspect_model = torch.nn.DataParallel(aspect_model)
        opinion_model = torch.nn.DataParallel(opinion_model)

    if args.mode == 'train':
        print('-------------------------------')
        print('开始加载训练与验证集')
        train_datasets = load_data(args, train_path, if_train=True)
        dev_datasets = load_data(args, dev_path, if_train=False)
        trainset = DataTterator(train_datasets, args)
        devset = DataTterator(dev_datasets, args)
        print('训练集与验证集加载完成')
        print('-------------------------------')

        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        tot_loss = 0
        best_aspect_f1, best_opinion_f1, best_APCE_f1, best_pairs_f1, best_triple_f1 = 0,0,0,0,0
        best_aspect_epoch, best_opinion_epoch, best_APCE_epoch, best_pairs_epoch, best_triple_epoch= 0,0,0,0,0
        for i in range(args.epochs):
            print('Epoch:{}'.format(i))
            for j in tqdm.trange(trainset.batch_count):
            # for j in range(trainset.batch_count):
                tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
                spans_aspect_tensor, spans_opinion_label_tensor, sentence_length = trainset.get_batch(j)
                bert_output = Bert(input_ids=tokens_tensor, attention_mask=attention_mask)
                aspect_class_logits, aspect_self_attention, aspect_cross_attention, spans_embedding = aspect_model(bert_output.last_hidden_state,
                                                                                                  attention_mask,
                                                                                                  bert_spans_tensor,
                                                                                                  spans_mask_tensor)
                if args.task == 'triples':
                    all_span_opinion_tensor,all_span_aspect_tensor,all_bert_embedding,all_attention_mask,all_spans_embedding,all_span_mask = opinion_stage_features(
                        bert_output.last_hidden_state, attention_mask, bert_spans_tensor, spans_mask_tensor,
                        spans_embedding,spans_aspect_tensor,spans_opinion_label_tensor)
                    opinion_class_logits, opinion_self_attention, opinion_cross_attention, opinion_attention = opinion_model(
                        all_bert_embedding,
                        all_attention_mask,
                        all_spans_embedding,
                        all_span_mask,
                        all_span_aspect_tensor)
                    loss = Loss(args, spans_ner_label_tensor, aspect_class_logits, all_span_opinion_tensor,
                            opinion_class_logits, spans_mask_tensor, all_span_mask)
                if args.accumulation_steps > 1:
                    loss = loss / args.accumulation_steps
                    loss.backward()
                    if ((j + 1) % args.accumulation_steps) == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                tot_loss += loss.item()

            print('Loss:', tot_loss)
            tot_loss = 0

            print('Evaluating, please wait')
            aspect_result, opinion_result, apce_result, pair_result, triplet_result = eval(Bert, aspect_model,
                                                                                           opinion_model, testset, args)
            # aspect_result, opinion_result, apce_result, pair_result, triplet_result = eval(Bert, aspect_model,
            #                                                                                opinion_model, devset, args)
            print('Evaluating complete')
            if aspect_result[2] > best_aspect_f1:
                best_aspect_f1 = aspect_result[2]
                best_aspect_precision = aspect_result[0]
                best_aspect_recall = aspect_result[1]
                best_aspect_epoch = i

            if opinion_result[2] > best_opinion_f1:
                best_opinion_f1 = opinion_result[2]
                best_opinion_precision = opinion_result[0]
                best_opinion_recall = opinion_result[1]
                best_opinion_epoch = i

            if apce_result[2] > best_APCE_f1:
                best_APCE_f1 = apce_result[2]
                best_APCE_precision = apce_result[0]
                best_APCE_recall = apce_result[1]
                best_APCE_epoch = i

            if pair_result[2] > best_pairs_f1:
                best_pairs_f1 = pair_result[2]
                best_pairs_precision = pair_result[0]
                best_pairs_recall = pair_result[1]
                best_pairs_epoch = i

            if triplet_result[2] > best_triple_f1:
                model_path = args.model_dir +args.dataset +'_'+ str(triplet_result[2]) + '.pt'
                state = {
                    "bert_model": Bert.state_dict(),
                    "aspect_model": aspect_model.state_dict(),
                    "opinion_model": opinion_model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                torch.save(state, model_path)
                print("_________________________________________________________")
                print("best model save")
                print("_________________________________________________________")
                best_triple_f1 = triplet_result[2]
                best_triple_precision = triplet_result[0]
                best_triple_recall = triplet_result[1]
                best_triple_epoch = i
        print(
            'best aspect epoch: {}\tbest aspect precision: {:.8f}\tbest aspect recall: {:.8f}\tbest aspect f1: {:.8f}\n\n'.
            format(best_aspect_epoch, best_aspect_precision, best_aspect_recall, best_aspect_f1))
        print(
            'best opinion epoch: {}\tbest opinion precision: {:.8f}\tbest opinion recall: {:.8f}\tbest opinion f1: {:.8f}\n\n'.
            format(best_opinion_epoch, best_opinion_precision, best_opinion_recall, best_opinion_f1))

        print('best APCE epoch: {}\tbest APCE precision: {:.8f}\tbest APCE recall: {:.8f}\tbest APCE f1: {:.8f}\n\n'.
              format(best_APCE_epoch, best_APCE_precision, best_APCE_recall, best_APCE_f1))
        print('best pair epoch: {}\tbest pair precision: {:.8f}\tbest pair recall: {:.8f}\tbest pair f1: {:.8f}\n\n'.
              format(best_pairs_epoch,best_pairs_precision,best_pairs_recall, best_pairs_f1))
        print('best triple epoch: {}\tbest triple precision: {:.8f}\tbest triple recall: {:.8f}\tbest triple f1: {:.8f}\n\n'.
              format(best_triple_epoch,best_triple_precision,best_triple_recall,best_triple_f1))

    print("Features build completed")
    print("Evaluation on testset:")
    model_path = args.model_dir + args.dataset+'_'+str(best_triple_f1) + '.pt'
    # model_path = args.model_dir +args.dataset +'_'+ str(0.6345381526104418) + '.pt'
    state = torch.load(model_path)
    Bert.load_state_dict(state['bert_model'])
    aspect_model.load_state_dict(state['aspect_model'])
    opinion_model.load_state_dict(state['opinion_model'])
    eval(Bert, aspect_model, opinion_model, testset, args)

def eval(bert_model, aspect_model, opinion_model, dataset, args):
    with torch.no_grad():
        bert_model.eval()
        aspect_model.eval()
        opinion_model.eval()
        gold_instances, gold_aspect_result, gold_aspect_with_sentiment, gold_opinion_result, pred_aspect_result, \
        pred_aspect_with_sentiment, pred_aspect_sentiment_logit,pred_opinion_result, pred_opinion_sentiment_logit,\
        aspect_mask_list = [],[],[],[],[],[],[],[],[],[]
        for j in range(dataset.batch_count):
            tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
            spans_aspect_tensor, spans_opinion_label_tensor, sentence_length = dataset.get_batch(j)

            if j ==0:
                bert_model.to("cpu")
                flop_bert, para_bert = profile(bert_model, inputs=(tokens_tensor, attention_mask,), custom_ops={})
                macs, param = clever_format([flop_bert, para_bert], "%.3f")
                print("BERT MACs: ", macs, "BERT Params: ", param)

            bert_features = bert_model(input_ids=tokens_tensor, attention_mask=attention_mask)

            aspect_class_logits, aspect_self_attention, aspect_cross_attention, spans_embedding = aspect_model(
                bert_features.last_hidden_state, attention_mask, bert_spans_tensor, spans_mask_tensor)

            pred_aspect_logits = torch.argmax(F.softmax(aspect_class_logits, dim=2), dim=2)
            pred_sentiment_ligits = F.softmax(aspect_class_logits, dim=2)
            pred_aspect_logits = torch.where(spans_mask_tensor == 1, pred_aspect_logits,
                                             torch.tensor(0).type_as(pred_aspect_logits))

            if torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0] == 0:
                gold_instances.append(dataset.instances[j])
                gold_aspect_result.append(spans_aspect_tensor)
                gold_aspect_with_sentiment.append(spans_ner_label_tensor)
                gold_opinion_result.append(spans_opinion_label_tensor)
                pred_aspect_result.append(torch.full_like(spans_aspect_tensor, -1))

                pred_aspect_with_sentiment.append(pred_aspect_logits)
                pred_aspect_sentiment_logit.append(pred_sentiment_ligits)

                pred_opinion_result.append(torch.full_like(spans_opinion_label_tensor, -1))
                pred_opinion_sentiment_logit.append(torch.full_like(spans_opinion_label_tensor.unsqueeze(-1).expand(-1,-1,4),-1))

                aspect_mask_list.append(spans_mask_tensor)
            else:
                pred_aspect_spans = torch.chunk(torch.nonzero(pred_aspect_logits, as_tuple=False),
                                                torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0], dim=0)
                pred_span_aspect_tensor = None
                for pred_aspect_span in pred_aspect_spans:
                    batch_num = pred_aspect_span.squeeze()[0]
                    span_aspect_tensor_unspilt_1 = bert_spans_tensor[batch_num, pred_aspect_span.squeeze()[1], :2]
                    span_aspect_tensor_unspilt = torch.tensor(
                        (batch_num, span_aspect_tensor_unspilt_1[0], span_aspect_tensor_unspilt_1[1])).unsqueeze(0)
                    if pred_span_aspect_tensor is None:
                        pred_span_aspect_tensor = span_aspect_tensor_unspilt
                    else:
                        pred_span_aspect_tensor = torch.cat((pred_span_aspect_tensor, span_aspect_tensor_unspilt),dim=0)
                _,all_span_aspect_tensor, all_bert_embedding, all_attention_mask, all_spans_embedding, all_span_mask = opinion_stage_features(
                    bert_features.last_hidden_state, attention_mask, bert_spans_tensor, spans_mask_tensor,
                    spans_embedding, pred_span_aspect_tensor)
                opinion_class_logits, opinion_self_attention, opinion_cross_attention, opinion_attention = opinion_model(
                    all_bert_embedding,
                    all_attention_mask,
                    all_spans_embedding,
                    all_span_mask,
                    all_span_aspect_tensor)
                gold_instances.append(dataset.instances[j])
                gold_aspect_result.append(spans_aspect_tensor)
                gold_aspect_with_sentiment.append(spans_ner_label_tensor)
                gold_opinion_result.append(spans_opinion_label_tensor)
                pred_aspect_result.append(pred_span_aspect_tensor)

                pred_aspect_with_sentiment.append(pred_aspect_logits)
                pred_aspect_sentiment_logit.append(pred_sentiment_ligits)

                pred_opinion_result.append(torch.argmax(F.softmax(opinion_class_logits, dim=2), dim=2))
                pred_opinion_sentiment_logit.append(F.softmax(opinion_class_logits, dim=2))

                aspect_mask_list.append(spans_mask_tensor)
        gold_instances = [x for i in gold_instances for x in i]
        assert len(gold_aspect_result) == len(gold_opinion_result) == len(pred_aspect_result) == len(pred_opinion_result)
        all_gold_aspect, all_gold_aspect_sentiment, all_gold_opinion, all_pred_aspect, all_pred_aspect_sentiment, \
        all_pred_aspect_sentiment_logit, all_pred_opinion, all_pred_opinion_sentiment_logit, all_gold_mask, \
        all_pred_mask = [], [], [], [], [], [], [], [],[],[]
        for i in range(len(gold_aspect_result)):
            gold_aspect_result_tolist = gold_aspect_result[i].tolist()
            gold_aspect_result_sentiment_tolist = gold_aspect_with_sentiment[i].tolist()
            gold_opinion_result_tolist = gold_opinion_result[i].tolist()

            pred_aspect_result_tolist = pred_aspect_result[i].tolist()
            pred_aspect_result_sentiment_tolist = pred_aspect_with_sentiment[i].tolist()
            pred_aspect_sentiment_logit_tolist = pred_aspect_sentiment_logit[i].tolist()

            pred_opinion_result_tolist = pred_opinion_result[i].tolist()
            pred_opinion_sentiment_logit_tolist = pred_opinion_sentiment_logit[i].tolist()

            aspect_mask_tolist = aspect_mask_list[i].tolist()

            # test
            if len(pred_aspect_result_tolist) != len(pred_opinion_result_tolist):
                raise IndexError('预测的aspect和opinion序列数不相等')

            for j in range(len(gold_aspect_result_sentiment_tolist)):
                gold_aspect_per_sent, gold_aspect_sentiment_per_sent, gold_opinion_per_sent, pred_aspect_per_sent, \
                pred_opinion_per_sent, pred_opinion_sentiment_logit_per_sent = [], [], [], [], [], []
                all_gold_aspect_sentiment.append(gold_aspect_result_sentiment_tolist[j])
                all_pred_aspect_sentiment.append(pred_aspect_result_sentiment_tolist[j])
                all_pred_aspect_sentiment_logit.append(pred_aspect_sentiment_logit_tolist[j])

                for k1, span in enumerate(gold_aspect_result_tolist):
                    if span[0] == j:
                        gold_aspect_per_sent.append(span)
                        gold_opinion_per_sent.append(gold_opinion_result_tolist[k1])
                for k2, pred_span in enumerate(pred_aspect_result_tolist):
                    if pred_span[0] == j:
                        pred_aspect_per_sent.append(pred_span)
                        pred_opinion_per_sent.append(pred_opinion_result_tolist[k2])
                        pred_opinion_sentiment_logit_per_sent.append(pred_opinion_sentiment_logit_tolist[k2])
                all_gold_aspect.append(gold_aspect_per_sent)
                all_gold_opinion.append(gold_opinion_per_sent)
                all_pred_aspect.append(pred_aspect_per_sent)

                all_pred_opinion.append(pred_opinion_per_sent)
                all_pred_opinion_sentiment_logit.append(pred_opinion_sentiment_logit_per_sent)

                all_gold_mask.append(aspect_mask_tolist[j])

        metric = Metric(args, all_gold_aspect, all_gold_aspect_sentiment, all_gold_opinion, all_pred_aspect,
                        all_pred_aspect_sentiment, all_pred_aspect_sentiment_logit, all_pred_opinion,
                        all_pred_opinion_sentiment_logit, all_gold_mask, gold_instances)
        aspect_result, opinion_result, apce_result, pair_result, triplet_result = metric.score_triples()
        print('aspect precision:', aspect_result[0], "aspect recall: ", aspect_result[1], "aspect f1: ", aspect_result[2])
        print('opinion precision:', opinion_result[0], "opinion recall: ", opinion_result[1], "opinion f1: ",
              opinion_result[2])
        print('APCE precision:', apce_result[0], "APCE recall: ", apce_result[1], "APCE f1: ",
              apce_result[2])
        print('pair precision:', pair_result[0], "pair recall:", pair_result[1], "pair f1:", pair_result[2])
        print('triple precision:', triplet_result[0], "triple recall: ", triplet_result[1], "triple f1: ", triplet_result[2])
    bert_model.train()
    aspect_model.train()
    opinion_model.train()
    return aspect_result, opinion_result, apce_result, pair_result, triplet_result


def main():
    parser = argparse.ArgumentParser(description="Train scrip")

    parser.add_argument("--dataset", default="lap14", type=str, choices=["lap14", "res14", "res15", "res16"],
                        help="specify the dataset")
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--mode', type=str, default="test", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')
    parser.add_argument("--RANDOM_SEED", type=int, default=41,
                        help="")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="batch size for training")


    '''按照句子的顺序输入排序'''
    parser.add_argument("--order_shuffle", default=True,
                        help="")

    '''随机化输入span排序'''
    # parser.add_argument("--shuffle_data", type=int, default=0,
    #                     help="")
    parser.add_argument("--shuffle_data", type=int, default=0,
                        help="")



    parser.add_argument("--dataset_path", default="./datasets/ASTE-Data-V2-EMNLP2020/", choices=["./datasets/BIO_form/", "./datasets/ASTE-Data-V2-EMNLP2020/"],
                        help="")
    parser.add_argument("--init_model", default="pretrained_models/bert-base-uncased", type=str, required=False,
                        help="Initial model.")
    parser.add_argument("--init_vocab", default="pretrained_models/bert-base-uncased", type=str, required=False,
                        help="Initial vocab.")
    parser.add_argument('--epochs', type=int, default=150,
                        help='training epoch number')
    parser.add_argument("--bert_feature_dim", default=768, type=int,
                        help="feature dim for bert")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=100, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--block_num", type=int, default=2,
                        help="number of block")
    parser.add_argument("--drop_out", type=int, default=0.1,
                        help="")
    parser.add_argument("--max_span_length", type=int, default=8,
                        help="")
    parser.add_argument("--embedding_dim4width", type=int, default=200,
                        help="")
    parser.add_argument("--use_all_bert_features", default=True,
                        help="Whether to use full span information(by combine all spans's bert features)")
    parser.add_argument("--task_learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--task", type=str, default='triples', choices=['senti', 'triples'])
    parser.add_argument("--muti_gpu", default=True)
    parser.add_argument("--output_path", default='triples.json')
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("keyboard break")

