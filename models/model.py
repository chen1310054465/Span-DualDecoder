import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput

from .data_BIO_loader import sentiment2id
from allennlp.nn.util import batched_index_select, batched_span_select



def opinion_stage_features(bert_feature,attention_mask,spans,span_mask,spans_embedding,spans_aspect_tensor,spans_opinion_tensor=None):
    # 对输入的aspect信息进行处理，去除掉无效的aspect span
    all_span_aspect_tensor = None
    all_span_opinion_tensor = None
    all_bert_embedding = None
    all_attention_mask = None
    all_spans_embedding = None
    all_span_mask = None
    spans_aspect_tensor_spilt = torch.chunk(spans_aspect_tensor, spans_aspect_tensor.shape[0], dim=0)
    for i, spans_aspect_tensor_unspilt in enumerate(spans_aspect_tensor_spilt):
        test = spans_aspect_tensor_unspilt.squeeze(0)
        batch_num = spans_aspect_tensor_unspilt.squeeze(0)[0]
        # mask4span_start = torch.where(span_mask[batch_num, :] == 1, spans[batch_num, :, 0], torch.tensor(-1).type_as(spans))
        span_index_start = torch.where(spans[batch_num, :, 0] == spans_aspect_tensor_unspilt.squeeze()[1],
                                       spans[batch_num, :, 1], torch.tensor(-1).type_as(spans))
        span_index_end = torch.where(span_index_start == spans_aspect_tensor_unspilt.squeeze()[2], span_index_start,
                                     torch.tensor(-1).type_as(spans))
        span_index = torch.nonzero((span_index_end > -1), as_tuple=False).squeeze(0)
        if min(span_index.shape) == 0:
            continue
        if spans_opinion_tensor is not None:
            spans_opinion_tensor_unspilt = spans_opinion_tensor[i,:].unsqueeze(0)
        aspect_span_embedding_unspilt = spans_embedding[batch_num, span_index, :].unsqueeze(0)
        bert_feature_unspilt = bert_feature[batch_num, :, :].unsqueeze(0)
        attention_mask_unspilt = attention_mask[batch_num, :].unsqueeze(0)
        spans_embedding_unspilt = spans_embedding[batch_num, :, :].unsqueeze(0)
        span_mask_unspilt = span_mask[batch_num, :].unsqueeze(0)
        if all_span_aspect_tensor is None:
            if spans_opinion_tensor is not None:
                all_span_opinion_tensor = spans_opinion_tensor_unspilt
            all_span_aspect_tensor = aspect_span_embedding_unspilt
            all_bert_embedding = bert_feature_unspilt
            all_attention_mask = attention_mask_unspilt
            all_spans_embedding = spans_embedding_unspilt
            all_span_mask = span_mask_unspilt
        else:
            if spans_opinion_tensor is not None:
                all_span_opinion_tensor = torch.cat((all_span_opinion_tensor, spans_opinion_tensor_unspilt), dim=0)
            all_span_aspect_tensor = torch.cat((all_span_aspect_tensor, aspect_span_embedding_unspilt), dim=0)
            all_bert_embedding = torch.cat((all_bert_embedding, bert_feature_unspilt), dim=0)
            all_attention_mask = torch.cat((all_attention_mask, attention_mask_unspilt), dim=0)
            all_spans_embedding = torch.cat((all_spans_embedding, spans_embedding_unspilt), dim=0)
            all_span_mask = torch.cat((all_span_mask, span_mask_unspilt), dim=0)
    return all_span_opinion_tensor,all_span_aspect_tensor,all_bert_embedding,all_attention_mask,all_spans_embedding,all_span_mask


class Aspect_Block(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Aspect_Block, self).__init__()
        self.args = args
        self.aspect_self_attn = BertAttention(bert_config)
        self.aspect_bert_attn = BertAttention(bert_config)
        self.aspect_intermediate = BertIntermediate(bert_config)
        self.aspect_output = BertOutput(bert_config)

    def forward(self, bert_features, attention_mask, spans_embedding, span_masks, span_aspect_tensor=None):
        #注意， mask需要和attention中的scores匹配，用来去掉对应的无意义的值
        #对应的score的维度为 (batch_size, num_heads, feature_dim, features_dim)
        span_attention_masks = span_masks[:, None, None, :]
        span_attention_masks = (1 - span_attention_masks) * -1e9
        bert_attention_mask = attention_mask[:, None, None, :]
        bert_attention_mask = (1 - bert_attention_mask) * -1e9
        self_att_output, self_attention = self.aspect_self_attn(hidden_states=spans_embedding,
                                                             attention_mask=span_attention_masks,
                                                             output_attentions=True)
        cross_attention_output, cross_attention = self.aspect_bert_attn(hidden_states=self_att_output,
                                                                    attention_mask=span_attention_masks,
                                                                    encoder_hidden_states=bert_features,
                                                                    encoder_attention_mask=bert_attention_mask,
                                                                    output_attentions=True)
        intermediate_output = self.aspect_intermediate(cross_attention_output)
        layer_output = self.aspect_output(intermediate_output, cross_attention_output)
        '''测试'''
        # self_attention = None
        return layer_output, self_attention, cross_attention


class Aspect_Extractor(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Aspect_Extractor, self).__init__()
        self.args = args
        self.bert_config = bert_config
        self.dropout_output = torch.nn.Dropout(args.drop_out)
        self.Aspect_Decoder_Layer = Aspect_Block(args, self.bert_config)
        self.decoder = nn.ModuleList([self.Aspect_Decoder_Layer for _ in range(args.block_num)])
        self.aspect2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        self.aspect_embedding4width = nn.Embedding(args.max_span_length+1, args.embedding_dim4width)
        self.aspect_linear4width = nn.Linear(args.embedding_dim4width + args.bert_feature_dim*2, args.bert_feature_dim)

    def forward(self, input_bert_features, attention_mask, spans, span_mask, spans_aspect2opinion_tensor=None, aspect_num=None):
        bert_feature = self.dropout_output(input_bert_features)
        spans_num = spans.shape[1]
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(bert_feature, spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(bert_feature, spans_end)

        if self.args.use_all_bert_features:
            # 如果使用全部span的bert信息：
            spans_width_start_end = spans[:, :, 0:2].view(spans.size(0), spans_num, -1)
            spans_width_start_end_embedding, spans_width_start_end_mask = batched_span_select(bert_feature,
                                                                                              spans_width_start_end)
            spans_width_start_end_mask = spans_width_start_end_mask.unsqueeze(-1).expand(-1, -1, -1,
                                                                                         self.args.bert_feature_dim)
            spans_width_start_end_embedding = torch.where(spans_width_start_end_mask, spans_width_start_end_embedding,
                                                          torch.tensor(0).type_as(spans_width_start_end_embedding))
            spans_width_start_end_mean = spans_width_start_end_embedding.mean(dim=2, keepdim=True).squeeze(-2)
            spans_embedding = spans_width_start_end_mean
        else:
            # 如果使用span区域大小进行embedding
            spans_width = spans[:, :, 2].view(spans.size(0), -1)
            spans_width_embedding = self.aspect_embedding4width(spans_width)
            # spans_embedding = torch.cat((spans_start_embedding, spans_width_embedding, spans_end_embedding), dim=-1)  # 预留可修改部分
            spans_embedding_dict = torch.cat((spans_start_embedding, spans_end_embedding, spans_width_embedding),
                                             dim=-1)
            spans_embedding_dict = self.aspect_linear4width(spans_embedding_dict)
            spans_embedding = spans_embedding_dict

        spans_features = spans_embedding
        for i, Decoder_layer in enumerate(self.decoder):
            layer_output, self_attention, cross_attention = Decoder_layer(bert_feature, attention_mask,
                                                                          spans_features, span_mask)
            spans_features = layer_output
        class_logits = self.aspect2class(spans_features)
        return class_logits, self_attention, cross_attention, spans_embedding


class Opinion_Block(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Opinion_Block, self).__init__()
        self.args = args
        self.dec_self_attn = BertAttention(bert_config)
        self.dec_enc_attn = BertAttention(bert_config)
        self.dec_opinion_attn = BertAttention(bert_config)
        self.intermediate = BertIntermediate(bert_config)
        self.output = BertOutput(bert_config)

    def forward(self, bert_features, attention_mask, spans_embedding, span_masks, span_aspect_tensor):
        #注意， mask需要和attention中的scores匹配，用来去掉对应的无意义的值
        #对应的score的维度为 (batch_size, num_heads, feature_dim, features_dim)
        span_attention_masks = span_masks[:, None, None, :]
        span_attention_masks = (1 - span_attention_masks) * -1e9
        bert_attention_mask = attention_mask[:, None, None, :]
        bert_attention_mask = (1 - bert_attention_mask) * -1e9
        self_att_output, self_attention = self.dec_self_attn(hidden_states=spans_embedding,
                                                             attention_mask=span_attention_masks,
                                                             output_attentions=True)
        cross_attention_output, cross_attention = self.dec_enc_attn(hidden_states=self_att_output,
                                                                    attention_mask=span_attention_masks,
                                                                    encoder_hidden_states=bert_features,
                                                                    encoder_attention_mask=bert_attention_mask,
                                                                    output_attentions=True)
        opinion_attention_output, opinion_attention = self.dec_opinion_attn(hidden_states=cross_attention_output,
                                                                            attention_mask=span_attention_masks,
                                                                            encoder_hidden_states=span_aspect_tensor,
                                                                            encoder_attention_mask=None,
                                                                            output_attentions=True)
        intermediate_output = self.intermediate(opinion_attention_output)
        layer_output = self.output(intermediate_output, opinion_attention_output)
        # '''测试'''
        # self_attention = None
        return layer_output, self_attention, cross_attention, opinion_attention


class Opinion_Extractor(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Opinion_Extractor, self).__init__()
        self.args = args
        self.bert_config = bert_config
        self.dropout_output = torch.nn.Dropout(args.drop_out)
        self.Decoder_layer = Opinion_Block(args, self.bert_config)
        self.decoder = nn.ModuleList([self.Decoder_layer for _ in range(args.block_num)])
        self.docoder2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        self.embedding4width = nn.Embedding(args.max_span_length+1, args.embedding_dim4width)
        self.linear4width = nn.Linear(args.embedding_dim4width + args.bert_feature_dim*2, args.bert_feature_dim)

    def forward(self, input_bert_features, attention_mask, spans_embedding, span_mask, spans_aspect_tensor):

        # 对opinion抽取的所需信息进行处理，使之适合attention操作
        for i, Decoder_layer in enumerate(self.decoder):
            layer_output, self_attention, cross_attention, opinion_attention = Decoder_layer(
                input_bert_features,
                attention_mask,
                spans_embedding,
                span_mask,
                spans_aspect_tensor)
            spans_embedding = layer_output
        class_logits = self.docoder2class(spans_embedding)
        return class_logits, self_attention, cross_attention, opinion_attention


def Loss(args, gold_aspect_label, pred_aspect_label, gold_opinion_label, pred_opinion_label, aspect_spans_mask_tensor, opinion_span_mask_tensor):
    loss_function = nn.CrossEntropyLoss(reduction='sum')

    # aspect预测Loss
    aspect_spans_mask_tensor = aspect_spans_mask_tensor.view(-1) == 1
    pred_aspect_label_logits = pred_aspect_label.view(-1, pred_aspect_label.shape[-1])
    gold_aspect_effective_label = torch.where(aspect_spans_mask_tensor, gold_aspect_label.view(-1),
                                              torch.tensor(loss_function.ignore_index).type_as(gold_aspect_label))
    aspect_loss = loss_function(pred_aspect_label_logits, gold_aspect_effective_label)


    opinion_span_mask_tensor = opinion_span_mask_tensor.view(-1) == 1
    pred_opinion_label_logits = pred_opinion_label.view(-1, pred_opinion_label.shape[-1])
    gold_opinion_effective_label = torch.where(opinion_span_mask_tensor, gold_opinion_label.view(-1),
                                                torch.tensor(loss_function.ignore_index).type_as(gold_opinion_label))
    opinion_loss = loss_function(pred_opinion_label_logits, gold_opinion_effective_label)
    loss = aspect_loss + opinion_loss
    return loss

if __name__ == '__main__':
    tensor1 = torch.zeros((3,3))
    tensor2 = torch.nonzero(tensor1, as_tuple=False)
    tensor1 = tensor1.type_as(tensor2)
    print('666')






