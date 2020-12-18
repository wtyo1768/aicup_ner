from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_bert import BertModel
from typing import List, Optional
import torch.nn as nn
from CRF import CRF


class Bert_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_CRF, self).__init__(config)
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        ):

        outputs =self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
            return outputs 
            
        return self.crf.decode(emissions = logits, mask=attention_mask) 


class Bert_BiLSTM_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_BiLSTM_CRF, self).__init__(config)
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        self.bilstm = nn.LSTM(
            config.hidden_size, 
            (config.hidden_size) // 2, 
            dropout=config.hidden_dropout_prob, 
            batch_first=True, 
            bidirectional=True,
            num_layers=2,
        )
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        ):

        outputs=self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        outputs, hc = self.bilstm(sequence_output)

        logits = self.linear(outputs)

        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
            return outputs 
            
        return self.crf.decode(emissions = logits, mask=attention_mask) 


