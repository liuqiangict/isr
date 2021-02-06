
import os
import sys
import logging
import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from .configuration_sparse import SparseConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_bert import BertEmbeddings, BertSelfAttention, BertIntermediate, BertLayerNorm, BertPooler, BertModel, BertPreTrainedModel, gelu
from .modeling_bert import AxialPositionEmbeddings
from .modeling_utils import create_position_ids_from_input_ids

from deepspeed.ops.sparse_attention import DenseSparsityConfig
from deepspeed.ops.sparse_attention import FixedSparsityConfig
from deepspeed.ops.sparse_attention import BigBirdSparsityConfig
from deepspeed.ops.sparse_attention import BSLongformerSparsityConfig
from deepspeed.ops.sparse_attention import VariableSparsityConfig
from deepspeed.ops.sparse_attention import SparseSelfAttention as DeepSpeedSparseSelfAttention 


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

def get_sparse_attention_config(config):
    if (config.mode == 'dense'):
        return DenseSparsityConfig(num_heads=config.num_attention_heads, 
                                        block=config.block,
                                        different_layout_per_head=config.different_layout_per_head)
    elif (config.mode == 'fixed'):
        return FixedSparsityConfig(num_heads=config.num_attention_heads,
                                        block=config.block,
                                        different_layout_per_head=config.different_layout_per_head,
                                        num_local_blocks=config.num_local_blocks,
                                        num_global_blocks=config.num_global_blocks,
                                        attention=config.attention,
                                        horizontal_global_attention=config.different_layout_per_head,
                                        num_different_global_patterns=config.num_different_global_patterns)
    elif (config.mode == 'bigbird'):
        return BigBirdSparsityConfig(num_heads=config.num_attention_heads,
                                        block=config.block,
                                        different_layout_per_head=config.different_layout_per_head,
                                        num_random_blocks=config.num_random_blocks,
                                        num_sliding_window_blocks=config.num_sliding_window_blocks,
                                        num_global_blocks=config.num_global_blocks)
    elif (config.mode == 'bslongformer'):
        return BSLongformerSparsityConfig(num_heads=config.num_attention_heads,
                                        block=config.block,
                                        different_layout_per_head=config.different_layout_per_head,
                                        num_sliding_window_blocks=config.num_sliding_window_blocks)#,
                                        #global_block_indices=config.global_block_indices,
                                        #global_block_end_indices=config.global_block_end_indices)
    elif (config.mode == 'variable'):
        return VariableSparsityConfig(num_heads=config.num_attention_heads,
                                        block=config.block,
                                        different_layout_per_head=config.different_layout_per_head,
                                        num_random_blocks=config.num_random_blocks,
                                        local_window_blocks=config.local_window_blocks,
                                        global_block_indices=config.global_block_indices,
                                        global_block_end_indices=config.global_block_end_indices,
                                        attention=config.attention,
                                        horizontal_global_attention=config.horizontal_global_attention)
    else:
        raise NotImplementedError(f'Given sparsity mode, {config.mode}, has not been implemented yet!')
    return None


class SparseEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = (
            AxialPositionEmbeddings(config) if config.axial_pos_embds else PositionEmbeddings(config)
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SparseSelfAttention(nn.Module):
    def __init__(self, config, sparsity_config):
        super(SparseSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size,
                                config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.sparse_self_attention = DeepSpeedSparseSelfAttention(sparsity_config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        context_layer = self.sparse_self_attention(query_layer,
                                                   key_layer,
                                                   value_layer,
                                                   key_padding_mask=attention_mask)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class SparseSelfOutput(nn.Module):
    def __init__(self, config):
        super(SparseSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SparseAttention(nn.Module):
    def __init__(self, config):
        super(SparseAttention, self).__init__()

        sparse_attention_config = get_sparse_attention_config(config)
        self.self = SparseSelfAttention(config, sparsity_config=sparse_attention_config)

        self.output = SparseSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_output = self.self(
            hidden_states, attention_mask
        )
        attention_output = self.output(self_output, hidden_states)

        return (attention_output,)


class SparseIntermediate(nn.Module):
    def __init__(self, config):
        super(SparseIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SparseOutput(nn.Module):
    def __init__(self, config):
        super(SparseOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SparseLayer(nn.Module):
    def __init__(self, config):
        super(SparseLayer, self).__init__()
        self.attention = SparseAttention(config)
        self.intermediate = SparseIntermediate(config)
        self.output = SparseOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs
        return outputs


class SparseEncoder(nn.Module):
    def __init__(self, config):
        super(SparseEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [SparseLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)

        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class SparseModel(BertPreTrainedModel):
    base_model_prefix = "sparse_transformer"
    def __init__(self, config):
        super(SparseModel, self).__init__(config)
        self.config = config

        self.embeddings = SparseEmbeddings(config)
        self.encoder = SparseEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]
        return outputs


class SparseForTokenClassification(BertPreTrainedModel):
    config_class = SparseConfig
    base_model_prefix = "sparse"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.sparse = SparseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
    ):
        outputs = self.sparse(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits
