import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel
from codes.model.moe import SwitchMoE


class MMoEEncoder(nn.Module):
    def __init__(self, args):
        super(MMoEEncoder, self).__init__()
        self.args = args
        current_directory = os.path.dirname(os.path.abspath(__file__))
        base_path = current_directory[0:current_directory.rfind('/')]
        self.base_path = base_path[0:base_path.rfind('/')]
        self.clip = CLIPModel.from_pretrained(self.base_path + self.args.pretrained_model)

        self.image_cls_fc = nn.Linear(self.args.model.input_hidden_dim, self.args.model.dv)
        self.image_tokens_fc = nn.Linear(self.args.model.input_image_hidden_dim, self.args.model.dv)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                pixel_values=None):
        clip_output = self.clip(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values)

        text_embeds = clip_output.text_embeds
        image_embeds = clip_output.image_embeds

        text_seq_tokens = clip_output.text_model_output[0]
        image_patch_tokens = clip_output.vision_model_output[0]

        image_embeds = self.image_cls_fc(image_embeds)
        image_patch_tokens = self.image_tokens_fc(image_patch_tokens)
        return text_embeds, image_embeds, text_seq_tokens, image_patch_tokens

class TextUnit(nn.Module):
    def __init__(self, args):
        super(TextUnit, self).__init__()
        self.args = args
        self.fc_query = nn.Linear(self.args.model.TGLU_hidden_dim, self.args.model.TGLU_hidden_dim)
        self.fc_key = nn.Linear(self.args.model.TGLU_hidden_dim, self.args.model.TGLU_hidden_dim)
        self.fc_value = nn.Linear(self.args.model.TGLU_hidden_dim, self.args.model.TGLU_hidden_dim)
        self.layer_norm = nn.LayerNorm(self.args.model.TGLU_hidden_dim)

        self.moe_layer = SwitchMoE(
            dim=self.args.model.input_hidden_dim,
            output_dim=self.args.model.TGLU_hidden_dim,
            num_experts=self.args.model.num_experts,
            top_k=self.args.model.top_experts
        )

    def forward(self,
                entity_text_cls,
                entity_text_tokens,
                mention_text_cls,
                mention_text_tokens):
        """
        :param entity_text_cls:     [num_entity, dim]
        :param entity_text_tokens:  [num_entity, max_seq_len, dim]
        :param mention_text_cls:    [batch_size, dim]
        :param mention_text_tokens: [batch_size, max_sqe_len, dim]
        :return:
        """
        entity_text_cls_tokens_features = torch.cat([entity_text_cls.unsqueeze(dim=1), entity_text_tokens], dim=1)
        entity_text_moe = self.moe_layer(entity_text_cls_tokens_features, entity_text_cls_tokens_features)
        entity_text_cls_moe = entity_text_moe[:, 0, :]
        entity_cls_fc = entity_text_cls_moe.unsqueeze(dim=1)
        entity_text_tokens = entity_text_moe[:, 1:, :]

        mention_text_cls_tokens_features = torch.cat([mention_text_cls.unsqueeze(dim=1), mention_text_tokens], dim=1)
        mention_text_moe = self.moe_layer(mention_text_cls_tokens_features, mention_text_cls_tokens_features)
        mention_text_tokens = mention_text_moe[:, 1:, :]

        query = self.fc_query(entity_text_tokens)  # [num_entity, max_seq_len, dim]
        key = self.fc_key(mention_text_tokens)  # [batch_size, max_sqe_len, dim]
        value = self.fc_value(mention_text_tokens)  # [batch_size, max_sqe_len, dim]

        query = query.unsqueeze(dim=1)  # [num_entity, 1, max_seq_len, dim]
        key = key.unsqueeze(dim=0)  # [1, batch_size, max_sqe_len, dim]
        value = value.unsqueeze(dim=0)  # [1, batch_size, max_sqe_len, dim]

        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # [num_entity, batach_size, max_seq_len, max_seq_len]

        attention_scores = attention_scores / math.sqrt(self.args.model.TGLU_hidden_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [num_entity, batch_size, max_seq_len, max_seq_len]

        context = torch.matmul(attention_probs, value)  # [num_entity, batch_size, max_seq_len, dim]
        context = torch.mean(context, dim=-2)  # [num_entity, batch_size, dim]
        context = self.layer_norm(context)

        g2l_matching_score = torch.sum(entity_cls_fc * context, dim=-1)  # [num_entity, batch_size]
        g2l_matching_score = g2l_matching_score.transpose(0, 1)  # [batch_size, num_entity]
        g2g_matching_score = torch.matmul(mention_text_cls, entity_text_cls.transpose(-1, -2))

        matching_score = (g2l_matching_score + g2g_matching_score) / 2
        return matching_score


class VisionUnit(nn.Module):
    def __init__(self, args):
        super(VisionUnit, self).__init__()
        self.args = args
        self.fc_query = nn.Linear(self.args.model.dv, self.args.model.IDLU_hidden_dim)
        self.fc_key = nn.Linear(self.args.model.dv, self.args.model.IDLU_hidden_dim)
        self.fc_value = nn.Linear(self.args.model.dv, self.args.model.IDLU_hidden_dim)
        self.layer_norm = nn.LayerNorm(self.args.model.IDLU_hidden_dim)

        self.moe_layer = SwitchMoE(
            dim=self.args.model.dv,
            output_dim=self.args.model.IDLU_hidden_dim,
            num_experts=self.args.model.num_experts,
            top_k=self.args.model.top_experts
        )

    def forward(self,
                entity_image_cls,
                entity_image_tokens,
                mention_image_cls,
                mention_image_tokens):
        """
        :param entity_image_cls:        [num_entity, dim]
        :param entity_image_tokens:     [num_entity, num_patch, dim]
        :param mention_image_cls:       [batch_size, dim]
        :param mention_image_tokens:    [batch_size, num_patch, dim]
        :return:
        """
        entity_image_cls_tokens_features = torch.cat([entity_image_cls.unsqueeze(dim=1), entity_image_tokens], dim=1)
        entity_image_moe = self.moe_layer(entity_image_cls_tokens_features, entity_image_cls_tokens_features)
        entity_image_cls_moe = entity_image_moe[:, 0, :]
        entity_cls_fc = entity_image_cls_moe.unsqueeze(dim=1)
        entity_image_tokens = entity_image_moe[:, 1:, :]

        mention_image_cls_tokens_features = torch.cat([mention_image_cls.unsqueeze(dim=1), mention_image_tokens], dim=1)
        mention_image_moe = self.moe_layer(mention_image_cls_tokens_features, mention_image_cls_tokens_features)
        mention_image_tokens = mention_image_moe[:, 1:, :]

        query = self.fc_query(entity_image_tokens)  # [num_entity, num_patch, dim]
        key = self.fc_key(mention_image_tokens)  # [batch_size, num_patch, dim]
        value = self.fc_value(mention_image_tokens)  # [batch_size, num_patch, dim]

        query = query.unsqueeze(dim=1)  # [num_entity, 1, num_patch, dim]
        key = key.unsqueeze(dim=0)  # [1, batch_size, num_patch, dim]
        value = value.unsqueeze(dim=0)  # [1, batch_size, num_patch, dim]

        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # [num_entity, batch_size, num_patch, num_patch]

        attention_scores = attention_scores / math.sqrt(self.args.model.IDLU_hidden_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [num_entity, batch_size, num_patch, num_patch]

        context = torch.matmul(attention_probs, value)  # [num_entity, batch_size, num_patch, dim]
        context = torch.mean(context, dim=-2)  # [num_entity, batch_size, dim]
        context = self.layer_norm(context)

        g2l_matching_score = torch.sum(entity_cls_fc * context, dim=-1)  # [num_entity, batch_size]
        g2l_matching_score = g2l_matching_score.transpose(0, 1)  # [batch_size, num_entity]
        g2g_matching_score = torch.matmul(mention_image_cls, entity_image_cls.transpose(-1, -2))

        matching_score = (g2l_matching_score + g2g_matching_score) / 2
        return matching_score

class CrossUnit(nn.Module):
    def __init__(self, args):
        super(CrossUnit, self).__init__()
        self.args = args
        self.text_fc = nn.Linear(self.args.model.input_hidden_dim, self.args.model.CMFU_hidden_dim)
        self.image_fc = nn.Linear(self.args.model.dv, self.args.model.CMFU_hidden_dim)
        self.gate_fc = nn.Linear(self.args.model.CMFU_hidden_dim, 1)
        self.gate_act = nn.Tanh()
        self.gate_layer_norm = nn.LayerNorm(self.args.model.CMFU_hidden_dim)
        self.context_layer_norm = nn.LayerNorm(self.args.model.CMFU_hidden_dim)

        self.moe_layer = SwitchMoE(
            dim=self.args.model.CMFU_hidden_dim,
            output_dim=self.args.model.CMFU_hidden_dim,
            num_experts=self.args.model.num_experts,
            top_k=self.args.model.top_experts
        )

    def forward(self,
                entity_text_cls,
                entity_text_tokens,
                mention_text_cls,
                mention_text_tokens,
                entity_image_cls,
                entity_image_tokens,
                mention_image_cls,
                mention_image_tokens):
        """
        :param entity_text_cls:         [num_entity, dim]
        :param entity_image_tokens:     [num_entity, num_patch, dim]
        :param mention_text_cls:        [batch_size, dim]
        :param mention_image_tokens:    [batch_size, num_patch, dim]
        :return:
        """
        entity_text_cls = self.text_fc(entity_text_cls)  # [num_entity, dim]
        entity_text_cls_ori = entity_text_cls
        entity_text_tokens = self.text_fc(entity_text_tokens)  # [num_entity, dim]
        mention_text_cls = self.text_fc(mention_text_cls)  # [num_entity, dim]
        mention_text_cls_ori = mention_text_cls
        mention_text_tokens = self.text_fc(mention_text_tokens)  # [batch_size, dim]

        entity_image_cls = self.image_fc(entity_image_cls)  # [num_entity, num_patch, dim]
        entity_image_cls_ori = entity_image_cls
        entity_image_tokens = self.image_fc(entity_image_tokens)  # [num_entity, num_patch, dim]
        mention_image_cls = self.image_fc(mention_image_cls)  # [batch_size, num_patch, dim]
        mention_image_cls_ori = mention_image_cls
        mention_image_tokens = self.image_fc(mention_image_tokens)  # [batch_size, num_patch, dim]

        entity_text_cls_image_tokens_features = torch.cat([entity_text_cls.unsqueeze(dim=1), entity_image_tokens], dim=1)
        entity_image_cls_text_tokens_features = torch.cat([entity_image_cls.unsqueeze(dim=1), entity_text_tokens], dim=1)
        mention_text_cls_image_tokens_features = torch.cat([mention_text_cls.unsqueeze(dim=1), mention_image_tokens], dim=1)
        mention_image_cls_text_tokens_features = torch.cat([mention_image_cls.unsqueeze(dim=1), mention_text_tokens], dim=1)

        entity_text_cls_image_tokens_moe = self.moe_layer(entity_text_cls_image_tokens_features, entity_image_cls_text_tokens_features)
        entity_text_cls = entity_text_cls_image_tokens_moe[:, 0, :]
        entity_image_tokens = entity_text_cls_image_tokens_moe[:, 1:, :]
        mention_text_cls_image_tokens_moe = self.moe_layer(mention_text_cls_image_tokens_features, mention_image_cls_text_tokens_features)
        mention_text_cls = mention_text_cls_image_tokens_moe[:, 0, :]
        mention_image_tokens = mention_text_cls_image_tokens_moe[:, 1:, :]

        entity_image_cls_text_tokens_moe = self.moe_layer(entity_image_cls_text_tokens_features, entity_text_cls_image_tokens_features)
        entity_image_cls = entity_image_cls_text_tokens_moe[:, 0, :]
        entity_text_tokens = entity_image_cls_text_tokens_moe[:, 1:, :]
        mention_image_cls_text_tokens_moe = self.moe_layer(mention_image_cls_text_tokens_features, mention_text_cls_image_tokens_features)
        mention_image_cls = mention_image_cls_text_tokens_moe[:, 0, :]
        mention_text_tokens = mention_image_cls_text_tokens_moe[:, 1:, :]

        entity_text_cls = entity_text_cls.unsqueeze(dim=1)  # [num_entity, 1, dim]
        entity_text_image_cross_modal_score = torch.matmul(entity_text_cls, entity_image_tokens.transpose(-1, -2))
        entity_text_image_cross_modal_probs = nn.Softmax(dim=-1)(entity_text_image_cross_modal_score)  # [num_entity, 1, num_patch]
        entity_text_image_context = torch.matmul(entity_text_image_cross_modal_probs, entity_image_tokens).squeeze()  # [num_entity, 1, dim]
        entity_text_image_gate_score = self.gate_act(self.gate_fc(entity_text_image_context))
        entity_text_image_context = self.gate_layer_norm((entity_text_cls_ori * entity_text_image_gate_score) + entity_text_image_context)

        mention_text_cls = mention_text_cls.unsqueeze(dim=1)  # [batch_size, 1, dim]
        mention_text_image_cross_modal_score = torch.matmul(mention_text_cls, mention_image_tokens.transpose(-1, -2))
        mention_text_image_cross_modal_probs = nn.Softmax(dim=-1)(mention_text_image_cross_modal_score)
        mention_text_image_context = torch.matmul(mention_text_image_cross_modal_probs, mention_image_tokens).squeeze()
        mention_text_image_gate_score = self.gate_act(self.gate_fc(mention_text_cls_ori))
        mention_text_image_context = self.gate_layer_norm((mention_text_cls_ori * mention_text_image_gate_score) + mention_text_image_context)

        score_text_image = torch.matmul(mention_text_image_context, entity_text_image_context.transpose(-1, -2))

        entity_image_cls = entity_image_cls.unsqueeze(dim=1)  # [num_entity, 1, dim]
        entity_image_text_cross_modal_score = torch.matmul(entity_image_cls, entity_text_tokens.transpose(-1, -2))
        entity_image_text_cross_modal_probs = nn.Softmax(dim=-1)(entity_image_text_cross_modal_score)  # [num_entity, 1, num_patch]
        entity_image_text_context = torch.matmul(entity_image_text_cross_modal_probs, entity_text_tokens).squeeze()  # [num_entity, 1, dim]
        entity_image_text_gate_score = self.gate_act(self.gate_fc(entity_image_text_context))
        entity_image_text_context = self.gate_layer_norm((entity_image_cls_ori * entity_image_text_gate_score) + entity_image_text_context)

        mention_image_cls = mention_image_cls.unsqueeze(dim=1)  # [batch_size, 1, dim]
        mention_image_text_cross_modal_score = torch.matmul(mention_image_cls, mention_text_tokens.transpose(-1, -2))
        mention_image_text_cross_modal_probs = nn.Softmax(dim=-1)(mention_image_text_cross_modal_score)
        mention_image_text_context = torch.matmul(mention_image_text_cross_modal_probs, mention_text_tokens).squeeze()
        mention_image_text_gate_score = self.gate_act(self.gate_fc(mention_image_cls_ori))
        mention_image_text_context = self.gate_layer_norm((mention_image_cls_ori * mention_image_text_gate_score) + mention_image_text_context)

        score_image_text = torch.matmul(mention_image_text_context, entity_image_text_context.transpose(-1, -2))

        score = (score_text_image + score_image_text) / 2

        return score


class MMoEMatcher(nn.Module):
    def __init__(self, args):
        super(MMoEMatcher, self).__init__()
        self.args = args
        self.text_module = TextUnit(self.args)
        self.vision_module = VisionUnit(self.args)
        self.cross_module = CrossUnit(self.args)

        self.text_cls_layernorm = nn.LayerNorm(self.args.model.dt)
        self.text_tokens_layernorm = nn.LayerNorm(self.args.model.dt)
        self.image_cls_layernorm = nn.LayerNorm(self.args.model.dv)
        self.image_tokens_layernorm = nn.LayerNorm(self.args.model.dv)

    def forward(self,
                entity_text_cls, entity_text_tokens,
                mention_text_cls, mention_text_tokens,
                entity_image_cls, entity_image_tokens,
                mention_image_cls, mention_image_tokens):
        """

        :param entity_text_cls:     [num_entity, dim]
        :param entity_text_tokens:  [num_entity, max_seq_len, dim]
        :param mention_text_cls:    [batch_size, dim]
        :param mention_text_tokens: [batch_size, max_sqe_len, dim]
        :param entity_image_cls:    [num_entity, dim]
        :param mention_image_cls:   [batch_size, dim]
        :param entity_image_tokens: [num_entity, num_patch, dim]
        :param mention_image_tokens:[num_entity, num_patch, dim]
        :return:
        """
        entity_text_cls = self.text_cls_layernorm(entity_text_cls)
        mention_text_cls = self.text_cls_layernorm(mention_text_cls)

        entity_text_tokens = self.text_tokens_layernorm(entity_text_tokens)
        mention_text_tokens = self.text_tokens_layernorm(mention_text_tokens)

        entity_image_cls = self.image_cls_layernorm(entity_image_cls)
        mention_image_cls = self.image_cls_layernorm(mention_image_cls)

        entity_image_tokens = self.image_tokens_layernorm(entity_image_tokens)
        mention_image_tokens = self.image_tokens_layernorm(mention_image_tokens)

        text_matching_score = self.text_module(entity_text_cls, entity_text_tokens,
                                        mention_text_cls, mention_text_tokens)
        image_matching_score = self.vision_module(entity_image_cls, entity_image_tokens,
                                         mention_image_cls, mention_image_tokens)
        image_text_matching_score = self.cross_module(entity_text_cls, entity_text_tokens,
                                              mention_text_cls, mention_text_tokens,
                                              entity_image_cls, entity_image_tokens,
                                              mention_image_cls, mention_image_tokens)

        score = (text_matching_score + image_matching_score + image_text_matching_score) / 3
        return score, (text_matching_score, image_matching_score, image_text_matching_score)