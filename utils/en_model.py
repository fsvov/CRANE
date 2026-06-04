import torch
from torch import nn
from transformers import RobertaModel, Data2VecAudioModel
from utils.crane_architecture import BertConfig, CRANEBlock
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WeightedFusion(nn.Module):
    def __init__(self):
        super(WeightedFusion, self).__init__()
        self.text_weight = nn.Parameter(torch.tensor(0.5))
        self.image_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, text_features, image_features):
        fused_features = self.text_weight * text_features + self.image_weight * image_features
        return fused_features


class BiDirectionalGatedMechanism(nn.Module):
    def __init__(self, config, feature_dim):
        super(BiDirectionalGatedMechanism, self).__init__()
        self.feature_dim = feature_dim
        self.audio_gate = nn.Linear(feature_dim, feature_dim)
        self.text_gate = nn.Linear(feature_dim, feature_dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, audio_features, text_features):
        Gate_audio = torch.sigmoid(self.audio_gate(audio_features))
        Gate_audio = self.drop(Gate_audio)
        Gate_text = torch.sigmoid(self.text_gate(text_features))
        Gate_text = self.drop(Gate_text)
        weighted_audio = Gate_text * audio_features
        weighted_text = Gate_audio * text_features
        fused_features = weighted_audio + weighted_text
        return fused_features


class UncertaintyBidirectionalGate(nn.Module):
    """Uncertainty-aware bidirectional gating (UBG).

    Learns per-sample modality confidence scores via lightweight projections
    (feature_dim → 1 each). Confidences gate the features before bi-gating,
    allowing the model to down-weight unreliable modalities per sample.

    Confidence projections are trained end-to-end with the main model.
    Stored confidences (_last_conf_text, _last_conf_audio) are accessible
    after forward for conformal analysis.
    """

    def __init__(self, config, feature_dim):
        super().__init__()
        self.bi_gate = BiDirectionalGatedMechanism(config, feature_dim)
        self.conf_proj_text = nn.Linear(feature_dim, 1)
        self.conf_proj_audio = nn.Linear(feature_dim, 1)
        self._last_conf_text = None
        self._last_conf_audio = None

    def forward(self, audio_features, text_features):
        conf_t = torch.sigmoid(self.conf_proj_text(text_features))
        conf_a = torch.sigmoid(self.conf_proj_audio(audio_features))
        weighted_audio = conf_a * audio_features
        weighted_text = conf_t * text_features
        fused = self.bi_gate(weighted_audio, weighted_text)
        self._last_conf_text = conf_t.detach()
        self._last_conf_audio = conf_a.detach()
        return fused


class CRANEModel(nn.Module):
    """CRANE: Conformal Reliable Augmented Neural Framework.

    Architecture differences from the original baseline:
      - UBG (UncertaintyBidirectionalGate) replaces plain bi-gating
      - Built-in dual head (mean + variance) for uncertainty estimation
    Both additions are inactive during standard training — zero impact on
    point prediction performance.
    """

    def __init__(self, config):
        super().__init__()
        self.fusion_method = config.fusion_method
        self.config = config
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.text_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        Bert_config = BertConfig(num_hidden_layers=config.num_hidden_layer)
        self.CRANEBlock = CRANEBlock(Bert_config)
        self.weightfusion = WeightedFusion().to(device)

        if self.fusion_method == 'v4':
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
            self.selfatt_fusion = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
            self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=False)
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
            self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=False)

        if self.fusion_method == 'v2':
            self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        else:
            self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        self.ubi_gated_model = UncertaintyBidirectionalGate(config, 768).to(device)

        # Dual head: variance head (C) — frozen in standard training,
        # activated via CRANEModelMVE or fine-tuning
        self.var_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        for p in self.var_head.parameters():
            p.requires_grad = False

    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        elif layer_name == 'text_mixed':
            embedding_layer = self.text_mixed_cls_emb
        elif layer_name == 'audio_mixed':
            embedding_layer = self.audio_mixed_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)
        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks

    def _extract_features(self, text_inputs, text_mask, audio_inputs, audio_mask):
        """Shared feature extraction: backbones + CRANE block + fusion."""
        raw_output = self.roberta_model(text_inputs, text_mask)
        T_hidden_states = raw_output.last_hidden_state
        audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            layer = 0
            while layer < 12:
                try:
                    padding_idx = sum(audio_out.attentions[layer][batch][0][0] != 0)
                    audio_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
        audio_mask_new = torch.zeros(A_hidden_states.shape[0], A_hidden_states.shape[1]).to(device)
        for batch in range(audio_mask_new.shape[0]):
            audio_mask_new[batch][:audio_mask_idx_new[batch]] = 1
        text_inputs, text_attn_mask = self.prepend_cls(T_hidden_states, text_mask, 'text')
        audio_inputs, audio_attn_mask = self.prepend_cls(A_hidden_states, audio_mask_new, 'audio')
        text_inputs, audio_inputs = self.CRANEBlock(text_inputs, text_attn_mask, audio_inputs, audio_attn_mask)

        if self.fusion_method == 'v1':
            fused_hidden_states = torch.cat((text_inputs[:, 0, :], audio_inputs[:, 0, :]), dim=1)
        elif self.fusion_method == 'v2':
            fused_hidden_states = self.ubi_gated_model(text_inputs[:, 0, :], audio_inputs[:, 0, :])
        elif self.fusion_method == 'v3':
            fused_hidden_states = self.weightfusion(text_inputs[:, 0, :], audio_inputs[:, 0, :])
        elif self.fusion_method == 'v4':
            combin_features = torch.cat([text_inputs, audio_inputs], dim=1)
            if text_attn_mask is not None and audio_attn_mask is not None:
                combin_mask = torch.cat([text_attn_mask, audio_attn_mask], dim=1)
            else:
                combin_mask = None
            fused_hidden_states = self.selfatt_fusion(combin_features, src_key_padding_mask=combin_mask)
            fused_hidden_states = fused_hidden_states.mean(dim=1)
        return fused_hidden_states

    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask):
        fused_hidden_states = self._extract_features(text_inputs, text_mask,
                                                      audio_inputs, audio_mask)
        return self.fused_output_layers(fused_hidden_states)


class CRANEModelMVE(CRANEModel):
    """CRANE with active MVE head for fine-tuning.

    Reuses the built-in var_head. Backbone is frozen, only var_head trained.
    Outputs (mean, log_var) for Gaussian NLL loss.
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask):
        fused_hidden_states = self._extract_features(text_inputs, text_mask,
                                                      audio_inputs, audio_mask)
        mean = self.fused_output_layers(fused_hidden_states)
        log_var = self.var_head(fused_hidden_states)
        return torch.cat([mean, log_var], dim=1)


def gaussian_nll_loss(pred, target):
    """Gaussian negative log-likelihood loss for MVE training."""
    mean = pred[:, 0:1]
    log_var = pred[:, 1:2]
    var = torch.exp(log_var) + 1e-6
    return torch.mean(0.5 * (torch.log(var) + (target - mean) ** 2 / var))


def freeze_backbone(model):
    """Freeze all parameters except variance head for MVE fine-tuning."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('var_head')


