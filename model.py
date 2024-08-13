import torch
import torch.nn as nn
import math

class DebertaEmbeddings(nn.Module):
  def __init__(self, config):
    super(DebertaEmbeddings, self).__init__()

    self.vocab_size = config.vocab_size
    self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)

    self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size,
                                        padding_idx=config.pad_token_id)

    self.type_vocab_size = config.type_vocab_size

    if self.type_vocab_size > 0:
      self.token_type_embeddings = nn.Embedding(self.type_vocab_size,
                                                self.embedding_size)

    self.LayerNorm = nn.LayerNorm(self.embedding_size, config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, input_ids, token_type_ids=None, mask = None):
    embeddings = self.word_embeddings(input_ids)

    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    if self.type_vocab_size > 0:
      token_type_embeddings = self.token_type_embeddings(token_type_ids)
      embeddings += token_type_embeddings

    mask = mask.unsqueeze(-1)

    embeddings = embeddings * mask
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)

    return embeddings

class DebertaSelfOutput(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.hidden_size = config.hidden_size
    self.layer_norm_eps = config.layer_norm_eps
    self.hidden_dropout_prob = config.hidden_dropout_prob

    self.dense = nn.Linear(self.hidden_size, self.hidden_size)
    self.LayerNorm = nn.LayerNorm(self.hidden_size, self.layer_norm_eps)
    self.dropout = nn.Dropout(self.hidden_dropout_prob)

  def forward(self, hidden_states, input_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states += input_states
    hidden_states = self.LayerNorm(hidden_states)

    return hidden_states

class DebertaIntermediate(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.intermediate_act_fn = nn.GELU()

  def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)

    return hidden_states

class DebertaOutput(nn.Module):
  def __init__(self, config):
    super(DebertaOutput, self).__init__()

    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.hidden_dropout_prob = config.hidden_dropout_prob
    self.layer_norm_eps = config.layer_norm_eps

    self.dense = nn.Linear(self.intermediate_size, self.hidden_size)
    self.LayerNorm = nn.LayerNorm(self.hidden_size, self.layer_norm_eps)
    self.dropout = nn.Dropout(self.hidden_dropout_prob)

  def forward(self, hidden_states, input_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states += input_states
    hidden_states = self.LayerNorm(hidden_states)

    return hidden_states

class DisentangledSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.hidden_size = config.hidden_size
    self.num_attention_heads = config.num_attention_heads
    self.head_size = int(self.hidden_size/self.num_attention_heads)

    self.query_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
    self.key_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.value_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    self.pos_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.pos_q_proj = nn.Linear(self.hidden_size, self.hidden_size)

    self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def reshape_states(self, x, size):

    return x.view(*x.shape[:-1], self.num_attention_heads, size).transpose(-2, -3)

  def forward(self, hidden_states, attention_mask, relative_pos, rel_embeddings, query_states=None):

    """
    param hidden_states: exit from the previous layer, float32[batch_size, seq_length, hidden_size]
    param attention_mask: bool[batch_size, 1, seq_length, seq_length]
    param relative_pos: relative distance matrix, int64[seq_length, seq_length]
    param rel_embeddings: relative position embedding vectors, float32[2*position_buckets, hidden_size]
    param query_states: information for decoding (e.g. hidden states, absolute position embedding
    or output from the previous ENHANCED MASK DECODER layer), float32[batch_size, seq_length, hidden_size]
    """

    if query_states is None:
      query_states = hidden_states

    query = self.query_proj(query_states) #[batch_size, num_heads, seq_len, head_size]
    key = self.key_proj(hidden_states)
    value =  self.value_proj(hidden_states)

    query, key, value =  self.reshape_states(torch.cat((query, key, value), dim=-1), 3*self.head_size).split(self.head_size, dim=-1)

    rel_embeddings = self.pos_dropout(rel_embeddings)
    pos_query = self.reshape_states(self.pos_q_proj(rel_embeddings), self.head_size)  #[num_heads, 2*position_buckets, head_size]
    pos_key = self.reshape_states(self.pos_proj(rel_embeddings), self.head_size)

    #context to context score
    c2c_score = query @ key.transpose(-1, -2)  #[batch_size, num_heads, seq_len, seq_len]

    #context to position score
    c2p_score = query @ pos_key.transpose(-1, -2)  #[batch_size, num_heads, seq_len, 2*position_buckets]
    c2p_pos = relative_pos.expand(*c2p_score.size()[:2], -1, -1)
    c2p_score = torch.gather(c2p_score, dim=-1, index = c2p_pos)

    #position to context score
    p2c_score = key @ pos_query.transpose(-1, -2)
    p2c_score = torch.gather(p2c_score, dim=-1, index = c2p_pos.transpose(-1, -2))
    p2c_score = p2c_score.transpose(-1, -2)

    #accamulate attention scores
    scale_factor = 1/math.sqrt(query.size(-1) * 3)
    score = (c2c_score + c2p_score + p2c_score) * scale_factor

    mask = ~(attention_mask)  #we do not pay attention to invalid tokens
    score = score.masked_fill(mask, float('-inf'))
    score = torch.softmax(score, dim = -1)
    score = score.masked_fill(mask, 0)
    score = self.dropout(score)
    score = score @ value
    score = score.transpose(1, 2)
    score_shape = score.shape

    hidden_states = score.reshape(*score_shape[:-2], score_shape[-2] * score_shape[-1])

    return hidden_states

class DebertaAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.self = DisentangledSelfAttention(config)
    self.output = DebertaSelfOutput(config)

  def forward(self, hidden_states, attention_mask, relative_pos, rel_embeddings, query_states=None):

    input_states = hidden_states

    hidden_states = self.self(hidden_states, attention_mask, relative_pos, rel_embeddings, query_states)

    attention_output = self.output(hidden_states, input_states)

    return attention_output

class DebertaLayer(nn.Module):
  def __init__(self, config):
    super(DebertaLayer, self).__init__()

    self.attention = DebertaAttention(config)
    self.intermediate = DebertaIntermediate(config)
    self.output = DebertaOutput(config)

  def forward(self, hidden_states, attention_mask, relative_pos, rel_embeddings, query_states=None):

    attention_output = self.attention(hidden_states, attention_mask, relative_pos, rel_embeddings, query_states)
    intermediate_output = self.intermediate(attention_output)
    output = self.output(intermediate_output, attention_output)

    return output

class DebertaEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.layer = nn.ModuleList([DebertaLayer(config) for _ in range(config.num_hidden_layers)])
    self.pos_ebd_size = config.max_position_embeddings
    self.hidden_size = config.hidden_size

    self.rel_embeddings = nn.Embedding(self.pos_ebd_size * 2, self.hidden_size)

  def get_attention_mask(self, attention_mask):
    ext_attention_mask = attention_mask.unsqueeze(-1).to(torch.float)
    ext_attention_mask = ext_attention_mask @ ext_attention_mask.transpose(-1, -2)
    return ext_attention_mask.unsqueeze(1).to(torch.bool)

  def create_distance_matrix(self, seq_len: int, max_distance: int):
    indices = torch.arange(seq_len)
    indices_t = indices.unsqueeze(-1)
    diff_indcs = indices_t - indices
    dis_mat = diff_indcs
    k = max_distance

    dis_mat[diff_indcs >= k] = 2 * k - 1
    dis_mat[torch.logical_and(-k < diff_indcs, diff_indcs < k)] += k
    dis_mat[diff_indcs <= -k] = 0

    return dis_mat.to(torch.int64)

  def forward(self, hidden_states, attention_mask, query_states = None, relative_pos=None):

    rel_embeddings = self.rel_embeddings.weight
    attention_mask = self.get_attention_mask(attention_mask)
    relative_pos = self.create_distance_matrix(hidden_states.shape[1],
                                               self.pos_ebd_size).to(hidden_states.device)

    for layer in self.layer:
      output_states = layer(hidden_states, attention_mask, relative_pos, rel_embeddings, query_states)
      hidden_states = output_states
      if query_states is not None:
        query_states = output_states

    return hidden_states

class DebertaModel(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.embeddings = DebertaEmbeddings(config)
    self.encoder = DebertaEncoder(config)

  def forward(self, input_ids, attention_mask, token_type_ids = None):

    embeddings = self.embeddings(input_ids, token_type_ids, attention_mask)
    encoder_output = self.encoder(embeddings, attention_mask)

    return encoder_output

class LoRALayer(nn.Module):
    def __init__(self, module: nn.Linear, rank):
      super().__init__()

      self.module = module

      self.adapter_A = nn.Parameter(torch.rand(module.in_features, rank, device=module.weight.device))
      self.adapter_B = nn.Parameter(torch.zeros(rank, module.out_features, device=module.weight.device))

    def forward(self, input):
      return self.module(input) + (input @ self.adapter_A) @ self.adapter_B

class DebertaQA(nn.Module):
  def __init__(self, deberta_config):
    super().__init__()

    self.deberta = DebertaModel(deberta_config)

    hidden_size = deberta_config.hidden_size
    lora_rank = 32

    for layer_name, module in self.deberta.named_modules():
      if 'DebertaAttention' in repr(type(module)):
        module.self.query_proj = LoRALayer(module.self.query_proj, lora_rank)
        module.self.key_proj = LoRALayer(module.self.key_proj, lora_rank)
        module.self.value_proj = LoRALayer(module.self.value_proj, lora_rank)

    assert sum(isinstance(module, LoRALayer) for module in self.deberta.modules()) == 12 * 3

    self.gelu = nn.GELU()

    self.linear1 = nn.Linear(hidden_size, 2)

  def forward(self, input_ids, token_type_ids, attention_mask):
    res = self.deberta(input_ids = input_ids,
                       token_type_ids = token_type_ids,
                       attention_mask = attention_mask)

    res = self.gelu(res)
    res = self.linear1(res)

    start, end = torch.split(res, 1, dim = -1)

    return start.squeeze(-1), end.squeeze(-1)