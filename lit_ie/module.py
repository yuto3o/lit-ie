# -*- coding: utf-8 -*-
from typing import List, Optional

import torch
import torch.nn as nn

INF = 1e8


class GlobalPointer(nn.Module):
    """ GlobalPointer
        https://spaces.ac.cn/archives/8373
    """

    def __init__(self, hidden_size, head_hidden_size, num_heads, mask_tril=False, use_rope=True, max_rope_len=512):

        super().__init__()

        self.hidden_size = hidden_size
        self.head_hidden_size = head_hidden_size
        self.num_heads = num_heads

        self.mask_tril = mask_tril
        self.use_rope = use_rope
        self.max_rope_len = max_rope_len

        self.linear = nn.Linear(self.hidden_size, self.num_heads * self.head_hidden_size * 2)

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(self.head_hidden_size, self.max_rope_len)

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.LongTensor = None):
        x = self.linear(hidden_state)
        x = torch.split(x, self.head_hidden_size * 2, dim=-1)  # [B, L, D*2] * K
        x = torch.stack(x, dim=1)  # B, K, L, D*2
        qw, kw = x[..., :self.head_hidden_size], x[..., self.head_hidden_size:]  # B, K, L, D

        if self.use_rope:
            cos, sin = self.rotary_emb(x)
            qw, kw = apply_rotary_pos_emb(qw, kw, cos, sin)

        x = torch.einsum('bkmd,bknd->bkmn', qw, kw) / self.head_hidden_size ** 0.5  # scale

        if attention_mask is not None:
            mask = torch.einsum('bm,bn->bmn', attention_mask, attention_mask)[:, None, :, :]
            x = x * mask - INF * (1 - mask)

        if self.mask_tril:
            mask = torch.tril(torch.ones_like(x), diagonal=-1)
            x = x * (1 - mask) - INF * mask

        return x


class EfficientGlobalPointer(nn.Module):
    """ EfficientGlobalPointer
        https://spaces.ac.cn/archives/8877
    """

    def __init__(self, hidden_size, head_hidden_size, num_heads, mask_tril=False, use_rope=True, max_rope_len=512):

        super().__init__()

        self.hidden_size = hidden_size
        self.head_hidden_size = head_hidden_size
        self.num_heads = num_heads

        self.mask_tril = mask_tril
        self.use_rope = use_rope
        self.max_rope_len = max_rope_len

        self.linear_p = nn.Linear(self.hidden_size, self.head_hidden_size * 2)
        self.linear_q = nn.Linear(self.head_hidden_size * 2, self.num_heads * 2)

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(self.head_hidden_size, self.max_rope_len)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.LongTensor = None):
        x = self.linear_p(hidden)  # B, L, D
        qw, kw = x[..., :self.head_hidden_size], x[..., self.head_hidden_size:]

        if self.use_rope:
            cos, sin = self.rotary_emb(x)
            qw, kw = apply_rotary_pos_emb(qw[:, None, ...], kw[:, None, ...], cos, sin)
            qw, kw = qw.squeeze(1), kw.squeeze(1)

        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_hidden_size ** 0.5
        bias = torch.einsum('bnk->bkn', self.linear_q(x)) / 2.
        x = logits[:, None] + bias[:, :self.num_heads, None] + bias[:, self.num_heads:, :, None]

        if attention_mask is not None:
            mask = torch.einsum('bm,bn->bmn', attention_mask, attention_mask)[:, None, :, :]
            x = x * mask - INF * (1 - mask)

        if self.mask_tril:
            mask = torch.tril(torch.ones_like(x), diagonal=-1)
            x = x * (1 - mask) - INF * mask

        return x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len and seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]

    b, _, seq_len, _ = q.size()
    position_ids = torch.arange(seq_len, device=q.device)[None, :].expand(b, -1)

    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SparseMultiLabelCELossWithLogitsLoss(nn.Module):
    """ Sparse vesion Multi-Label Cross Entropy with Logits
        from https://kexue.fm/archives/7359
    """

    def __init__(self, reduction: str = 'mean', ignore_index: int = -100) -> None:
        super(SparseMultiLabelCELossWithLogitsLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.LongTensor) -> torch.LongTensor:
        """ Sparse Multi-Label Categorical Cross Entropy with Logits
        Args:
            input: Tensor, [B, ..., num_classes]
            target: LongTensor, [B, ..., num_positive_classes]

        Returns:
            torch.LongTensor
        """
        assert target.ndim == input.ndim

        ignore_index_mask = (target == self.ignore_index)
        target[ignore_index_mask] = 0
        # add fake class S_0 as decision boundary
        s_0 = torch.zeros_like(input[..., :1])
        A = input
        P = torch.gather(A, -1, target)

        # pos_loss
        _P = torch.where(ignore_index_mask, torch.tensor(INF).to(P), P)  # mask padding
        pos_loss = torch.logsumexp(-torch.cat([_P, s_0], dim=-1), dim=-1)

        # neg_loss = a - b
        a = torch.logsumexp(torch.cat([A, s_0], dim=-1), dim=-1)
        _P = torch.where(ignore_index_mask, -torch.tensor(INF).to(P), P)  # mask padding
        b = torch.logsumexp(_P, dim=-1)
        b_a = torch.clip(1 - torch.exp(b - a), 1e-10, 1)
        neg_loss = a + torch.log(b_a)

        loss = pos_loss + neg_loss

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise NotImplementedError(self.reduction)


class ConditionalRandomField(nn.Module):
    """ Conditional random field
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: int, Number of tags.
    Attributes:
        start_transitions: torch.nn.Parameter, Start transition score tensor of size
        end_transitions: torch.nn.Parameter, End transition score tensor of size
        transitions: torch.nn.Parameter, Transition score tensor of size
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int) -> None:
        super(ConditionalRandomField, self).__init__()
        self.num_tags = num_tags

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def neg_log_likelihood_loss(self,
                                input: torch.Tensor,
                                target: torch.LongTensor,
                                mask: Optional[torch.LongTensor] = None,
                                reduction: str = 'token_mean') -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given input scores.
        Args:
            input: torch.Tensor, Emission score tensor of size
                (batch_size, seq_length, num_tags)
            target: torch.LongTensor, Sequence of tags tensor of size
                (batch_size, seq_length)
            mask: torch.LongTensor, Mask tensor of size
                (batch_size, seq_length)
        Returns:
            torch.Tensor, The log likelihood. This will have size (batch_size,)
        """

        if mask is None:
            mask = torch.ones_like(target, dtype=torch.long)

        # to simplify the progress
        input = input.transpose(0, 1)
        target = target.transpose(0, 1)
        mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(input, target, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(input, mask)
        # shape: (batch_size,)
        llh = denominator - numerator

        if reduction == 'none':
            return llh
        elif reduction == 'sum':
            return llh.sum()
        elif reduction == 'mean':
            return llh.mean()
        elif reduction == 'token_mean':
            return llh.sum() / mask.float().sum()
        else:
            raise NotImplementedError(reduction)

    def forward(self, input: torch.Tensor, mask: Optional[torch.LongTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            input (`~torch.Tensor`): Emission score tensor of size
                (batch_size, seq_length, num_tags)
            mask (`~torch.LongTensor`): Mask tensor of size
                (batch_size, seq_length)
        Returns:
            List of list containing the best tag sequence for each batch.
        """

        if mask is None:
            mask = input.new_ones(input.shape[:2], dtype=torch.long)

        input = input.transpose(0, 1)
        mask = mask.transpose(0, 1)

        return self._viterbi_decode(input, mask)

    def _compute_score(self, input: torch.Tensor, target: torch.LongTensor, mask: torch.LongTensor) -> torch.Tensor:
        # input: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)

        seq_length, batch_size = target.shape
        mask = mask.type_as(input)

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[target[0]]
        score += input[0, torch.arange(batch_size), target[0]]

        for i in range(1, seq_length):

            # Transition score to next tag, only added if next time_step is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[target[i - 1], target[i]] * mask[i]

            # Emission score for next tag, only added if next time_step is valid (mask == 1)
            # shape: (batch_size,)
            score += input[i, torch.arange(batch_size), target[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = target[seq_ends.long(), torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, input: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        # input: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        seq_length = input.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first time_step has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + input[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_input = input[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_input

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this time_step is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1) == 1, next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, input: torch.FloatTensor, mask: torch.LongTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert input.dim() == 3 and mask.dim() == 2
        assert input.shape[:2] == mask.shape
        assert input.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + input[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_input = input[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_input

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1) == 1, next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
