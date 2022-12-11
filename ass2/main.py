# Import libraries used throughout - if you need other libraries,
# you are free to import them.
import functools
import random
from typing import Any

import numpy as np
import torch
import torchtext
import torchtext.functional as F
import torchtext.transforms
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import UDPOS
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertModel, BertTokenizer
import pickle

# Constants and hyperparameters - you are free to change these for
# the bonus question.
SEED = 42
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
TRANSFORMER = "bert-base-uncased"

# Reproducibility.
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)

# Setting up dataloaders for training.
tokenizer = BertTokenizer.from_pretrained(TRANSFORMER)
init_token = tokenizer.cls_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token
sep_token = tokenizer.sep_token
init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)
sep_token_idx = tokenizer.convert_tokens_to_ids(sep_token)
max_input_length = tokenizer.max_model_input_sizes[TRANSFORMER]

train_datapipe = UDPOS(split="train")
valid_datapipe = UDPOS(split="valid")
pos_vocab = build_vocab_from_iterator(
    [i[1] for i in list(train_datapipe)],
    specials=[init_token, pad_token, sep_token],
)


def prepare_words(tokens, tokenizer, max_input_length, init_token, sep_token):
    """Preprocesses words such that they may be passed into BERT.

    Parameters
    ---
    tokens : List
        List of strings, each of which corresponds to one token in a sequence.
    tokenizer : transformers.models.bert.tokenization_bert.BertTokenizer
        Tokenizer to be used for transforming word strings into word indices
        to be used with BERT.
    max_input_length : int
        Maximum input length of each sequence as expected by our version of BERT.
    init_token : str
        String representation of the beginning of sentence marker for our tokenizer.
    sep_token : str
        String representation of the end of sentence marker for our tokenizer.

    Returns
    ---
    tokens : List
        List of preprocessed tokens.
    """
    # Append beginning of sentence and end of sentence markers
    # lowercase each token and cut them to the maximum length
    # (minus two to account for beginning and end of sentence).
    tokens = [init_token] + [i.lower() for i in tokens[: max_input_length - 2]] + [sep_token]
    # Convert word strings to indices.
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return tokens


def prepare_tags(tokens, max_input_length, init_token, sep_token):
    """Convert tag strings into indices for use with torch. For symmetry, we perform
        identical preprocessing as on our words, even though we do not need beginning
        of sentence and end of sentence markers for our tags.

    Parameters
    ---
    tokens : List
        List of strings, each of which corresponds to one token in a sequence.
    max_input_length : int
        Maximum input length of each sequence as expected by our version of BERT.
    init_token : str
        String representation of the beginning of sentence marker for our tokenizer.
    sep_token : str
        String representation of the end of sentence marker for our tokenizer.

    Returns
    ---
    tokens : List
        List of preprocessed tags.
    """
    # Append beginning of sentence and end of sentence markers
    # cut the tagging sequence to the maximum length (minus two to account for beginning and end of sentence).
    tokens = [init_token] + tokens[: max_input_length - 2] + [sep_token]
    # Convert tag strings to indices.
    tokens = torchtext.transforms.VocabTransform(pos_vocab)(tokens)
    return tokens


text_preprocessor = functools.partial(
    prepare_words,
    tokenizer=tokenizer,
    max_input_length=max_input_length,
    init_token=init_token,
    sep_token=sep_token,
)

tag_preprocessor = functools.partial(
    prepare_tags,
    max_input_length=max_input_length,
    init_token=init_token,
    sep_token=sep_token,
)


def apply_transform(x):
    return text_preprocessor(x[0]), tag_preprocessor(x[1])


train_datapipe = (
    train_datapipe.map(apply_transform).batch(BATCH_SIZE).rows2columnar(["words", "pos"])
)
train_dataloader = DataLoader(train_datapipe, batch_size=None, shuffle=False)
valid_datapipe = (
    valid_datapipe.map(apply_transform).batch(BATCH_SIZE).rows2columnar(["words", "pos"])
)
valid_dataloader = DataLoader(valid_datapipe, batch_size=None, shuffle=False)

from heapq import heappush, heappop, heapify
import itertools


class PriorityQueue:
    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = "<removed-task>"  # placeholder for a removed task
        self.counter = itertools.count()
        self.length = 0  # unique sequence count

    def __len__(self):
        return self.length

    def push(self, entry):
        "Add a new task or update the priority of an existing task"
        priority = entry[0] * -1
        task = entry[1]
        self.length += 1
        if task in self.entry_finder:
            priority = min(self.remove(task), priority)
        count = next(self.counter)
        new_entry = [priority, count, task]
        self.entry_finder[task] = new_entry
        heappush(self.pq, new_entry)

    def remove(self, task):
        self.length -= 1
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED
        return entry[0]

    def pop(self):
        self.length -= 1
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return (-1 * priority, task)
        raise KeyError("pop from an empty priority queue")


class TagLSTM(nn.Module):
    """Models an LSTM on top of a transformer to predict POS in a Neural CRF."""

    def __init__(self, nb_labels, emb_dim, hidden_dim=256):
        """Constructor.

        Parameters
        ---
        nb_labels : int
            Number of POS tags to be considered.

        emb_dim : int
            Input_size of the LSTM - effectively embedding dimension of our pretrained transformer.

        hidden_dim : int
            Hidden dimension of the LSTM.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.tag = nn.Linear(hidden_dim, nb_labels)
        self.hidden = None

    def init_hidden(self, batch_size):
        return (
            torch.randn(2, batch_size, self.hidden_dim // 2),
            torch.randn(2, batch_size, self.hidden_dim // 2),
        )

    def forward(self, x):
        self.hidden = self.init_hidden(x.shape[0])
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.tag(x)
        return x


class NeuralCRF(nn.Module):
    """Class modeling a neural CRF for POS tagging.
    We model tag-tag dependencies with a weight for each transition
    and word-tag influence through an LSTM on top of a pretrained transformer.
    """

    def __init__(
        self,
        pad_idx_word,
        pad_idx_pos,
        bos_idx,
        eos_idx,
        bot_idx,
        eot_idx,
        t_cal,
        transformer,
        lstm_hidden_dim=64,
        beta=0,
    ):
        """Constructor.

        Parameters
        ---
        pad_idx_word : int
            Index corresponding to padding in the word sequences.
        pad_idx_pos : int
            Index corresponding to padding in the tag sequences.
        bos_idx : int
            Index corresponding to beginning of speech marker in the word sequences.
        eos_idx : int
            Index corresponding to end of speech marker in the word sequences.
        bot_idx : int
            Index corresponding to beginning of tag marker in the tag sequences.
        eot_idx : int
            Index corresponding to end of tag marker in the tag sequences.
        t_cal : List[int]
            List containing all indices corresponding to tags in the tag sequences.
        transformer : BertModel
            Pretrained transformer used to embed sentences before feeding them
            into the LSTM.
        lstm_hiden_dim : int
            Hidden dimension of the LSTM used for POS tagging. Note that
            since we are bidirectional, the effective hidden dimension
            is half of this number.
        beta : float
            Regularization hyperparameter of the entropy regularizer.
            Entropy regularization is only applied for \beta > 0.
        """
        super().__init__()
        self.pad_idx_word = pad_idx_word
        self.pad_idx_pos = pad_idx_pos
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.bot_idx = bot_idx
        self.eot_idx = eot_idx
        self.t_cal = t_cal
        self.transformer = transformer
        self.lstm_hidden_dim = lstm_hidden_dim
        self.beta = beta
        self.transitions = nn.Parameter(torch.empty(len(t_cal), len(t_cal)))
        self.emissions = TagLSTM(
            len(t_cal),
            transformer.config.to_dict()["hidden_size"],
            lstm_hidden_dim,
        )
        self.init_weights()
        # self.device = "cpu"

    def init_weights(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, W):
        """Decode each sentence within W and return predicted tagging.

        Parameters
        ---
        W : torch.tensor
            Word sequences of dimension batch size x max sentence length within batch + 2.

        Returns
        ---
        sequences : list
            List of tensors, each of which contains the predicted tag indices for a particular
            word sequence.
        """
        # Calculate scores.
        emissions = self.calculate_emissions(W)
        # Run viterbi sentence by sentence.
        sequences = []
        for sentence in range(W.shape[0]):
            # Exclude beginning and end markers from each word sequence.
            scores, backpointers = self.backward_viterbi_log(
                W[sentence, 1:], emissions[sentence, :]
            )
            sequences += [self.get_viterbi(backpointers)]
        return sequences

    def calculate_emissions(self, W):
        """Calculate emissions (i.e., scores for each word and batch).

        Parameters
        ---
        W : torch.tensor
            Word sequences of dimension batch size x max sentence
            length within batch + 2.

        Returns
        ---
        emissions : torch.tensor
            Word level scores for each tag of dimension batch_size x max
            sentence length within batch + 1 x |T|.
            The scores for the initial BOS index are already removed here
            since we only needed it for the transformer.
        """
        # Directly exclude emissions for the initial word in each sentence
        # since these correspond to the BOS indices that we only need
        # for BERT.
        return self.emissions(self.transformer(W)[0])[:, 1:, :]

    def loss(self, T, W):
        """Calculate the loss for a batch.

        Parameters
        ---
        T : torch.tensor
            True taggings for each sequence within the batch.
            Of dimension batch size x longest sequence within batch + 2.
            Note the paddings, EOS and BOS that have been added to T
            for symmetry with W which needs this for BERT.
        W : torch.tensor
            Words for each sequence within the batch.
            Of dimension batch size x longest sequence within batch + 2.
            Note the paddings, EOS and BOS that have been added to W
            for usage with BERT.

        Returns
        ---
        torch.tensor
            Mean loss for the batch.
        """
        emissions = self.calculate_emissions(W)
        # Note that we have to handle paddings and EOS within the score
        # and backward functions, but we can already skip the BOS tokens
        # here.
        scores = self.score(emissions, W[:, 1:], T[:, 1:])
        log_normalizer = self.backward_log_Z(W[:, 1:], emissions)
        loss = torch.negative(torch.mean(scores - log_normalizer))
        if self.beta > 0.0:
            unnormalized_entropy = self.backward_entropy(W[:, 1:], emissions)
            entropy = (unnormalized_entropy / torch.exp(log_normalizer)) + log_normalizer
            return loss + torch.negative(self.beta * torch.mean(entropy))
        else:
            return loss

    def score(self, emissions, W, T):
        """Calculate scores for specified taggings and word sequences.

        Parameters
        ---
        emissions : torch.tensor
        T : torch.tensor
            Taggings for each sequence within the batch.
            Of dimension batch size x longest sequence within batch + 1.
            Note the paddings, EOS and BOS that have been added to T
            for symmetry with W which needs this for BERT.
            We expect T to already have the initial BOT tag indices removed
            (see `loss` for details).
        W : torch.tensor
            Words for each sequence within the batch.
            Of dimension batch size x longest sequence within batch + 1.
            Note the paddings, EOS and BOS that have been added to W
            for usage with BERT so we mask them out here. We expect
            W to already have the initial BOS word indices taken out
            (see `loss` for details).

        Returns
        ---
        scores : torch.tensor
            score(T, W) for all W[idx]s in W.
        """
        scores = (
            emissions[:, 0].gather(1, (T[:, 0]).unsqueeze(1)).squeeze()
            + self.transitions[self.bot_idx, T[:, 0]]
        )
        for word in range(1, emissions.shape[1]):
            mask = torch.where(W[:, word] == self.pad_idx_word, 0, 1) * torch.where(
                W[:, word] == self.eos_idx, 0, 1
            )
            scores += mask * (
                emissions[:, word].gather(1, (T[:, word]).unsqueeze(1)).squeeze()
                + self.transitions[T[:, word - 1], T[:, word]]
            )
        return scores

    def viterbi_naive(self, W, emissions):
        """Calculate best tagging naively and return both the best score and best tagging in log space.

        NB: This naive version is not vectorized over W[idx]s.

        Parameters
        ---
        W : torch.tensor
            Of dimension longest sequence within batch + 2 or less.
            Note the paddings, EOS and BOS that have been added to W
            for usage with BERT so we manually remove them here if present.
        emissions : torch.tensor
            Word level scores for each tag of dimension max
            sentence length within batch + 1 x |T| (we assume scores for the BOT
            initial tag have already been removed since BOT/BOS is
            only needed for the transformer).

        Returns
        ---
        Tuple[torch.tensor, torch.tensor]
            Tuple containing the log-score of the best tagging and the
            indices of the best tagging for W.
        """
        T = self.t_cal
        # Remove padding.
        if torch.any(W == self.pad_idx_word):
            W = W[torch.where(W != self.pad_idx_word)[0]]
        # Remove EOS and BOS if present.
        if torch.any(W == self.eos_idx):
            W = W[:-1]
        if torch.any(W == self.bos_idx):
            W = W[1:]
        T_abs = len(T)
        combinations = torch.combinations(T, r=W.shape[0], with_replacement=True)
        combinations = torch.cartesian_prod(*[T for ix in range(W.shape[0])])
        best_score = torch.tensor(0.0, dtype=torch.float64)
        best_tag = torch.tensor([])
        for ix, combination in enumerate(combinations):
            if W.shape[0] == 1:
                current_score = (
                    emissions[0, combination] + self.transitions[self.bot_idx, combination]
                )
            else:
                current_score = (
                    emissions[0, combination[0]] + self.transitions[self.bot_idx, combination[0]]
                )
                for qx in range(1, combination.shape[0]):
                    current_score += (
                        emissions[qx, combination[qx]]
                        + self.transitions[combination[qx - 1], combination[qx]]
                    )

            if (current_score) > best_score:
                best_score = current_score.double()
                best_tag = combination
        return best_score, best_tag

    def log_Z_naive(self, W, emissions):
        """Calculate log Z naively.

        NB: This naive version is not vectorized over W[idx]s.

        Parameters
        ---
        W : torch.tensor
            Of dimension longest sequence within batch + 2 or less.
            Note the paddings, EOS and BOS that have been added to W
            for usage with BERT so we manually remove them here if present.
        emissions : torch.tensor
            Word level scores for each tag of dimension max
            sentence length within batch + 1 x |T| (we assume scores for the BOT
            initial tag have already been removed since BOT/BOS is
            only needed for the transformer).

        Returns
        ---
        torch.tensor
            Log Z for W.
        """
        T = self.t_cal
        # Remove padding
        W = W[torch.where(W != self.pad_idx_word)[0]]
        # Remove EOS and BOS if present
        if torch.any(W == self.eos_idx):
            W = W[:-1]
        if torch.any(W == self.bos_idx):
            W = W[1:]
        T_abs = len(T)

        # Generate \mathcal{T}^N.
        combinations = torch.cartesian_prod(*[T for ix in range(W.shape[0])])
        log_normalizer = torch.zeros(combinations.shape[0], dtype=torch.float64)
        # Loop over all possible combinations naively.
        # NB: This is essentially line one on Slide 50.
        for ix, combination in enumerate(combinations):
            # Kludge since indexing is slightly different for one-dim
            # tensors vs two tensors.
            if W.shape[0] == 1:
                # Calculate score as the sum of emissions (i.e., how well
                # does a word match a tag based on BERT embeddings) and
                # transitions (globally, how likely is a transition
                # from the previous tag to the current tag).
                # NB: For the first word, the initial tag is always BOT.
                # NB 2: Since we are in log-space, the exp
                # of the score goes away.
                log_normalizer[ix] = (
                    emissions[0, combination] + self.transitions[self.bot_idx, combination]
                )

            else:
                # Initial score is identical to above.
                log_normalizer[ix] = (
                    emissions[0, combination[0]] + self.transitions[self.bot_idx, combination[0]]
                )
                for qx in range(1, combination.shape[0]):
                    # Score within each potential tagging
                    # is calculated the same as above except that we now
                    # actually use the previous tag instead of always
                    # BOT.
                    log_normalizer[ix] += (
                        emissions[qx, combination[qx]]
                        + self.transitions[combination[qx - 1], combination[qx]]
                    )
        # Calculate logsumexp numerically stable
        # since we are in log-space.
        return torch.logsumexp(log_normalizer, 0)

    def entropy_naive(self, W, emissions):
        """Calculate the unnormalized entropy naively.

        NB: This naive version is not vectorized over W[idx]s.

        Parameters
        ---
        W : torch.tensor
            Words for each sequence within the batch.
            Of dimension longest sequence within batch + 2 or less.
            Note the paddings, EOS and BOS that have been added to W
            for usage with BERT so we manually remove them here if present.
        emissions : torch.tensor
            Word level scores for each tag of dimension max
            sentence length within batch + 1 x |T| (we assume scores for the BOT
            initial tag have already been removed since BOT/BOS is
            only needed for the transformer).

        Returns
        ---
        torch.tensor
            Log Z for W.
        """
        T = self.t_cal
        # Remove padding
        W = W[torch.where(W != self.pad_idx_word)[0]]
        # Remove EOS and BOS if present
        if torch.any(W == self.eos_idx):
            W = W[:-1]
        if torch.any(W == self.bos_idx):
            W = W[1:]
        T_abs = len(T)
        combinations = torch.combinations(T, r=W.shape[0], with_replacement=True)
        combinations = torch.cartesian_prod(T, T)
        combinations = torch.cartesian_prod(*[T for ix in range(W.shape[0])])
        entropy = torch.zeros(1, dtype=torch.float64)
        for ix, combination in enumerate(combinations):
            if W.shape[0] == 1:
                entropy -= torch.exp(
                    (emissions[0, combination] + self.transitions[self.bot_idx, combination])
                ) * (emissions[0, combination] + self.transitions[self.bot_idx, combination])

            else:
                local_score = (
                    emissions[0, combination[0]] + self.transitions[self.bot_idx, combination[0]]
                )
                for qx in range(1, combination.shape[0]):
                    local_score += (
                        emissions[qx, combination[qx]]
                        + self.transitions[combination[qx - 1], combination[qx]]
                    )
                entropy -= torch.exp(local_score) * local_score
        return entropy

    def get_viterbi(self, backpointer_matrix):
        """Return the best tagging based on a backpointer matrix.

        Parameters
        ---
        backpointer_matrix : torch.tensor
            Backpointer matrix from Viterbi indicating which
            tag is the highest scoring for each element in the sequence.

        Returns
        ---
        torch.tensor
            Indices of the best tagging based on `backpointer_matrix`.
        """
        N = backpointer_matrix.shape[0]
        tagging = torch.zeros(N, dtype=torch.int)

        next_tag = 0
        for i in range(0, N):
            next_tag = backpointer_matrix[i, next_tag]
            tagging[i] = next_tag

        return tagging

    def backward_log_Z(self, W, emissions):
        """Calculate log Z using the backward algorithm.

        NB: You do need to vectorize this over W[idx]s.

        Parameters
        ---
        W : torch.tensor
            Words for each sequence within the batch.
            Of dimension batch size x longest sequence within batch + 1.
            Note the paddings, EOS and BOS that have been added to W
            for usage with BERT so we mask them out here. We expect
            W to already have the initial BOS word indices taken out.
        emissions : torch.tensor
            Word level scores for each tag of dimension batch_size x max
            sentence length within batch + 1 x |T| (scores for the BOS
            initial tag have already been removed since BOS is
            only needed for the transformer).

        Returns
        ---
        torch.tensor
            Log Z for each sample in W.
        """
        T = self.t_cal

        normalizer = torch.zeros([W.shape[0], W.shape[1] + 1, T.shape[0]], dtype=torch.float64)

        for i in range(W.shape[0]):
            # Remove padding
            idx = torch.where(W[i] == self.eos_idx)[0]
            # Remove EOS and BOS if present
            if torch.any(idx):
                # print(f"EOS at {idx}")
                normalizer[i, idx, :] = 1
            else:
                # print("No EOS")
                normalizer[i, -1, :] = 1

        N = W.shape[0]
        for i in reversed(range(W.shape[1])):
            for t1 in T:
                normalizer[:, i, t1] += torch.sum(
                    torch.exp(emissions[:, i, :] + self.transitions[None, t1, :])
                    * normalizer[:, i + 1, :].clone(),
                    1,
                )
                # for t2 in T:
                #     normalizer[:, i, t1] += torch.exp(emissions[:, i, t2] + self.transitions[t1, t2]) * normalizer[:, i+1, t2].clone()

        # print(torch.log(normalizer))
        return torch.log(normalizer[:, 0, 0])

    def forward_log_Z(self, W, emissions):
        """Calculate log Z using the forward algorithm.

        NB: You do need to vectorize this over samples.

        Parameters
        ---
        W : torch.tensor
            Words for each sequence within the batch.
            Of dimension batch size x longest sequence within batch + 1.
            Note the paddings, EOS and BOS that have been added to W
            for usage with BERT so we mask them out here. We expect
            W to already have the initial BOS word indices taken out.
        emissions : torch.tensor
            Word level scores for each tag of dimension batch_size x max
            sentence length within batch + 1 x |T| (scores for the BOS
            initial tag have already been removed since BOS is
            only needed for the transformer).

        Returns
        ---
        torch.tensor
            Log Z for each sample in W.
        """
        T = self.t_cal

        normalizer = torch.ones([W.shape[0]], dtype=torch.float64)

        for idx, sample in enumerate(W):
            # Remove padding
            sample = sample[torch.where(sample != self.pad_idx_word)[0]]
            # Remove EOS and BOS if present
            if torch.any(sample == self.eos_idx):
                sample = sample[:-1]
            if torch.any(sample == self.bos_idx):
                sample = sample[1:]

            last_round = torch.ones(T.shape[0], dtype=torch.float64)

            for t in T:
                last_round[t] = torch.exp(emissions[idx, 0, t] + self.transitions[0, t])

            N = sample.shape[0]
            for i in range(1, N):

                temp = torch.zeros(T.shape[0], dtype=torch.float64)

                for t1 in T:
                    for t2 in T:
                        temp[t1] += (
                            torch.exp(emissions[idx, i, t1] + self.transitions[t2, t1])
                            * last_round[t2]
                        )

                last_round = temp

            normalizer[idx] = torch.sum(last_round)
        # print(torch.log(normalizer))
        return torch.log(normalizer)

    def backward_entropy(self, W, emissions):
        """Calculate the unnormalized entropy using the backward algorithm.

        NB: You do need to vectorize this over samples.

        Parameters
        ---
        W : torch.tensor
            Words for each sequence within the batch.
            Of dimension batch size x longest sequence within batch + 1.
            Note the paddings, EOS and BOS that have been added to W
            for usage with BERT so we mask them out here. We expect
            W to already have the initial BOS word indices taken out.
        emissions : torch.tensor
            Word level scores for each tag of dimension batch_size x max
            sentence length within batch + 1 x |T| (scores for the EOS
            initial tag have already been removed since EOS is
            only needed for the transformer).

        Returns
        ---
        torch.tensor
            Unnormalized entropy for each sample in W.
        """
        T = self.t_cal

        beta = torch.zeros([W.shape[0], W.shape[1] + 1, T.shape[0], 2], dtype=torch.float64)

        for i in range(W.shape[0]):
            idx = torch.where(W[i] == self.eos_idx)[0]
            if torch.any(idx):
                # print(f"EOS at {idx}")
                beta[i, idx, :, 0] = 1
                beta[i, idx, :, 1] = 0
            else:
                # print("No EOS")
                beta[i, -1, :, 0] = 1
                beta[i, -1, :, 1] = 0

        # with open("beta.txt", "w") as f:
        #     for i in range(W.shape[1] + 1):
        #         f.write(f"beta start ({i})=\n{beta[0, i, :, 0]}, {beta[0, i, :, 1]}\n")

        emissions.type(torch.float64)
        self.transitions.type(torch.float64)
        for i in reversed(range(W.shape[1])):
            for t1 in T:
                w = torch.exp(emissions[:, i, :] + self.transitions[None, t1, :])
                y = -w * torch.log(w)
                beta[:, i, t1, 0] += torch.sum(w * beta[:, i + 1, :, 0].clone(), 1)
                beta[:, i, t1, 1] += torch.sum(
                    w * beta[:, i + 1, :, 1].clone() + y * beta[:, i + 1, :, 0].clone(), 1
                )

            # with open("beta.txt", "a") as f:
            #     f.write(f"beta[{i}] {beta.shape}=\n{beta[0, i, :, 0]}, {beta[0, i, :, 1]}\n")

        # print(beta[0, :, :, :])
        return beta[:, 0, 0, 1]

    def backward_viterbi_log(self, W, emissions):
        """Calculate the best tagging using the backward algorithm and return
            both the scoring matrix in log-space and the backpointer matrix.

        NB: You do not need to vectorize this over samples.

        Parameters
        ---
        W : torch.tensor
            Of dimension longest sequence within batch + 2 or less.
            Note the paddings, EOS and BOS that have been added to W
            for usage with BERT so we manually remove them here if present.
        emissions : torch.tensor
            Word level scores for each tag of dimension max
            sentence length within batch + 1 x |T| (we assume scores for the BOT
            initial tag have already been removed since BOT/BOS is
            only needed for the transformer).

        Returns
        ---
        Tuple[torch.tensor, torch.tensor]
            Tuple containing the scoring matrix in log-space and the
            backpointer matrix for recovering the best tagging.
        """
        T = self.t_cal

        # Remove padding
        W = W[torch.where(W != self.pad_idx_word)[0]]
        # Remove EOS and BOS if present
        if torch.any(W == self.eos_idx):
            W = W[:-1]
        if torch.any(W == self.bos_idx):
            W = W[1:]

        N = W.shape[0]
        vit = torch.ones([N + 1, T.shape[0]], dtype=torch.float64)

        bpm = torch.zeros([N, T.shape[0]], dtype=torch.int)
        for i in reversed(range(0, N)):

            temp = torch.ones(T.shape[0], dtype=torch.float64)

            for t1 in T:
                max = torch.zeros(1, dtype=torch.float64)
                parent = -1
                for t2 in T:
                    if (
                        max
                        < torch.exp(emissions[i, t2] + self.transitions[t1, t2]) * vit[i + 1, t2]
                    ):
                        max = (
                            torch.exp(emissions[i, t2] + self.transitions[t1, t2]) * vit[i + 1, t2]
                        )
                        parent = t2

                bpm[i, t1] = parent
                temp[t1] = max

            vit[i] = temp
        return [torch.log(vit[:-1]), bpm]

    def dijkstra_viterbi_log(self, W, emissions):
        """Calculate the best tagging using Dijsktra's algorithm and return
            both the best score and best tagging in log space.

        NB: You do not need to vectorize this over samples.

        Parameters
        ---
        W : torch.tensor
            Of dimension longest sequence within batch + 2 or less.
            Note the paddings, EOS and BOS that have been added to W
            for usage with BERT so we manually remove them here if present.
        emissions : torch.tensor
            Word level scores for each tag of dimension max
            sentence length within batch + 1 x |T| (we assume scores for the BOT
            initial tag have already been removed since BOT/BOS is
            only needed for the transformer).


        Returns
        ---
        Tuple[torch.tensor, None, log_Z]
            Tuple containing the log-score of the best tagging.
            NB: Since there were some changes in the assignment,
            we don't expect you to return the backpointer matrix
            this year.
            NB 2: We return log_Z if we already use it within the method
            to calculate probabilities, such that we don't have to
        """
        T = self.t_cal

        log_Z = self.backward_log_Z(W.unsqueeze(0), emissions.unsqueeze(0))
        # print(log_Z)

        # print("bare:")
        # print(W)
        # Remove padding
        W = W[torch.where(W != self.pad_idx_word)[0]]
        # Remove EOS and BOS if present
        if torch.any(W == self.eos_idx):
            W = W[:-1]
        if torch.any(W == self.bos_idx):
            W = W[1:]
        # print("trimmed:")
        # print(W)

        g = torch.zeros([W.shape[0] + 1, T.shape[0]], dtype=torch.float64)
        # print(f"g: {g.shape}")
        pq = PriorityQueue()
        popped = set()

        pq.push((0, (self.bot_idx, 0)))
        # print(f"push:   0, (0, {self.bot_idx})")

        while len(pq) > 0:
            p = pq.pop()
            score = p[0]
            n = p[1][0]
            t1 = p[1][1]
            # print(f"pop:   {score}, ({n}, {t1})")
            # print(f"score:  {score}")
            # print(f"n:      {n}")
            # print(f"t1:     {t1}")

            popped.add((n, t1))
            g[n, t1] = score

            if n < W.shape[0]:
                for t2 in T:
                    if (n + 1, t2) not in popped:
                        # print(emissions[n, t2])
                        new_score = emissions[n, t2] + self.transitions[t1, t2] - log_Z + g[n, t1]
                        # print(new_score)
                        pq.push((new_score, (n + 1, t2.item())))
                        # print(f"push:   {new_score}, ({n+1}, {t2})")

        # print(g)
        return [torch.max(g[W.shape[0], :]), None, log_Z]


for i, data in enumerate(train_dataloader):
    W_train = F.to_tensor(data["words"], padding_value=pad_token_idx)
    T_train = F.to_tensor(data["pos"], padding_value=pos_vocab[pad_token])
    if i == 0:
        break

bert = BertModel.from_pretrained(TRANSFORMER)
T_CAL = torch.tensor([i for i in range(pos_vocab.__len__())])
crf = NeuralCRF(
    pad_idx_word=pad_token_idx,
    pad_idx_pos=pos_vocab[pad_token],
    bos_idx=init_token_idx,
    eos_idx=sep_token_idx,
    bot_idx=pos_vocab[init_token],
    eot_idx=pos_vocab[sep_token],
    t_cal=T_CAL,
    transformer=bert,
)

import pickle


def train_model_report_accuracy(
    crf,
    lr,
    epochs,
    train_dataloader,
    dev_dataloader,
    pad_token_idx_word,
    pad_token_idx_tag,
):

    """Train model for `epochs` epochs and report performance on
        dev set after each epoch.

    Parameters
    ---
    crf : NeuralCRF
    lr : float
        Learning rate to train with.
    epochs : int
        For how many epochs to train.
    train_dataloader : torch.DataLoader
    dev_dataloder : torch.DataLoader
    pad_token_idx_word : int
        Index with which to pad the word indices.
    pad_token_idx_tag : int
        Index with which to pad the tag indices.
    """
    crf.train(True)
    optimizer = torch.optim.Adam(crf.parameters(), lr=lr)
    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader):
            if i % 20 == 0:
                print(f"Epoch {epoch}, batch {i}")
            torch.autograd.set_detect_anomaly(True)
            W = F.to_tensor(data["words"], padding_value=pad_token_idx_word)
            T = F.to_tensor(data["pos"], padding_value=pad_token_idx_tag)
            # W.to(crf.device)
            # T.to(crf.device)
            for param in crf.parameters():
                param.grad = None
            loss = crf.loss(T, W)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predicted_sequences = []
            true_sequences = []
            for i_dev, data_dev in enumerate(valid_dataloader):
                W_dev = F.to_tensor(data_dev["words"], padding_value=pad_token_idx_word)
                T_dev = F.to_tensor(data_dev["pos"], padding_value=pad_token_idx_tag)
                sequence_viterbi = crf(W_dev)
                predicted_sequences += sequence_viterbi
                for ix in range(W_dev.shape[0]):
                    true_sequences += [T_dev[ix, 1 : (sequence_viterbi[ix].shape[0] + 1)]]

            acc = torch.tensor(0.0)
            for ix in range(len(predicted_sequences)):
                acc += torch.mean((predicted_sequences[ix] == true_sequences[ix]).float())
            acc = acc / len(predicted_sequences)
            print("-------------------------")
            print(f"Epoch: {epoch + 1} / {epochs}")
            print(f"Development set accuracy: {acc}")
            print("-------------------------")
            pickle.dump(crf, open(f"crf_epoch{epoch+1}", "wb"))
        epoch += 1
    return None


print(f"Starting training with beta 1.0")

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

entropy_regularized_crf = NeuralCRF(
    pad_idx_word=pad_token_idx,
    pad_idx_pos=pos_vocab[pad_token],
    bos_idx=init_token_idx,
    eos_idx=sep_token_idx,
    bot_idx=pos_vocab[init_token],
    eot_idx=pos_vocab[sep_token],
    t_cal=T_CAL,
    transformer=bert,
    beta=1.0,
)
train_model_report_accuracy(
    entropy_regularized_crf,
    LR,
    EPOCHS,
    train_dataloader,
    valid_dataloader,
    pad_token_idx,
    pos_vocab[pad_token],
)

pickle.dump(crf, open("crf_b1_0", "wb"))

print(f"Starting training with beta 10.0")

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

entropy_regularized_crf = NeuralCRF(
    pad_idx_word=pad_token_idx,
    pad_idx_pos=pos_vocab[pad_token],
    bos_idx=init_token_idx,
    eos_idx=sep_token_idx,
    bot_idx=pos_vocab[init_token],
    eot_idx=pos_vocab[sep_token],
    t_cal=T_CAL,
    transformer=bert,
    beta=10.0,
)
train_model_report_accuracy(
    entropy_regularized_crf,
    LR,
    EPOCHS,
    train_dataloader,
    valid_dataloader,
    pad_token_idx,
    pos_vocab[pad_token],
)

pickle.dump(crf, open("crf_b10", "wb"))

print(f"Starting training with beta 0.1")

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

entropy_regularized_crf = NeuralCRF(
    pad_idx_word=pad_token_idx,
    pad_idx_pos=pos_vocab[pad_token],
    bos_idx=init_token_idx,
    eos_idx=sep_token_idx,
    bot_idx=pos_vocab[init_token],
    eot_idx=pos_vocab[sep_token],
    t_cal=T_CAL,
    transformer=bert,
    beta=0.1,
)
train_model_report_accuracy(
    entropy_regularized_crf,
    LR,
    EPOCHS,
    train_dataloader,
    valid_dataloader,
    pad_token_idx,
    pos_vocab[pad_token],
)

pickle.dump(crf, open("crf_b0_1", "wb"))
