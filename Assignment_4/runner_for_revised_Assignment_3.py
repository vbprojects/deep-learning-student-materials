from importlib import import_module

from urllib.request import urlretrieve
from pathlib import Path
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
# Utilities for handling variable length sequences
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import numpy as np
import random
import unicodedata
import re

import time

import math

from collections import Counter


def run(EncoderGRU, DecoderGRU, Seq2Seq, train, evaluate):

    # 3.0 Download the Data

    url = "https://download.pytorch.org/tutorial/data.zip"
    zip_path = Path("data.zip")

    # wget https://download.pytorch.org/tutorial/data.zip
    urlretrieve(url, zip_path)

    # unzip -o data.zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=".")


    # 3.1 Imports and Utilities (Provided)

    # Set random seeds for reproducibility
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Define special tokens
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3


    # 3.2 Vocabulary and Data Loading (Provided)

    class Lang:
        """A class to hold the vocabulary of a language."""
        def __init__(self, name):
            self.name = name
            self.word2index = {"<PAD>": PAD_IDX, "<SOS>": SOS_IDX, "<EOS>": EOS_IDX, "<UNK>": UNK_IDX}
            self.index2word = {PAD_IDX: "<PAD>", SOS_IDX: "<SOS>", EOS_IDX: "<EOS>", UNK_IDX: "<UNK>"}
            self.n_words = 4

        def addSentence(self, sentence):
            for word in sentence.split(' '):
                self.addWord(word)

        def addWord(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

    def normalizeString(s):
        s = s.lower().strip()
        # Normalize Unicode characters (e.g., remove accents)
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s.strip()

    # We filter for relatively short sentences
    MAX_LENGTH = 15
    NUM_EXAMPLES = 15000

    def prepareData(lang1, lang2):
        print("Reading lines...")
        lines = open(f'data/{lang1}-{lang2}.txt', encoding='utf-8').read().strip().split('\n')
        
        # Limit the number of examples and normalize
        pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines[:NUM_EXAMPLES]]

        # Filter pairs by length
        pairs = [p for p in pairs if len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH]
        
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

        print(f"Trimmed to {len(pairs)} sentence pairs")
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print(f"Vocabularies: {input_lang.name} ({input_lang.n_words}), {output_lang.name} ({output_lang.n_words})")
        return input_lang, output_lang, pairs

    input_lang, output_lang, pairs = prepareData('eng', 'fra')


    # 3.3 Dataset and DataLoader (Provided)

    class TranslationDataset(Dataset):
        def __init__(self, pairs, input_lang, output_lang, reverse_source=True):
            self.pairs = pairs
            self.input_lang = input_lang
            self.output_lang = output_lang
            self.reverse_source = reverse_source

        def __len__(self):
            return len(self.pairs)

        def indexesFromSentence(self, lang, sentence):
            return [lang.word2index.get(word, UNK_IDX) for word in sentence.split(' ')]

        def __getitem__(self, idx):
            pair = self.pairs[idx]
            src_text = pair[0]
            tgt_text = pair[1]

            src_indices = self.indexesFromSentence(self.input_lang, src_text)
            tgt_indices = self.indexesFromSentence(self.output_lang, tgt_text)

            # Apply the Reversal Trick to the source sentence
            if self.reverse_source:
                src_indices.reverse()

            # Add EOS token to both
            src_indices.append(EOS_IDX)
            tgt_indices.append(EOS_IDX)

            return torch.tensor(src_indices, dtype=torch.long), \
                torch.tensor(tgt_indices, dtype=torch.long)

    # Collate function to handle padding and return lengths
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_item, tgt_item in batch:
            src_batch.append(src_item)
            tgt_batch.append(tgt_item)
        
        # Get the lengths of the source sequences BEFORE padding
        src_lengths = torch.tensor([len(s) for s in src_batch])
        
        # Pad the sequences
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)

        # We return the lengths as well for packing later
        return src_batch.to(device), src_lengths, tgt_batch.to(device)

    # Create Datasets and DataLoaders
    BATCH_SIZE = 64
    dataset = TranslationDataset(pairs, input_lang, output_lang, reverse_source=True)

    # Split into train and validation (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


    # 5.1 Initialization (Provided)

    # Hyperparameters
    INPUT_DIM = input_lang.n_words
    OUTPUT_DIM = output_lang.n_words
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2 # Using 2 layers
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # Initialize models (Ensure Tasks 1-3 are completed first!)
    enc = EncoderGRU(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
    dec = DecoderGRU(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT).to(device)
    model = Seq2Seq(enc, dec, device).to(device)

    # Initialize weights (common practice for RNNs)
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
    model.apply(init_weights)

    # Optimizer
    optimizer = optim.Adam(model.parameters())

    # Loss function: CrossEntropyLoss, ignoring the padding index
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def extract_tokens_until_eos(indices, lang):
        tokens = []
        for idx in indices:
            idx = int(idx)
            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, SOS_IDX):
                continue
            tokens.append(lang.index2word[idx])
        return tokens
    
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def corpus_bleu(list_of_references, hypotheses, max_n = 4):
        clipped_counts = [0] * max_n
        total_counts = [0] * max_n
        ref_length = 0
        hyp_length = 0
        for references, hypothesis in zip(list_of_references, hypotheses):
            reference = references[0]
            ref_length += len(reference)
            hyp_length += len(hypothesis)
            for n in range(1, max_n + 1):
                hyp_ngrams = Counter(ngrams(hypothesis, n))
                ref_ngrams = Counter(ngrams(reference, n))
                total_counts[n - 1] += sum(hyp_ngrams.values())
                for gram, count in hyp_ngrams.items():
                    clipped_counts[n - 1] += min(count, ref_ngrams.get(gram, 0))
        precisions = []
        for clipped, total in zip(clipped_counts, total_counts):
            if total == 0:
                precisions.append(0.0)
            else:
                precisions.append(clipped / total)
        if min(precisions) == 0.0:
            return 0.0
        brevity_penalty = 1.0 if hyp_length > ref_length else math.exp(1 - ref_length / max(hyp_length, 1))
        bleu = brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)
        return bleu

    def compute_validation_bleu(model, dataset, src_lang, tgt_lang, device, max_examples = None):
        list_of_references = []
        hypotheses = []
        total = len(dataset) if max_examples is None else min(len(dataset), max_examples)
        for i in range(total):
            src_tensor, tgt_tensor = dataset[i]
            src_sentence = pairs[dataset.indices[i]][0]
            predicted_tokens = translate_sentence(src_sentence, src_lang, tgt_lang, model, device)
            reference_tokens = extract_tokens_until_eos(tgt_tensor.tolist(), tgt_lang)
            list_of_references.append([reference_tokens])
            hypotheses.append(predicted_tokens)
        return corpus_bleu(list_of_references, hypotheses)

    print(f'The model has {count_parameters(model):,} trainable parameters')


    ### 5.3 Running the Training (Provided)

    N_EPOCHS = 30  # Note: 15 epochs may not be sufficient for good translations!
    CLIP = 1

    best_valid_loss = float('inf')

    print("Starting training...")

    # NOTE: Uncomment the loop content after completing the tasks above.
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss, valid_token_accuracy = evaluate(model, val_loader, criterion, PAD_IDX)

        end_time = time.time()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'seq2seq-gru-model-without-attention.pt')

        print(f'Epoch: {epoch+1:02} | Time: {int(end_time - start_time)}s')
        # PPL (Perplexity) is exp(loss), a common metric for language models.
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print(f'\t Val. Token Acc: {100.0 * valid_token_accuracy:6.2f}%')


    ### 6.1 Inference (Greedy Decoding)

    def translate_sentence(sentence, src_lang, tgt_lang, model, device, max_len=50):
        model.eval()

        # 1. Preprocess the input sentence (normalize and reverse!)
        normalized_sentence = normalizeString(sentence)
        reversed_sentence = ' '.join(normalized_sentence.split(' ')[::-1])

        # 2. Convert to indices and tensor
        indices = [src_lang.word2index.get(word, UNK_IDX) for word in reversed_sentence.split(' ')] + [EOS_IDX]
        src_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device) # (1, T)
        src_len = torch.tensor([len(indices)])

        # 3. Encode the sentence
        with torch.no_grad():
            hidden = model.encoder(src_tensor, src_len)

        # 4. Start decoding
        trg_indices = [SOS_IDX]
        input_tensor = torch.tensor([SOS_IDX], dtype=torch.long).to(device) # (1)

        for i in range(max_len):
            with torch.no_grad():
                output, hidden = model.decoder(input_tensor, hidden)

            # 5. Greedy Decoding
            pred_token = output.argmax(1).item()
            trg_indices.append(pred_token)

            # Check for <EOS>
            if pred_token == EOS_IDX:
                break

            # Prepare the next input
            input_tensor = torch.tensor([pred_token], dtype=torch.long).to(device)

        # 6. Convert indices back to words
        trg_tokens = [tgt_lang.index2word[i] for i in trg_indices]
        return trg_tokens[1:-1] # Exclude <SOS> and <EOS>

    # Qualitative Analysis (Uncomment after training)
    model.load_state_dict(torch.load('seq2seq-gru-model-without-attention.pt'))
    validation_bleu = compute_validation_bleu(model, val_dataset, input_lang, output_lang, device)
    print(f'\nWITHOUT ATTENTION -- VALIDATION BLEU: {100.0 * validation_bleu:.2f}')
    examples = ["i am cold", "she is happy", "he is running", "we are ready"]
    for example in examples:
        translation = translate_sentence(example, input_lang, output_lang, model, device)
        print(f"EN: {example}")
        print(f"FR: {' '.join(translation)}\n")


    def diagnose_model(model, src_lang, tgt_lang, pairs, device, num_examples=5):
        """
        Check if model can translate training examples (memorization test)
        """
        print("\n" + "=" * 70)
        print("MODEL DIAGNOSIS - Testing on Training Examples")
        print("=" * 70)
        
        for i in range(num_examples):
            en_sentence = pairs[i][0]
            fr_actual = pairs[i][1]
            fr_predicted = translate_sentence(en_sentence, src_lang, tgt_lang, model, device)
            
            print(f"\nExample {i+1}:")
            print(f"  EN (input):     {en_sentence}")
            print(f"  FR (expected):  {fr_actual}")
            print(f"  FR (predicted): {' '.join(fr_predicted)}")
            
            # Calculate word overlap
            expected_words = set(fr_actual.split())
            predicted_words = set(fr_predicted)
            overlap = expected_words.intersection(predicted_words)
            if len(expected_words) > 0:
                accuracy = len(overlap) / len(expected_words) * 100
                print(f"  Word overlap:   {len(overlap)}/{len(expected_words)} ({accuracy:.1f}%)")

    # Run after loading best model
    diagnose_model(model, input_lang, output_lang, pairs, device)

    print("\n" + "=" * 70)
    print("FINAL METRICS -- WITHOUT ATTENTION")
    print("=" * 70)
    final_valid_loss, final_valid_token_accuracy = evaluate(model, val_loader, criterion, PAD_IDX)
    print(f"Validation Loss: {final_valid_loss:.4f}")
    print(f"Validation Token Acc: {100.0 * final_valid_token_accuracy:.2f}%")
    print(f"Validation BLEU: {100.0 * validation_bleu:.2f}")


def main() -> None:
    EncoderGRU = import_module("EncoderGRU")
    DecoderGRU = import_module("DecoderGRU")
    Seq2Seq = import_module("Seq2Seq")
    train = import_module("train")
    evaluate = import_module("evaluate")
    run(EncoderGRU.EncoderGRU, DecoderGRU.DecoderGRU, Seq2Seq.Seq2Seq, train.train, evaluate.evaluate)


if __name__ == '__main__':
    main()