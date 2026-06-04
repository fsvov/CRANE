import os
import torch
import torchaudio
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random
import nltk
# ============================================================
# NLTK data path — set NLTK_DATA env var or update this default
# ============================================================
_nltk_path = os.environ.get('NLTK_DATA', os.path.expanduser('~/nltk_data'))
if os.path.isdir(_nltk_path):
    nltk.data.path.insert(0, _nltk_path)
from nltk.corpus import wordnet

def random_word_insertion(words, seed = 42):
    random.seed(seed)
    random_word = random.choice(words)
    synonyms = wordnet.synsets(random_word)
    if not synonyms:
        return None
    synonym = synonyms[0].lemmas()[0].name()
    return synonym
def synonym_replacement(text, n, seed = 42):
    random.seed(seed)
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)
def random_insertion(text, n, seed = 42):
    random.seed(seed)
    words = text.split()
    for _ in range(n):
        new_word = random_word_insertion(words)
        if new_word:
            words.insert(random.randint(0, len(words)), new_word)
    return ' '.join(words)
def random_deletion(text, seed=42):
    random.seed(seed)
    words = text.split()
    if len(words) == 1:
        return text
    index_to_delete = random.randint(0, len(words) - 1)
    new_sentence = [word for i, word in enumerate(words) if i != index_to_delete]
    return ' '.join(new_sentence)

def add_gaussian_noise(audio, std=0.005, seed = 42):
    """Add Gaussian noise to the audio."""
    torch.manual_seed(seed)
    noise = torch.randn_like(audio) * std
    return audio + noise
def random_volume_adjustment(audio, seed = 42, min_gain=0.9, max_gain=1.1):
    """Randomly adjust the volume of the audio."""
    random.seed(seed)
    gain = random.uniform(min_gain, max_gain)
    return audio * gain
def time_masking(audio, max_mask_length=100,seed = 42):
    """Apply time masking to the audio."""
    if len(audio) < max_mask_length:
        return audio
    random.seed(seed)
    # Choose a random start point for the mask
    start = random.randint(0, len(audio) - max_mask_length)
    # Apply mask by setting a random segment to zero
    audio[start:start + max_mask_length] = 0
    return audio
def frequency_masking(audio, max_mask_length=1600, seed = 42):
    """Apply frequency masking to the audio (simulated on the audio tensor)."""
    length = audio.size(-1)
    if length < max_mask_length:
        return audio
    random.seed(seed)
    # Select a random position to start the mask
    start = random.randint(0, length - max_mask_length)
    # Apply mask (zero out the selected portion)
    audio[:, start:start + max_mask_length] = 0
    return audio
def random_crop(audio, crop_length=2000,seed = 42):
    """Randomly crop the audio."""
    if len(audio) <= crop_length:
        return audio
    random.seed(seed)
    # Random start position for cropping
    start = random.randint(0, len(audio) - crop_length)
    return audio[start:start + crop_length]
def augment_text(text):
    text = synonym_replacement(text, 1)
    #text = random_insertion(text,1)
    #text = random_deletion(text)
    return text
def augment_audio(audio, sample_rate=16000):
    """Apply a series of audio augmentations."""
    #audio = add_gaussian_noise(audio, std=0.05)#0.05
    audio = random_volume_adjustment(audio, 42, min_gain=0.9, max_gain=1.1)
    #audio = time_masking(audio, max_mask_length=100)
    #audio = frequency_masking(audio, max_mask_length=1600)
    #audio = random_crop(audio, crop_length=2000)
    return audio

class Dataset_mosi(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_directory, mode, augment):
        df = pd.read_csv(csv_path)
        df = df[df['mode']==mode].sort_values(by=['video_id','clip_id']).reset_index()
        self.augment = augment
        # store labels
        self.targets_M = df['label']
        # store texts
        df['text'] = df['text'].str[0]+df['text'].str[1::].apply(lambda x: x.lower())
        self.texts = df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        # store audio
        self.audio_file_paths = []
        self.audio_file_aguments_output_path = []
        ## loop through the csv entries
        for i in range(0,len(df)):
            file_name = str(df['video_id'][i])+'/'+str(df['clip_id'][i])+'.wav'
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        # store context
        self.video_id = df['video_id']
    def __getitem__(self, index):
        text = str(self.texts[index])
        sound, sample_rate = torchaudio.load(self.audio_file_paths[index])
        if self.augment:
            text = augment_text(text)
            sound = augment_audio(sound, sample_rate)
        soundData = torch.mean(sound, dim=0, keepdim=False)
        tokenized_text = self.tokenizer(
                text,            
                max_length = 96,
                padding = "max_length",     # Pad to the specified max_length. 
                truncation = True,          # Truncate to the specified max_length. 
                add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
                return_attention_mask = True            
            )
        features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000, return_attention_mask=True, truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()
        #audio_features = torch.tensor(features['input_values'][0], dtype=torch.float32)  # [feature_dim]
        #audio_masks = torch.tensor(features['attention_mask'][0], dtype=torch.long)
        return { # text
                "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
                "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
                # audio
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                 # labels
                "targets": torch.tensor(self.targets_M[index], dtype=torch.float),
                }
    def __len__(self):
        return len(self.targets_M)


def data_loader(batch_size, dataset, seed=42):
    if dataset == 'mosi':
        # ============================================================
        # Data paths — update to your local MOSI dataset location
        # Expected:  ./data/MOSI/label.csv  and  ./data/MOSI/wav/
        # ============================================================
        csv_path = './data/MOSI/label.csv'
        audio_file_path = "./data/MOSI/wav"
        print(os.path.exists(csv_path))
        print(os.path.exists(audio_file_path))
        train_full = Dataset_mosi(csv_path, audio_file_path, 'train', augment = True)
        test_data = Dataset_mosi(csv_path, audio_file_path, 'test', augment = False)

        # Cal = all of val (10%, ~220 samples, no augment) — from val, matches test distribution
        cal_data = Dataset_mosi(csv_path, audio_file_path, 'valid', augment = False)

        # ES = 5% of total (~110 samples) from train, no augment for clean evaluation
        n_train_full = len(train_full)
        n_es = int(n_train_full * 0.077)  # ~110 / ~1430
        indices = torch.randperm(n_train_full,
                                 generator=torch.Generator().manual_seed(seed))
        es_indices = indices[:n_es].tolist()
        train_indices = indices[n_es:].tolist()

        es_data = torch.utils.data.Subset(
            Dataset_mosi(csv_path, audio_file_path, 'train', augment=False),
            es_indices)
        train_data = torch.utils.data.Subset(train_full, train_indices)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        es_loader = DataLoader(es_data, batch_size=batch_size, shuffle=False)
        cal_loader = DataLoader(cal_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, es_loader, cal_loader
    else:
        raise

