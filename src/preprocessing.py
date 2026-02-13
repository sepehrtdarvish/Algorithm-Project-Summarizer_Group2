import nltk
from typing import List

required_resources = ['punkt', 'punkt_tab']

for resource in required_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        print(f">> Downloading missing NLTK resource: {resource}...")
        nltk.download(resource)

def split_into_sentences(text: str) -> List[str]:
    """
    Splits the raw text into a list of sentences using NLTK.
    """
    sentences = nltk.sent_tokenize(text)
    
    cleaned_sentences = [s.strip().replace('\n', ' ') for s in sentences]
    return cleaned_sentences

def filter_sentences(sentences: List[str], min_length: int) -> List[str]:
    """
    Filters out sentences shorter than min_length.
    """
    return [s for s in sentences if len(s) >= min_length]