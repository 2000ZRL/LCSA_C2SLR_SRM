from .datasets import VideoTextDataset
from .corpora import PhoenixCorpus, PhoenixTCorpus, PhoenixSICorpus, PhoenixSI7Corpus, CSLCorpus, CSLDailyCorpus, CFSWCorpus, TVBCorpus


class PhoenixVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixCorpus

class PhoenixTVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixTCorpus

class PhoenixSIVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixSICorpus

class PhoenixSI7VideoTextDataset(VideoTextDataset):
    Corpus = PhoenixSI7Corpus
    
class CSLVideoTextDataset(VideoTextDataset):
    Corpus = CSLCorpus

class CSLDailyVideoTextDataset(VideoTextDataset):
    Corpus = CSLDailyCorpus
    
class CFSWVideoTextDataset(VideoTextDataset):
    Corpus = CFSWCorpus

class TVBVideoTextDataset(VideoTextDataset):
    Corpus = TVBCorpus