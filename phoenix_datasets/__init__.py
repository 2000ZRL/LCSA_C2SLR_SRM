from .datasets import VideoTextDataset
from .corpora import PhoenixCorpus, PhoenixTCorpus, PhoenixSICorpus, CSLCorpus, CSLDailyCorpus, CFSWCorpus


class PhoenixVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixCorpus

class PhoenixTVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixTCorpus

class PhoenixSIVideoTextDataset(VideoTextDataset):
    Corpus = PhoenixSICorpus
    
class CSLVideoTextDataset(VideoTextDataset):
    Corpus = CSLCorpus

class CSLDailyVideoTextDataset(VideoTextDataset):
    Corpus = CSLDailyCorpus
    
class CFSWVideoTextDataset(VideoTextDataset):
    Corpus = CFSWCorpus