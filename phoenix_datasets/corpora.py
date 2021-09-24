import pandas as pd
from pathlib import Path
from pandarallel import pandarallel
from functools import partial

from .utils import LookupTable

pandarallel.initialize(verbose=1)


class Corpus:
    def __init__(self, root):
        pass

    def load_data_frame(self, split):
        raise NotImplementedError

    def create_vocab(self, data_frame=None):
        df = data_frame or self.load_data_frame("train")
        sentences = df["annotation"].to_list()
        LT = LookupTable([gloss for sentence in sentences for gloss in sentence])
        #print('len LT: ', len(LT.table))
        return LT


class PhoenixCorpus(Corpus):
    def __init__(self, root, max_len):
        self.root = Path(root)

    def load_alignment(self):
        dirname = self.root / "annotations" / "automatic"

        # important to literally read NULL instead read it as nan
        read = partial(pd.read_csv, sep=" ", na_filter=False)
        ali = read(dirname / "train.alignment", header=None, names=["id", "classlabel"])
        cls = read(dirname / "trainingClasses.txt")

        df = pd.merge(ali, cls, how="left", on="classlabel")
        del df["classlabel"]

        df["gloss"] = df["signstate"].apply(lambda s: s.rstrip("012"))

        df["id"] = df["id"].parallel_apply(lambda s: "/".join(s.split("/")[3:-2]))
        grouped = df.groupby("id")

        gdf = grouped["gloss"].agg(" ".join)
        sdf = grouped["signstate"].agg(" ".join)

        df = pd.merge(gdf, sdf, "inner", "id")

        assert (
            len(df) == 5671
        ), f"Alignment file is not correct, expect to have 5671 entries but got {len(df)}."

        return df

    def load_data_frame(self, split, aligned_annotation=False):
        """Load corpus."""
        path = self.root / "annotations" / "manual" / f"{split}.corpus.csv"
        df = pd.read_csv(path, sep="|")
        df["annotation"] = df["annotation"].apply(str.split)

        if split == "train" and aligned_annotation:
            # append alignment to data frame
            # note that only train split has alignment
            adf = self.load_alignment()
            adf = adf.rename({"gloss": "annotation"}, axis=1)
            adf = adf["annotation"]
            del df["annotation"]
            df = pd.merge(df, adf, "left", "id")

        df["folder"] = split + "/" + df["folder"].apply(lambda s: s.rsplit("/", 1)[0])

        return df

    def get_frames(self, sample, type):
        frames = (self.root / "features" / type / sample["folder"]).glob("*.png")
        return sorted(frames)


class PhoenixSICorpus(Corpus):
    def __init__(self, root, max_len):
        self.root = Path(root)

    def load_data_frame(self, split, aligned_annotation=False):
        """Load corpus."""
        path = self.root / "annotations" / "manual" / f"{split}.SI5.corpus.csv"
        df = pd.read_csv(path, sep="|")
        df["annotation"] = df["annotation"].apply(str.split)

        if split == "train" and aligned_annotation:
            # append alignment to data frame
            # note that only train split has alignment
            adf = self.load_alignment()
            adf = adf.rename({"gloss": "annotation"}, axis=1)
            adf = adf["annotation"]
            del df["annotation"]
            df = pd.merge(df, adf, "left", "id")

        df["folder"] = split + "/" + df["folder"].apply(lambda s: s.rsplit("/", 1)[0])

        return df

    def get_frames(self, sample, type):
        frames = (self.root / "features" / type / sample["folder"]).glob("*.png")
        return sorted(frames)
    

class PhoenixTCorpus(PhoenixCorpus):
    def __init__(self, root, max_len):
        self.root = Path(root)
        self.max_len = max_len

    def load_data_frame(self, split):
        """Load corpus."""
        if split == 'train':
            path = self.root / "annotations" / "manual" / "PHOENIX-2014-T.train-complex-annotation.corpus.csv"
        else:
            path = self.root / "annotations" / "manual" / f"PHOENIX-2014-T.{split}.corpus.csv"
        df = pd.read_csv(path, sep="|")
        df = df.rename(columns={"speaker": "signer", "name": "id", "video": "folder", "orth": "annotation"})
        
        df["annotation"] = df["annotation"].apply(str.split)
        df["folder"] = split + "/" + df["folder"].apply(lambda s: s.rsplit("/", 2)[0])
        
        # remove too long videos
        # if split == 'train':
        #     drop_idx = []
        #     for i in range(len(df)):
        #         if len(list((self.root / "features" / "fullFrame-210x260px" / df.iloc[i]["folder"]).glob("*.png"))) > self.max_len:
        #             drop_idx.append(i)
        #     df.drop(drop_idx, inplace=True)

        return df

    def get_frames(self, sample, type):
        frames = (self.root / "features" / type / sample["folder"]).glob("*.png")
        return sorted(frames)
    

class CSLCorpus(PhoenixCorpus):
    def __init__(self, root, max_len):
        self.root = Path(root[0])
        self.split_doc = root[1]
        self.max_len = max_len

    def load_data_frame(self, split):
        """Load corpus."""
        path = self.root / self.split_doc
        df = pd.read_csv(path, sep=",")
        df = df.rename(columns={"VideoID": "id", "Description": "annotation",})
        df["annotation"] = df["annotation"].apply(str.split)
        df["signer"] = df["id"].apply(lambda s: s.split("_")[0])
        
        if split == 'train':
            df = df[df['Type']=='train']
            # remove too long videos
            # drop_idx = []
            # for i in range(len(df)):
            #     if len((list(self.root / "features" / "rgb" / df.iloc[i]["folder"]).glob("*.png"))) > self.max_len:
            #         drop_idx.append(i)
            # df.drop(drop_idx, inplace=True)
        elif split == 'dev':
            if 'dev' in self.split_doc:
                df = df[df['Type']=='dev']
            else:
                df = df[df['Type']=='test']
        elif split == 'test':
            df = df[df['Type']=='test']
        else:
            raise ValueError('We only support train, dev and test but got {}'.format(split))
        
        df = df.dropna()
        return df

    def get_frames(self, sample, type):
        type = "rgb"  #default setting. Make it compatible with phoenixdataset scripts
        frames = (self.root / type / sample["id"]).glob("*.png")
        return sorted(frames)


class CSLDailyCorpus(PhoenixCorpus):
    def __init__(self, root, max_len):
        self.root = Path(root)
        self.split_doc = 'split_1.txt'
        self.max_len = max_len

    def load_data_frame(self, split):
        """Load corpus."""
        path = self.root / self.split_doc
        spl = pd.read_csv(path, sep="|")

        import pickle
        with open(self.root / 'csl2020ct_v1.pkl', 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data['info'])
        df = df.merge(spl, how='inner', on='name')
        df = df.rename(columns={"name": "id", "label_gloss": "annotation"})
        # df["annotation"] = df["annotation"].apply(str.split)
        # df["signer"] = df["id"].apply(lambda s: s.split("_")[0])
        
        if split == 'train':
            df = df[df['split']=='train']
        elif split == 'dev':
            df = df[df['Type']=='dev']
        elif split == 'test':
            df = df[df['Type']=='test']
        else:
            raise ValueError('We only support train, dev and test but got {}'.format(split))
        
        df = df.dropna()
        return df

    def get_frames(self, sample, type):
        # type = "rgb"  #default setting. Make it compatible with phoenixdataset scripts
        frames = (self.root / sample["id"]).glob("*.jpg")
        return sorted(frames)


class CFSWCorpus(PhoenixCorpus):
    def __init__(self, root, max_len):
        self.root = Path(root)
    
    def load_data_frame(self, split):
        path = self.root / "ChicagoFSWild.csv"
        df = pd.read_csv(path)
        _ = df.drop(columns=['url', 'start_time', 'number_of_frames', 'width', 'height', 'label_raw', 'label_notes'], inplace=True)
        df = df[df['partition'] == split]
        df.rename(columns={'label_proc': 'annotation'}, inplace=True)
        df["annotation"] = df["annotation"].apply(lambda x: list(filter(str.isalpha, x)))
        df['id'] = df['filename']
        df = df.dropna()
        return df
    
    def get_frames(self, sample, type):
        type = 'Frames'
        frames = (self.root / type / sample['filename']).glob("*.jpg")
        return sorted(frames)
