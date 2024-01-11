import unicodedata
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
        df["signer"] = df["signer"].apply(lambda x: int(x[-2:])-1)

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
        path = self.root / "annotations" / "manual" / f"new_{split}.SI5.corpus.csv"
        # if split == 'test':
        #     path = "/2tssd/rzuo/data/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/{}.corpus.csv".format(split)
        df = pd.read_csv(path, sep="|")
        df["annotation"] = df["annotation"].apply(str.split)
        
        # if split == 'test':
        #     df = df[df["signer"] != 'Signer05']

        signer_lst = [1,2,3,4,6,7,8,9,5]  #Signer05 only appears in dev and test set.
        df["signer"] = df["signer"].apply(lambda x: signer_lst.index(int(x[-2:])))

        if split == "train" and aligned_annotation:
            # append alignment to data frame
            # note that only train split has alignment
            adf = self.load_alignment()
            adf = adf.rename({"gloss": "annotation"}, axis=1)
            adf = adf["annotation"]
            del df["annotation"]
            df = pd.merge(df, adf, "left", "id")

        df["folder"] = df["folder"].apply(lambda s: s.rsplit("/", 1)[0])
        # df["folder"] = split + "/" + df["folder"].apply(lambda s: s.rsplit("/", 1)[0])
        return df

    def get_frames(self, sample, type):
        frames = (self.root / "features" / type / sample["folder"]).glob("*.png")
        # frames = (Path('../../data/phoenix2014-release/phoenix-2014-multisigner') / "features" / type / sample["folder"]).glob("*.png")
        
        return sorted(frames)


class PhoenixSI7Corpus(Corpus):
    def __init__(self, root, max_len):
        self.root = Path(root)

    def load_data_frame(self, split, aligned_annotation=False):
        """Load corpus."""
        path = self.root / "annotations" / "manual" / f"new_{split}.SI3.corpus.csv"
        df = pd.read_csv(path, sep="|")
        df["annotation"] = df["annotation"].apply(str.split)
        # signer_lst = [1,2,3,4,5,6,8,9,7]  #signer07 only appears in dev and test set.
        signer_lst = [1,2,4,5,6,7,8,9,3]  #signer03 only appears in dev and test set.
        df["signer"] = df["signer"].apply(lambda x: signer_lst.index(int(x[-2:])))

        if split == "train" and aligned_annotation:
            # append alignment to data frame
            # note that only train split has alignment
            adf = self.load_alignment()
            adf = adf.rename({"gloss": "annotation"}, axis=1)
            adf = adf["annotation"]
            del df["annotation"]
            df = pd.merge(df, adf, "left", "id")

        df["folder"] = df["folder"].apply(lambda s: s.rsplit("/", 1)[0])
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
        df["signer"] = df["signer"].apply(lambda x: int(x[-2:]))
        
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


class TVBCorpus(PhoenixCorpus):
    def __init__(self, root, max_len):
        self.root = Path(root)

    def load_data_frame(self, split) -> pd.DataFrame:
        if split == 'dev':
            split = 'val'
        path = self.root / "split" / "v5.3" / f"{split}.csv"
        df = pd.read_csv(path, sep="|")
        df = df.dropna()

        words = df["words"].apply(lambda s: unicodedata.normalize("NFKC", s))
        words = words.apply(list)
        df["words"] = words

        glosses = df["glosses"].apply(lambda s: unicodedata.normalize("NFKC", s))
        # glosses = glosses.str.replace(r"(\d+)([^ ]+)", r"\g<1> \g<2>", regex=True)
        # glosses = glosses.str.replace("[#%*!@]", "", regex=True)
        # glosses = glosses.str.replace("BAD-SEGMENT", "", regex=False)
        # glosses = glosses.str.replace("MUMBLE", "", regex=False)
        # glosses = glosses.str.replace(r"\(.+?\)", "", regex=True)
        # glosses = glosses.str.replace(r" +", " ", regex=True)
        glosses = glosses.str.split("[ +]")
        df["glosses"] = glosses
        # df = df[df['glosses'] != ""]
        df = df.rename(columns={"glosses": "annotation"})
        # df["id"] = df["folder"].apply(lambda s: s.replace("/", "-"))
        df["folder"] = df["signer"] = df["id"]
        return df
    
    def get_frames(self, sample, type):
        frames = (self.root / "grouped" / "sign" / sample["folder"]).glob("*.jpg")
        return sorted(frames)

    # def gloss_table(self):
    #     df = self.load_csv("train")
    #     return LookupTable(words={w for ws in df["glosses"].tolist() for w in ws})

    # def word_table(self):
    #     df = self.load_csv("train")
    #     return LookupTable(words={w for ws in df["words"].tolist() for w in ws})


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
        df["signer"] = df["id"].apply(lambda s: int(s.split("_")[0][1:])-1)
        
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
        signer_lst = []
        k = 0
        for i in range(10):
            if 'SI'+str(i) in self.split_doc:
                k = i
                continue
            signer_lst.append(i)
        signer_lst.append(k)
        df['signer'] = df["signer"].apply(lambda x: signer_lst.index(x))
        
        if split == 'train':
            df = df[df['split']=='train']
        elif split == 'dev':
            df = df[df['split']=='dev']
        elif split == 'test':
            df = df[df['split']=='test']
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
