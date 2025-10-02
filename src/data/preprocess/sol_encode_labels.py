'''
This script encodes labels for "instrument family" and "pitch" 
in the SOLPMT metadata, then adds these encoded labels 
to the metadata file.
'''
import pandas as pd
import re
from pathlib import Path

from data.utils import str2midi


def main(path: str) -> None:
    path = Path(path)
    assert path.exists()
    response = input(f"You have launched a script to overwrite {path}. Proceed (y/n)? ")
    if response.lower() != "y":
        print("Exiting...")
        return

    df = pd.read_csv(path)

    # rename column: label to label_modulation
    df.rename(columns={"label": "label_modulation"}, inplace=True)

    # add new column: label_instrument_family
    # first convert instrument_family to codes
    df["label_instrument_family"] = \
        df["instrument family"].astype("category").cat.codes

    # add new columns: pitch, midi_pitch
    # use regex to extract pitch from filename
    regex = r'[A-Ga-g]#?\d'
    search = lambda f: re.search(regex, f).group(0) \
        if re.search(regex, f) else None
    df["pitch"] = df["file_name"].apply(search)
    # transform with str2midi
    df["midi_pitch"] = df["pitch"].apply(lambda p: str2midi(p) \
                                         if p else None).astype("Int64")

    # write changes to csv
    df.to_csv(path, index=False)

if __name__ == "__main__":
    main()