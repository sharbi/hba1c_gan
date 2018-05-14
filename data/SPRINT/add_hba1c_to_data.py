import numpy as np
import pandas as pd

def add_hba1c(filename):
    df = pd.read_csv(filename)
    df['HBA1C'] = pd.Series(np.random.randint(35, 150), index=df.index)
    for i, row in df.iterrows():
        hba1c = np.random.randint(35, 150)
        df.set_value(i, 'HBA1C', hba1c)
    df.to_csv("./bp_with_hba1c.csv")


if __name__ == "__main__":
    add_hba1c("bp.csv")
