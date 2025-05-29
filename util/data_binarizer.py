import pandas as pd
from constants import p


def binarize_bool(x: bool | str) -> int:
    if isinstance(x, str):
        if x == "True":
            return 1
        else:
            return 0
    elif isinstance(x, bool):
        return int(x)


def binarize_true_false_features(data_frame: pd.DataFrame) -> pd.DataFrame:
    bool_columns = [
        "smoke", "drink", "pesticide",
        "skin_cancer_history", "cancer_history",
        "has_piped_water", "has_sewage_system",
        "itch", "grew",	"hurt",	"changed",
        "bleed", "elevation"
    ]
    for column in bool_columns:
        data_frame[column] = data_frame[column].apply(binarize_bool)

    return data_frame


if __name__ == "__main__":
    DF = pd.read_csv(p("result/features.csv"))
    DF.dropna(inplace=True)
    DF.drop(["gender", "background_father", "background_mother"], axis=1, inplace=True)
    DF = binarize_true_false_features(DF)

    DF.to_csv(p("result/features_binarized.csv"))

    print(DF)
