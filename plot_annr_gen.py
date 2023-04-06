import matplotlib, matplotlib.pyplot as plt, seaborn as sns, pandas as pd

if __name__ == "__main__":
    language_dataset = [(lang, "2016") for lang in ["turkish", "hungarian", "georgian"]]
    language_dataset += [(lang, "2019") for lang in ["arabic", "english", "french", "portuguese", "bashkir", "slovene", "adyghe", "swahili", "zulu", "slovak", "sanskrit", "welsh", "hebrew"]]
    
    # load genmorpho autoencoders
    ae_dfs = []
    for language, dataset in language_dataset:
        file_name=f"genmorpho/logs/ae/{dataset}/{language}/summary.csv"
        df = pd.read_csv(file_name).sort_values(by="POSIX_timestamp", axis=0).drop_duplicates(["split_seed", "model_seed", "language"], keep="last")
        #print(file_name, len(df.index))
        df = df[:50]
        ae_dfs.append(df)
    ae_df = pd.concat(ae_dfs, axis=0)
    ae_df["Language"] = ae_df["language"].str.capitalize()
    ae_df=ae_df.sort_values(by="Language", axis=0)
    ae_df["Dataset"] = "Sigmorphon " + ae_df["dataset"].astype(str)
    #ae_df["Dataset"][ae_df["Dataset"] == "Sigmorphon 2016"] = "Sigmorphon 2016 + Japanese Bigger Analogy Test Set"

    # load annr + genmorpho
    annr_gen_dfs = []
    for language, dataset in language_dataset:
        try:
            file_name=f"logs/ae_annr/{dataset}/{language}/summary.csv"
            df = pd.read_csv(file_name)
            df=df.sort_values(by="POSIX_timestamp", axis=0)
            df=df.drop_duplicates(["split_seed", "model_seed", "language"], keep="last")
            #print(file_name, len(df.index))
            #df = df[:50]
            annr_gen_dfs.append(df)
            print(file_name, len(df))
        except FileNotFoundError:
            pass
    annr_gen_df = pd.concat(annr_gen_dfs, axis=0)
    annr_gen_df["Language"] = annr_gen_df["language"].str.capitalize()
    annr_gen_df=annr_gen_df.sort_values(by="Language", axis=0)
    annr_gen_df["Dataset"] = "Sigmorphon " + annr_gen_df["dataset"].astype(str)
    annr_gen_df["Dataset"][annr_gen_df["Dataset"] == "Sigmorphon 2016"] = "Sigmorphon 2016 + Japanese Bigger Analogy Test Set"


    annr_gen_df["Model"] = "ANNr + genmorpho"
    ae_df["Model"] = "genmorpho AE"
    ae_df["accuracy"] = ae_df["gen_max_word_accuracy"]
    ae_df["cer"] = ae_df["gen_max_cer"]
    both = pd.concat([annr_gen_df, ae_df])

    sns.barplot()
    fig = plt.figure(figsize=(12,4))
    ax: matplotlib.axes.Axes = sns.barplot(data=both, x="Language", hue="Model", y="accuracy", errorbar="se")
    ax.set_ylabel("Word accuracy")
    ax.set_title("Autoencoder word accuracy at test time using inference mode, higher is better (error bars are standard error of the mean)")
    ax.figure.autofmt_xdate(rotation=60)
    plt.tight_layout()
    plt.savefig("figs/annr-gen-word-acc.png")


    sns.barplot()
    fig = plt.figure(figsize=(12,4))
    ax: matplotlib.axes.Axes = sns.barplot(data=both, x="Language", hue="Model", y="cer", errorbar="se")
    ax.set_ylabel("CER")
    ax.set_title("Autoencoder CER at test time using inference mode, lower is better (error bars are standard error of the mean)")
    ax.figure.autofmt_xdate(rotation=60)
    plt.tight_layout()
    plt.savefig("figs/annr-gen-cer.png")



if False:
    # %%
    import matplotlib, matplotlib.pyplot as plt, seaborn as sns, pandas as pd
    language_dataset = [(lang, "2016") for lang in ["turkish", "hungarian", "georgian"]]
    language_dataset += [(lang, "2019") for lang in ["arabic", "english", "french", "portuguese", "bashkir", "slovene", "adyghe", "swahili", "zulu", "slovak", "sanskrit", "welsh", "hebrew"]]
    
    annr_gen_dfs = []
    for language, dataset in language_dataset:
        for seed in range(10):
            try:
                file_name=f"models/ret/{dataset}/{language}/{seed}/summary.csv"
                df = pd.read_csv(file_name)
                annr_gen_dfs.append(df)
                print(file_name, len(df))
            except FileNotFoundError:
                pass
    # %%
    annr_gen_df = pd.concat(annr_gen_dfs, axis=0)
    annr_gen_df["language"] = annr_gen_df["lang"]
    annr_gen_df["Language"] = annr_gen_df["language"].str.capitalize()
    annr_gen_df=annr_gen_df.sort_values(by="Language", axis=0)
    annr_gen_df["Dataset"] = "Sigmorphon " + annr_gen_df["dataset"].astype(str)
    #annr_gen_df["Dataset"][annr_gen_df["Dataset"] == "Sigmorphon 2016"] = "Sigmorphon 2016 + Japanese Bigger Analogy Test Set"
    # %%
    from utils.ckpt import get_ret_report_path, get_3cos_report_path, get_ret_annc_report_path
    import pandas as pd
    
    language_dataset = [(lang, "2016") for lang in ["turkish", "hungarian", "georgian"]] # "russian", 
    language_dataset += [(lang, "2019") for lang in ["arabic", "english", "french", "portuguese", "bashkir", "slovene", "adyghe", "swahili", "zulu", "slovak", "sanskrit", "welsh", "hebrew"]]

    #for data_seed_id in range(5):
    ret = []
    _3cos = []
    ret_annc = []
    data_seed_id = 0
    for model_seed_id in range(10):
        for language, dataset in language_dataset:
            _3cos.append(pd.read_csv(get_3cos_report_path("mul", dataset, language, model_seed_id)))
            ret.append(pd.read_csv(get_ret_report_path(dataset, language, model_seed_id)))
            ret_annc.append(pd.read_csv(get_ret_annc_report_path(dataset, language, model_seed_id)))

    
    df_3cos = pd.concat(_3cos)
    df_3cos["precision%"] = df_3cos["precision"]*100
    df_3cos.groupby(["dataset", "lang"])["precision%"].mean().apply(lambda x: f"{x:>5.2f}") + " $\pm$ " + df_3cos.groupby(["dataset", "lang"])["precision%"].std().apply(lambda x: f"{x:>5.2f}")

    
    df_ret = pd.concat(ret)
    df_ret["precision%"] = df_ret["precision"]*100
    df_ret.groupby(["dataset", "lang"])["precision%"].mean().apply(lambda x: f"{x:>5.2f}") + " $\pm$ " + df_ret.groupby(["dataset", "lang"])["precision%"].std().apply(lambda x: f"{x:>5.2f}")

    df_ret_annc = pd.concat(ret_annc)
    df_ret_annc["precision%"] = df_ret_annc["precision"]*100
    df_ret_annc.groupby(["dataset", "lang"])["precision%"].mean().apply(lambda x: f"{x:>5.2f}") + " $\pm$ " + df_ret_annc.groupby(["dataset", "lang"])["precision%"].std().apply(lambda x: f"{x:>5.2f}") + df_ret_annc.groupby(["dataset", "lang"])["precision%"].count().apply(lambda x: f" ({x:>2}/10)")

    # %%
    df = df_ret
    result1 = df.groupby(["dataset", "lang"])["precision"].mean().apply(lambda x: f"{x*100:>5.2f}") #+ " $\pm$ " + df.groupby(["dataset", "lang"])["precision"].std().apply(lambda x: f"{x*100:>5.2f}")
    result3 = df.groupby(["dataset", "lang"])["success@3"].mean().apply(lambda x: f"{x*100:>5.2f}") #+ " $\pm$ " + df.groupby(["dataset", "lang"])["success@3"].std().apply(lambda x: f"{x*100:>5.2f}")
    result5 = df.groupby(["dataset", "lang"])["success@5"].mean().apply(lambda x: f"{x*100:>5.2f}") #+ " $\pm$ " + df.groupby(["dataset", "lang"])["success@5"].std().apply(lambda x: f"{x*100:>5.2f}")
    result10 = df.groupby(["dataset", "lang"])["success@10"].mean().apply(lambda x: f"{x*100:>5.2f}") #+ " $\pm$ " + df.groupby(["dataset", "lang"])["success@10"].std().apply(lambda x: f"{x*100:>5.2f}")
    result_ret = pd.concat([result1, result3, result5, result10], axis=1)
    result_ret
    # %%
    df = df_3cos
    result1 = df.groupby(["dataset", "lang"])["precision"].mean().apply(lambda x: f"{x*100:>5.2f}") #+ " $\pm$ " + df.groupby(["dataset", "lang"])["precision"].std().apply(lambda x: f"{x*100:>5.2f}")
    result3 = df.groupby(["dataset", "lang"])["success@3"].mean().apply(lambda x: f"{x*100:>5.2f}") #+ " $\pm$ " + df.groupby(["dataset", "lang"])["success@3"].std().apply(lambda x: f"{x*100:>5.2f}")
    result5 = df.groupby(["dataset", "lang"])["success@5"].mean().apply(lambda x: f"{x*100:>5.2f}") #+ " $\pm$ " + df.groupby(["dataset", "lang"])["success@5"].std().apply(lambda x: f"{x*100:>5.2f}")
    result10 = df.groupby(["dataset", "lang"])["success@10"].mean().apply(lambda x: f"{x*100:>5.2f}") #+ " $\pm$ " + df.groupby(["dataset", "lang"])["success@10"].std().apply(lambda x: f"{x*100:>5.2f}")
    result_3cos = pd.concat([result1, result3, result5, result10], axis=1)
    result_3cos
    # %%
    df = df_ret_annc
    result1 = df.groupby(["dataset", "lang"])["precision"].mean().apply(lambda x: f"{x*100:>5.2f}") #+ " $\pm$ " + df.groupby(["dataset", "lang"])["precision"].std().apply(lambda x: f"{x*100:>5.2f}")
    result3 = df.groupby(["dataset", "lang"])["success@3"].mean().apply(lambda x: f"{x*100:>5.2f}") #+ " $\pm$ " + df.groupby(["dataset", "lang"])["success@3"].std().apply(lambda x: f"{x*100:>5.2f}")
    result5 = df.groupby(["dataset", "lang"])["success@5"].mean().apply(lambda x: f"{x*100:>5.2f}") #+ " $\pm$ " + df.groupby(["dataset", "lang"])["success@5"].std().apply(lambda x: f"{x*100:>5.2f}")
    result10 = df.groupby(["dataset", "lang"])["success@10"].mean().apply(lambda x: f"{x*100:>5.2f}") #+ " $\pm$ " + df.groupby(["dataset", "lang"])["success@10"].std().apply(lambda x: f"{x*100:>5.2f}")
    result_ret_annc = pd.concat([result1, result3, result5, result10], axis=1)
    result_ret_annc

    # %%
    dfs = {
    key:
        pd.concat([
            df.groupby(["dataset", "lang"])[key].mean().apply(lambda x: f"{x*100:>5.2f}") + " $\pm$ " + df.groupby(["dataset", "lang"])[key].std().apply(lambda x: f"{x*100:>5.2f}")
            for df in [df_ret, df_3cos, df_ret_annc]
        ], axis=1, keys=[
            "CNN+ANNr",
            "CNN+3CosMul",
            "CNN+ANNc",
        ])
            for key in ["precision"] + [f"success@{i}" for i in [3,5,10]]
    }
    dfs

    df_set = [df_ret, df_3cos, df_ret_annc]
    df_keys = ["CNN+ANNr", "CNN+3CosMul", "CNN+ANNc",]
    outperf = {
    key:
            df_3cos.groupby(["dataset", "lang"])[key].mean() <
            df_ret_annc.groupby(["dataset", "lang"])[key].mean()
            for key in ["precision"] + [f"success@{i}" for i in [3,5,10]]
    }
    outperf

    # %%
    df_set = [df_ret, df_3cos, df_ret_annc]
    df_keys = ["CNN+ANNr", "CNN+3CosMul", "CNN+ANNc"]
    for df in df_set:
        df["Lang"] = df["lang"].str.capitalize() + (df["lang"].apply(len).max() - df["lang"].apply(len)).apply(lambda x: " "*x) 
    for key in ["precision"] + [f"success@{i}" for i in [3,5,10]]:
        print(key)
        dfs = [
            df.groupby(["dataset", "Lang"])[key].mean().apply(lambda x: f"{x*100:>5.2f}") + " $\pm$ " + df.groupby(["dataset", "Lang"])[key].std().apply(lambda x: f"{x*100:>5.2f}")
            for df in df_set
        ]
        greater_than = [
            [
                df.groupby(["dataset", "Lang"])[key].mean() > df_other.groupby(["dataset", "Lang"])[key].mean() for j, df_other in enumerate(df_set) if i!= j
            ]
            for i, df in enumerate(df_set)
        ]
        n_stars = [
            pd.concat(greater_than_, axis=1, ).sum(axis=1,)
            for greater_than_ in greater_than
        ]
        dfs_with_bold = [
            n_stars[i].apply(lambda x: "\textbf{" if x >= len(dfs) - 1 else " "*8) 
            + dfs[i] + 
            n_stars[i].apply(lambda x: "}" if x >= len(dfs) - 1 else " ")
            for i in range(len(dfs))]
        dfs_with_stars = [dfs_with_bold[i] + " " + n_stars[i].apply(lambda x: "*"*x + " "*(len(dfs) - x)) for i in range(len(dfs))]
        #print(n_stars)

        df = pd.concat(dfs_with_stars, axis=1, keys=[
            "CNN+ANNr",
            "CNN+3CosMul",
            "CNN+ANNc",
        ])
        print(df.to_latex(multicolumn_format='c', escape=False))

    # %%
    outperf = {
    key:
            df_3cos.groupby(["dataset", "lang"])[key].mean().apply(lambda x: f"{x*100:>5.2f}") <
            df_ret_annc.groupby(["dataset", "lang"])[key].mean().apply(lambda x: f"{x*100:>5.2f}")
            for key in ["mrr", "precision"] + [f"success@{i}" for i in [3,5,10]]
    }
    outperf

# %%
"""
2016     georgian      65.00 $\pm$  8.11 ( 2/10)
         hungarian     73.15 $\pm$  7.68 ( 2/10)
         turkish       32.29 $\pm$  0.89 ( 2/10)
2019     adyghe        86.14 $\pm$   nan ( 1/10)
         arabic        34.82 $\pm$   nan ( 1/10)
         bashkir       71.34 $\pm$   nan ( 1/10)
         english       67.32 $\pm$   nan ( 1/10)
         french        37.62 $\pm$   nan ( 1/10)
         hebrew        36.27 $\pm$   nan ( 1/10)
         portuguese    75.74 $\pm$   nan ( 1/10)
         sanskrit      38.88 $\pm$   nan ( 1/10)
         slovak        35.29 $\pm$   nan ( 1/10)
         slovene       47.36 $\pm$   nan ( 1/10)
         swahili       46.94 $\pm$   nan ( 1/10)
         welsh         41.61 $\pm$   nan ( 1/10)
         zulu          48.27 $\pm$   nan ( 1/10)

         
\midrule
2016 & Georgian   &  \textbf{97.60 $\pm$  0.23} **  &          85.58 $\pm$  7.37  *   &          76.77 $\pm$  9.57      \\
     & Hungarian  &  \textbf{89.06 $\pm$  1.71} **  &          74.89 $\pm$  5.27  *   &          72.95 $\pm$  8.02      \\
     & Turkish    &  \textbf{84.75 $\pm$  2.04} **  &          52.42 $\pm$  4.33  *   &          44.43 $\pm$ 13.95      \\
2019 & Adyghe     &  \textbf{93.37 $\pm$  0.97} **  &          58.01 $\pm$  8.66      &          80.11 $\pm$  5.10  *   \\
     & Arabic     &  \textbf{72.08 $\pm$  3.49} **  &          21.67 $\pm$  5.44      &          40.73 $\pm$  9.93  *   \\
     & Bashkir    &          57.63 $\pm$  5.48  *   &          37.76 $\pm$ 11.12      &  \textbf{65.38 $\pm$  6.01} **  \\
     & English    &  \textbf{92.29 $\pm$  0.91} **  &          66.83 $\pm$ 15.67  *   &          65.51 $\pm$  6.25      \\
     & French     &  \textbf{93.15 $\pm$  0.96} **  &          80.37 $\pm$  7.97  *   &          62.87 $\pm$ 17.65      \\
     & Hebrew     &  \textbf{66.52 $\pm$  2.59} **  &          15.47 $\pm$ 10.12      &          36.10 $\pm$  4.28  *   \\
     & Portuguese &  \textbf{93.12 $\pm$  1.10} **  &          58.52 $\pm$ 18.48      &          70.53 $\pm$  9.88  *   \\
     & Sanskrit   &  \textbf{64.18 $\pm$  2.62} **  &          33.20 $\pm$  9.34      &          42.59 $\pm$  4.88  *   \\
     & Slovak     &  \textbf{56.23 $\pm$  4.57} **  &          49.43 $\pm$  3.13  *   &          39.58 $\pm$  3.37      \\
     & Slovene    &  \textbf{71.99 $\pm$  2.24} **  &          57.79 $\pm$  8.59  *   &          51.88 $\pm$  6.69      \\
     & Swahili    &  \textbf{68.56 $\pm$  6.09} **  &          44.84 $\pm$  7.77  *   &          44.46 $\pm$  3.85      \\
     & Welsh      &  \textbf{63.80 $\pm$  3.13} **  &          47.30 $\pm$  4.67      &          47.58 $\pm$  5.72  *   \\
     & Zulu       &  \textbf{76.59 $\pm$  2.65} **  &          58.53 $\pm$  4.42  *   &          42.56 $\pm$  6.55      \\
\bottomrule

Georgian    & \textbf{97.60 $\pm$ 0.23} & 85.58 $\pm$  7.37 &         76.77 $\pm$  9.57  & {87.50$\pm$ 2.08}    \\ 
Hungarian   & \textbf{89.06 $\pm$ 1.71} & 74.89 $\pm$  5.27 &         72.95 $\pm$  8.02  & {90.58$\pm$ 0.63}    \\ 
Turkish     & \textbf{84.75 $\pm$ 2.04} & 52.42 $\pm$  4.33 &         44.43 $\pm$ 13.95  & {79.81$\pm$ 8.58}    \\ 
Adyghe      & \textbf{93.37 $\pm$ 0.97} & 58.01 $\pm$  8.66 &         80.11 $\pm$  5.10  & {98.50 $\pm$ 0.25}   \\ 
Arabic      & \textbf{72.08 $\pm$ 3.49} & 21.67 $\pm$  5.44 &         40.73 $\pm$  9.93  & {83.99 $\pm$ 1.28}   \\ 
Bashkir     &         57.63 $\pm$ 5.48  & 37.76 $\pm$ 11.12 & \textbf{65.38 $\pm$  6.01} & {95.15 $\pm$ 0.92}   \\ 
English     & \textbf{92.29 $\pm$ 0.91} & 66.83 $\pm$ 15.67 &         65.51 $\pm$  6.25  & {92.29 $\pm$ 4.57}   \\ 
French      & \textbf{93.15 $\pm$ 0.96} & 80.37 $\pm$  7.97 &         62.87 $\pm$ 17.65  & {88.25 $\pm$ 2.24}   \\ 
Hebrew      & \textbf{66.52 $\pm$ 2.59} & 15.47 $\pm$ 10.12 &         36.10 $\pm$  4.28  & {92.50 $\pm$ 0.25}   \\ 
Portuguese  & \textbf{93.12 $\pm$ 1.10} & 58.52 $\pm$ 18.48 &         70.53 $\pm$  9.88  & {93.11 $\pm$ 8.32}   \\ 
Sanskrit    & \textbf{64.18 $\pm$ 2.62} & 33.20 $\pm$  9.34 &         42.59 $\pm$  4.88  & {91.48 $\pm$ 0.78}   \\ 
Slovak      & \textbf{56.23 $\pm$ 4.57} & 49.43 $\pm$  3.13 &         39.58 $\pm$  3.37  & {78.90 $\pm$ 0.86}   \\ 
Slovene     & \textbf{71.99 $\pm$ 2.24} & 57.79 $\pm$  8.59 &         51.88 $\pm$  6.69  & {82.41 $\pm$ 10.98}  \\ 
Swahili     & \textbf{68.56 $\pm$ 6.09} & 44.84 $\pm$  7.77 &         44.46 $\pm$  3.85  & {97.49 $\pm$ 0.21}   \\ 
Welsh       & \textbf{63.80 $\pm$ 3.13} & 47.30 $\pm$  4.67 &         47.58 $\pm$  5.72  & {96.79 $\pm$ 0.23}   \\ 
Zulu        & \textbf{76.59 $\pm$ 2.65} & 58.53 $\pm$  4.42 &         42.56 $\pm$  6.55  & {93.42 $\pm$ 0.73}   \\ 



\midrule
\multicolumn{8}{c}{\textit{Sigmorphon 2016}}\\
Georgian    & \textbf{97.60 $\pm$ 0.23} & 85.58 $\pm$  7.37 &         76.77 $\pm$  9.57  & \textbf{87.50$\pm$ 2.08} & \textbf{87.06 $\pm$ 6.28}  & 84.97 & 79.94 \\ 
Hungarian   & \textbf{89.06 $\pm$ 1.71} & 74.89 $\pm$  5.27 &         72.95 $\pm$  8.02  & \textbf{90.58$\pm$ 0.63} & 83.57 $\pm$ 6.63           & 35.24 & 32.07 \\ 
Turkish     & \textbf{84.75 $\pm$ 2.04} & 52.42 $\pm$  4.33 &         44.43 $\pm$ 13.95  & \textbf{79.81$\pm$ 8.58} & \textbf{83.03 $\pm$ 14.07} & 42.09 & 39.45 \\ 
Adyghe      & \textbf{93.37 $\pm$ 0.97} & 58.01 $\pm$  8.66 &         80.11 $\pm$  5.10  & \textbf{98.50 $\pm$ 0.25} & 81.90 $\pm$ 7.35           & 47.94 & 31.25 \\ 
Arabic      & \textbf{72.08 $\pm$ 3.49} & 21.67 $\pm$  5.44 &         40.73 $\pm$  9.93  & \textbf{83.99 $\pm$ 1.28} & 78.40 $\pm$ 12.37          &  2.21 &  3.34 \\ 
Bashkir     &         57.63 $\pm$ 5.48  & 37.76 $\pm$ 11.12 & \textbf{65.38 $\pm$  6.01} & \textbf{95.15 $\pm$ 0.92} & 80.38 $\pm$ 18.55          & 22.29 & 29.89 \\ 
English     & \textbf{92.29 $\pm$ 0.91} & 66.83 $\pm$ 15.67 &         65.51 $\pm$  6.25  & \textbf{92.29 $\pm$ 4.57} & 65.61 $\pm$ 19.96          & 60.15 & 47.69 \\ 
French      & \textbf{93.15 $\pm$ 0.96} & 80.37 $\pm$  7.97 &         62.87 $\pm$ 17.65  & \textbf{88.25 $\pm$ 2.24} & 76.04 $\pm$ 10.20          & 54.48 & 54.39 \\ 
Hebrew      & \textbf{66.52 $\pm$ 2.59} & 15.47 $\pm$ 10.12 &         36.10 $\pm$  4.28  & \textbf{92.50 $\pm$ 0.25} & \textbf{91.16 $\pm$ 7.29}  & 19.50 & 16.17 \\ 
Portuguese  & \textbf{93.12 $\pm$ 1.10} & 58.52 $\pm$ 18.48 &         70.53 $\pm$  9.88  & \textbf{93.11 $\pm$ 8.32} & 62.88 $\pm$ 24.26          & 78.01 & 71.28 \\ 
Sanskrit    & \textbf{64.18 $\pm$ 2.62} & 33.20 $\pm$  9.34 &         42.59 $\pm$  4.88  & \textbf{91.48 $\pm$ 0.78} & 83.83 $\pm$ 5.37           & 42.80 & 28.83 \\ 
Slovak      & \textbf{56.23 $\pm$ 4.57} & 49.43 $\pm$  3.13 &         39.58 $\pm$  3.37  & \textbf{78.90 $\pm$ 0.86} & 82.62 $\pm$ 6.66           & 30.66 & 28.81 \\ 
Slovene     & \textbf{71.99 $\pm$ 2.24} & 57.79 $\pm$  8.59 &         51.88 $\pm$  6.69  & \textbf{82.41 $\pm$ 10.98} & \textbf{80.95 $\pm$ 8.19} &  2.64 &  5.43 \\ 
Swahili     & \textbf{68.56 $\pm$ 6.09} & 44.84 $\pm$  7.77 &         44.46 $\pm$  3.85  & \textbf{97.49 $\pm$ 0.21} & 81.94 $\pm$ 21.80          & 60.23 & 43.02 \\ 
Welsh       & \textbf{63.80 $\pm$ 3.13} & 47.30 $\pm$  4.67 &         47.58 $\pm$  5.72  & \textbf{96.79 $\pm$ 0.23} & 87.62 $\pm$ 12.69          & 14.47 & 19.15 \\ 
Zulu        & \textbf{76.59 $\pm$ 2.65} & 58.53 $\pm$  4.42 &         42.56 $\pm$  6.55  & \textbf{93.42 $\pm$ 0.73} & 81.96 $\pm$ 15.81          & 26.17 & 27.69 \\ 


Georgian    & 97.60 $\pm$ 0.23 & 85.58 $\pm$  7.37 & \textit{ 65.00 $\pm$  8.11 ( 2/10) } & \textbf{87.50$\pm$ 2.08}   & \textbf{87.06 $\pm$ 6.28}  & 84.97 & 79.94 \\ 
Hungarian   & 89.06 $\pm$ 1.71 & 74.89 $\pm$  5.27 & \textit{ 73.15 $\pm$  7.68 ( 2/10) } & \textbf{90.58$\pm$ 0.63}   & 83.57 $\pm$ 6.63           & 35.24 & 32.07 \\ 
Turkish     & 84.75 $\pm$ 2.04 & 52.42 $\pm$  4.33 & \textit{ 32.29 $\pm$  0.89 ( 2/10) } & \textbf{79.81$\pm$ 8.58}   & \textbf{83.03 $\pm$ 14.07} & 42.09 & 39.45 \\ 
Adyghe      & 93.37 $\pm$ 0.97 & 58.01 $\pm$  8.66 & \textit{ 86.14 $\pm$   nan ( 1/10) } & \textbf{98.50 $\pm$ 0.25}  & 81.90 $\pm$ 7.35           & 47.94 & 31.25 \\ 
Arabic      & 72.08 $\pm$ 3.49 & 21.67 $\pm$  5.44 & \textit{ 34.82 $\pm$   nan ( 1/10) } & \textbf{83.99 $\pm$ 1.28}  & 78.40 $\pm$ 12.37          &  2.21 &  3.34 \\ 
Bashkir     & 57.63 $\pm$ 5.48 & 37.76 $\pm$ 11.12 & \textit{ 71.34 $\pm$   nan ( 1/10) } & \textbf{95.15 $\pm$ 0.92}  & 80.38 $\pm$ 18.55          & 22.29 & 29.89 \\ 
English     & 92.29 $\pm$ 0.91 & 66.83 $\pm$ 15.67 & \textit{ 67.32 $\pm$   nan ( 1/10) } & \textbf{92.29 $\pm$ 4.57}  & 65.61 $\pm$ 19.96          & 60.15 & 47.69 \\ 
French      & 93.15 $\pm$ 0.96 & 80.37 $\pm$  7.97 & \textit{ 37.62 $\pm$   nan ( 1/10) } & \textbf{88.25 $\pm$ 2.24}  & 76.04 $\pm$ 10.20          & 54.48 & 54.39 \\ 
Hebrew      & 66.52 $\pm$ 2.59 & 15.47 $\pm$ 10.12 & \textit{ 36.27 $\pm$   nan ( 1/10) } & \textbf{92.50 $\pm$ 0.25}  & \textbf{91.16 $\pm$ 7.29}  & 19.50 & 16.17 \\ 
Portuguese  & 93.12 $\pm$ 1.10 & 58.52 $\pm$ 18.48 & \textit{ 75.74 $\pm$   nan ( 1/10) } & \textbf{93.11 $\pm$ 8.32}  & 62.88 $\pm$ 24.26          & 78.01 & 71.28 \\ 
Sanskrit    & 64.18 $\pm$ 2.62 & 33.20 $\pm$  9.34 & \textit{ 38.88 $\pm$   nan ( 1/10) } & \textbf{91.48 $\pm$ 0.78}  & 83.83 $\pm$ 5.37           & 42.80 & 28.83 \\ 
Slovak      & 56.23 $\pm$ 4.57 & 49.43 $\pm$  3.13 & \textit{ 35.29 $\pm$   nan ( 1/10) } & \textbf{78.90 $\pm$ 0.86}  & 82.62 $\pm$ 6.66           & 30.66 & 28.81 \\ 
Slovene     & 71.99 $\pm$ 2.24 & 57.79 $\pm$  8.59 & \textit{ 47.36 $\pm$   nan ( 1/10) } & \textbf{82.41 $\pm$ 10.98} & \textbf{80.95 $\pm$ 8.19}  &  2.64 &  5.43 \\ 
Swahili     & 68.56 $\pm$ 6.09 & 44.84 $\pm$  7.77 & \textit{ 46.94 $\pm$   nan ( 1/10) } & \textbf{97.49 $\pm$ 0.21}  & 81.94 $\pm$ 21.80          & 60.23 & 43.02 \\ 
Welsh       & 63.80 $\pm$ 3.13 & 47.30 $\pm$  4.67 & \textit{ 41.61 $\pm$   nan ( 1/10) } & \textbf{96.79 $\pm$ 0.23}  & 87.62 $\pm$ 12.69          & 14.47 & 19.15 \\ 
Zulu        & 76.59 $\pm$ 2.65 & 58.53 $\pm$  4.42 & \textit{ 48.27 $\pm$   nan ( 1/10) } & \textbf{93.42 $\pm$ 0.73}  & 81.96 $\pm$ 15.81          & 26.17 & 27.69 \\ 

\begin{tabular}{llllll}
\toprule
       &                 F1 &  Balanced accuracy &                TPR &                TNR \\
 Model &                    &                    &                    &                    \\
\midrule
 GloVe6B100 &  84.56 $\pm$  1.27 &  91.26 $\pm$  0.69 &  94.51 $\pm$  0.55 &  88.02 $\pm$  1.42 \\
 GloVe6B200 &  84.03 $\pm$  0.46 &  90.95 $\pm$  0.28 &  94.36 $\pm$  0.55 &  87.54 $\pm$  0.62 \\
 GloVe6B300 &  82.50 $\pm$  0.42 &  90.09 $\pm$  0.47 &  94.35 $\pm$  2.52 &  85.83 $\pm$  1.73 \\
 GloVe6B50 &  81.70 $\pm$  1.43 &  89.81 $\pm$  0.80 &  95.30 $\pm$  0.90 &  84.32 $\pm$  1.88 \\
 GloVe6B100+ &  98.37 $\pm$  0.17 &  99.06 $\pm$  0.11 &  99.11 $\pm$  0.30 &  99.01 $\pm$  0.19 \\
 GloVe6B200+ &  98.76 $\pm$  0.15 &  99.34 $\pm$  0.10 &  99.51 $\pm$  0.18 &  99.17 $\pm$  0.12 \\
 GloVe6B300+ &  98.83 $\pm$  0.15 &  99.36 $\pm$  0.09 &  99.47 $\pm$  0.19 &  99.25 $\pm$  0.14 \\
 GloVe6B50+ &  94.83 $\pm$  0.86 &  97.15 $\pm$  0.47 &  97.77 $\pm$  0.52 &  96.52 $\pm$  0.68 \\
\bottomrule
\end{tabular}

\begin{tabular}{llllll}
\toprule
       &                 F1 &  Balanced accuracy &                TPR &                TNR \\
 Model &                    &                    &                    &                    \\
\midrule
 GloVe6B100 &  52.82 $\pm$  0.21 &  67.48 $\pm$  0.18 &  60.05 $\pm$  1.66 &  74.90 $\pm$  1.78 \\
 GloVe6B200 &  52.88 $\pm$  0.33 &  67.52 $\pm$  0.25 &  60.57 $\pm$  1.96 &  74.47 $\pm$  1.92 \\
 GloVe6B300 &  52.70 $\pm$  0.24 &  67.38 $\pm$  0.21 &  60.07 $\pm$  1.57 &  74.68 $\pm$  1.86 \\
 GloVe6B50 &  52.61 $\pm$  0.21 &  67.30 $\pm$  0.19 &  60.43 $\pm$  2.33 &  74.17 $\pm$  2.55 \\
 GloVe6B100+ &  55.20 $\pm$  0.25 &  69.31 $\pm$  0.19 &  60.89 $\pm$  1.10 &  77.74 $\pm$  0.89 \\
 GloVe6B200+ &  55.54 $\pm$  0.34 &  69.59 $\pm$  0.24 &  62.46 $\pm$  1.53 &  76.72 $\pm$  1.78 \\
 GloVe6B300+ &  55.96 $\pm$  0.24 &  69.92 $\pm$  0.18 &  63.03 $\pm$  1.71 &  76.81 $\pm$  1.66 \\
 GloVe6B50+ &  54.38 $\pm$  0.34 &  68.70 $\pm$  0.26 &  59.82 $\pm$  1.37 &  77.57 $\pm$  1.47 \\
\bottomrule
\end{tabular}
"""
# %%
if False:
    # %%
    from utils.ckpt import get_clf_report_path
    import pandas as pd
    
    language_dataset = [(lang, "2016") for lang in ["turkish", "hungarian", "georgian"]] # "russian", 
    language_dataset += [(lang, "2019") for lang in ["arabic", "english", "french", "portuguese", "bashkir", "slovene", "adyghe", "swahili", "zulu", "slovak", "sanskrit", "welsh", "hebrew"]]

    #for data_seed_id in range(5):
    clf = []
    data_seed_id = 0
    for model_seed_id in range(10):
        for language, dataset in language_dataset:
            clf.append(pd.read_csv(get_clf_report_path(dataset, language, model_seed_id)))
    
    df_clf = pd.concat(clf)
    # %%
    df_clf["F1%"] = df_clf["F1"]*100
    df_clf.groupby(["dataset", "lang"])["F1%"].mean().apply(lambda x: f"{x:>5.2f}") + " $\pm$ " + df_clf.groupby(["dataset", "lang"])["F1%"].std().apply(lambda x: f"{x:>5.2f}")
    # %%
    df_clf["balacc%"] = df_clf["balacc"]*100
    df_clf.groupby(["dataset", "lang"])["balacc%"].mean().apply(lambda x: f"{x:>5.2f}") + " $\pm$ " + df_clf.groupby(["dataset", "lang"])["balacc%"].std().apply(lambda x: f"{x:>5.2f}")
    # %%
    df_clf["Sigmorphon"] = df_clf["dataset"]
    df_clf["Balanced accuracy"] = df_clf["balacc"]
    df_clf["Language"] = df_clf["lang"].str.capitalize() + (df_clf["lang"].apply(len).max() - df_clf["lang"].apply(len)).apply(lambda x: " "*x) 
    keys = ["F1", "Balanced accuracy", "TPR", "TNR"]
    df = pd.concat([
            df_clf.groupby(["Sigmorphon", "Language"])[key].mean().apply(lambda x: f"{x*100:>5.2f}") + " $\pm$ " + df_clf.groupby(["Sigmorphon", "Language"])[key].std().apply(lambda x: f"{x*100:>5.2f}")
            for key in keys
        ], axis=1, keys=keys)
    print(df.to_latex(multicolumn_format='c', escape=False))
    df
# %%

# %%
if False:
    # %%
    from sem_baseline.train_clf_glove_fasttext import get_clf_report_path, MODELS
    import pandas as pd

    #for data_seed_id in range(5):
    clf = []
    data_seed_id = 0
    for model_seed_id in range(10):
        for model_ in MODELS.keys():
            model, emb_size = MODELS[model_]
            if model == "CNN+ANNc": continue
            for post_emb in [True,False]:
                clf.append(pd.read_csv(get_clf_report_path(seed_id=model_seed_id, covered=False, post_emb=post_emb, model=model, emb_size=emb_size)))
                clf[-1]["post_emb"] = post_emb
                clf[-1]["covered_analogies_only"] = False
                clf[-1]["Model"] = model_ + ("+" if post_emb else "")
    for model_seed_id in range(5):
        for model_ in MODELS.keys():
            model, emb_size = MODELS[model_]
            if model == "CNN+ANNc": continue
            for post_emb in [True,False]:
                clf.append(pd.read_csv(get_clf_report_path(seed_id=model_seed_id, covered=True, post_emb=post_emb, model=model, emb_size=emb_size)))
                clf[-1]["post_emb"] = post_emb
                clf[-1]["Model"] = model_ + ("+" if post_emb else "")
    
    df_clf = pd.concat(clf)
    # %%
    group_keys = ["covered_analogies_only", "post_emb", "Model"]
    df_clf["F1%"] = df_clf["F1"]*100
    df_clf.groupby(group_keys)["F1%"].mean().apply(lambda x: f"{x:>5.2f}") + " $\pm$ " + df_clf.groupby(group_keys)["F1%"].std().apply(lambda x: f"{x:>5.2f}")
    # 
    df_clf["balacc%"] = df_clf["balacc"]*100
    df_clf.groupby(group_keys)["balacc%"].mean().apply(lambda x: f"{x:>5.2f}") + " $\pm$ " + df_clf.groupby(group_keys)["balacc%"].std().apply(lambda x: f"{x:>5.2f}")
    # 
    df_clf["Balanced accuracy"] = df_clf["balacc"]
    keys = ["F1", "Balanced accuracy", "TPR", "TNR"]
    group_keys = ["post_emb", "Model"]
    for df_ in [df_clf[df_clf["covered_analogies_only"]], df_clf[~df_clf["covered_analogies_only"]]]:
        df = pd.concat([
                df_.groupby(group_keys)[key].mean().apply(lambda x: f"{x*100:>5.2f}") + " $\pm$ " + df_.groupby(group_keys)[key].std().apply(lambda x: f"{x*100:>5.2f}")
                for key in keys
            ], axis=1, keys=keys)
        print(df.to_latex(multicolumn_format='c', escape=False))
    #df
# %%
