# File: correlation_with_description_analyse.py
# Author: ZHANG, Ziyang
# Student ID: 21266920
# Email: zzhangmc@connect.ust.hk
# Date: 2025-09-30
# Description: The analysing and plotting code

import numpy as np
import pandas as pd
import textstat
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr

marketing_words = {"best", "amazing", "perfect", "ultimate", "limited",
                   "exclusive", "must-have", "super", "top",
                   "incredible", "love", "enjoy", "happy"}


def get_marketing_score(text):
    words = text.lower().split()
    count = sum(1 for w in words if w in marketing_words)
    return count / (len(words) + 1) * 100


def get_spearmanr_conclusion(corr: float, p_value: float) -> str:
    if p_value > 0.001:
        return "No significant correlation"
    elif abs(corr) > 0.35:
        return "Highly correlated"
    elif abs(corr) > 0.20:
        return "Correlated"
    elif abs(corr) > 0.16:
        return "Slightly correlated"
    else:
        return "No significant correlation"


def add_spearmanr_conclusion(df: pd.DataFrame) -> pd.DataFrame:
    df['conclusion'] = df.apply(lambda x: get_spearmanr_conclusion(x['corr'], x['p_value']), axis=1)
    return df


def draw_product_score_distribution(df_amazon_product_info: pd.DataFrame):
    tmp = df_amazon_product_info[df_amazon_product_info.NumRatings > 20].copy()

    ax = tmp.plot.scatter(x='Score', y='ScorePolarizationIndex', alpha=0.06, figsize=(10, 5), color='#F28A63')
    ax.set_title("Distribution of Score & Score Polarization", fontweight='bold', fontsize=12)

    ax.axvline(x=4.5, color='grey', linestyle=':', linewidth=2, alpha=0.4)
    ax.axhline(y=0.25, color='grey', linestyle=':', linewidth=2, alpha=0.4)

    ax2 = ax.twinx()
    ax2.set_ylabel('Number of Samples')
    tmp['bin_sc'] = pd.cut(tmp.Score, bins=int((tmp.Score.max() - tmp.Score.min()) * 10 + 1))
    bin_sc = tmp.groupby("bin_sc", observed=False).agg(value=('Score', 'count'))
    g2 = ax2.bar(np.arange(tmp.Score.min(), tmp.Score.max() + 0.1, 0.1), bin_sc.value, width=0.05, color='#B8D38F',
                 label='Distribution of Score')
    ax2.set_ylim(0, 2000)
    # ax2.legend(title='Stocks', loc='upper left')

    ax3 = ax.twiny()
    ax3.set_xlabel('Number of Samples')
    tmp['bin_scpi'] = pd.cut(tmp.ScorePolarizationIndex, bins=50)
    bin_scpi = tmp.groupby("bin_scpi", observed=False).agg(value=('ScorePolarizationIndex', 'count'))
    g3 = ax3.barh([e.mid for e in bin_scpi.index], bin_scpi.value, height=0.006, color='#E0E0A0',
                  label='Distribution of Score Polarization')
    ax3.set_xlim(0, 2000)

    ax.legend([g2, g3], [g2.get_label(), g3.get_label()], loc='upper left')

    return ax


def draw_correlation_with_description_length(df_amazon_product_info: pd.DataFrame):
    tmp = df_amazon_product_info[df_amazon_product_info.NumRatings > 10][
        ['Score', 'ScorePolarizationIndex', 'product_description']].copy().dropna()
    tmp['CountDescLength'] = np.log2(tmp.product_description.apply(lambda x: len(x.split(' '))))
    tmp['CountDescLengthGroup'] = pd.cut(tmp.CountDescLength, bins=15)

    corr_s, p_value_s = spearmanr(tmp["Score"], tmp["CountDescLength"])
    # print(f"Spearman correlation coefficient: {corr:.4f}; P value: {p_value:.1e}")

    corr_p, p_value_p = spearmanr(tmp["ScorePolarizationIndex"], tmp["CountDescLength"])
    # print(f"Spearman correlation coefficient: {corr:.4f}; P value: {p_value:.1e}")

    df = tmp.groupby("CountDescLengthGroup", observed=False).agg(
        n_samples=("Score", "count"),
        good_score_proportion=("Score", lambda x: (x > 4.5).mean()),
        low_polarity_proportion=("ScorePolarizationIndex", lambda x: (x < 0.25).mean()))
    df.index = [e.mid for e in df.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Description Length & Good percentage", fontweight='bold', fontsize=12)
    ax.set_xlabel("Description Text Length (log2)")
    ax2 = ax.twinx()

    ax.plot(df.index, df[['good_score_proportion', 'low_polarity_proportion']],
            label=['good_score_proportion', 'low_polarity_proportion'])
    ax.set_ylabel("Good Percentage")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    ax2.bar(df.index, df['n_samples'], width=0.5, label='bar', color="skyblue")
    ax2.set_ylabel("Number of Samples")

    ax.annotate(f"Spearman Correlation\n"
                f"Score       : {get_spearmanr_conclusion(corr_s, p_value_s)} \n"
                f"            - Corr: {corr_s:6.3f}, P: {p_value_s:.1e}\n"
                f"Polarization: {get_spearmanr_conclusion(corr_p, p_value_p)}\n"
                f"            - Corr: {corr_p:6.3f}, P: {p_value_p:.1e}",
                xy=(1.5, 0.5), xytext=(10, 10), textcoords='offset points', fontfamily="monospace", alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1, alpha=0.4), )

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend()


def draw_correlation_with_n_images(df_amazon_product_info: pd.DataFrame):
    tmp = df_amazon_product_info[['Score', 'ScorePolarizationIndex', 'CountAltImages']].dropna()

    corr_s, p_value_s = spearmanr(tmp["Score"], tmp["CountAltImages"])
    # print(f"Spearman correlation coefficient: {corr:.4f}; P value: {p_value:.1e}")

    corr_p, p_value_p = spearmanr(tmp["ScorePolarizationIndex"], tmp["CountAltImages"])
    # print(f"Spearman correlation coefficient: {corr:.4f}; P value: {p_value:.1e}")

    df = df_amazon_product_info[df_amazon_product_info.NumRatings > 10].groupby("CountAltImages").agg(
        n_samples=("Score", "count"),
        good_score_proportion=("Score", lambda x: (x > 4.5).mean()),
        low_polarity_proportion=("ScorePolarizationIndex", lambda x: (x < 0.25).mean()))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Number of Sample Images & Good percentage", fontweight='bold', fontsize=12)
    ax.set_xlabel("Number of sample images")
    ax2 = ax.twinx()

    ax.plot(df.index, df[['good_score_proportion', 'low_polarity_proportion']],
            label=['good_score_proportion', 'low_polarity_proportion'])
    ax.set_ylabel("Good Percentage")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    ax2.bar(df.index, df['n_samples'], width=0.6, label='bar', color="skyblue")
    ax2.set_ylabel("Number of Samples")

    ax.annotate(f"Spearman Correlation\n"
                f"Score       : {get_spearmanr_conclusion(corr_s, p_value_s)} \n"
                f"            - Corr: {corr_s:6.3f}, P: {p_value_s:.1e}\n"
                f"Polarization: {get_spearmanr_conclusion(corr_p, p_value_p)}\n"
                f"            - Corr: {corr_p:6.3f}, P: {p_value_p:.1e}",
                xy=(0.5, 0.33), xytext=(10, 10), textcoords='offset points', fontfamily="monospace", alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1, alpha=0.4))

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend()

    return ax


def get_correlation_with_marketing_sentiment(df_amazon_product_info: pd.DataFrame):
    tmp = df_amazon_product_info[list(
        filter(lambda x: x in df_amazon_product_info,
               ['Score', 'ScorePolarizationIndex', 'product_description', 'sentiment_score']))].dropna()
    tmp['marketing_tone_score'] = tmp.product_description.apply(get_marketing_score)
    available_keys = list(filter(lambda x: x in tmp, ['sentiment_score', 'marketing_tone_score']))
    result = pd.DataFrame(
        [dict(sentiment_type=key, compare_with=s, **dict(zip(('corr', 'p_value'), spearmanr(tmp[s], tmp[key]))))
         for s in ['Score', 'ScorePolarizationIndex']
         for key in available_keys]).sort_values(
        ['sentiment_type', 'compare_with', 'corr'])

    fig, axes = plt.subplots(1, len(available_keys), figsize=(len(available_keys) * 4, 2))
    axes = axes if (isinstance(axes, list) or isinstance(axes, np.ndarray)) else [axes]
    for i, key in enumerate(available_keys):
        axes[i].hist(tmp[key], bins=25, color=f'C{i}', width=(tmp[key].max() - tmp[key].min()) / 35)
        axes[i].set_title(key + ' (log)')
        axes[i].set_yscale('log')

    fig.suptitle("Distribution of marketing sentiment index", fontsize=12, fontweight='bold')
    fig.tight_layout()

    return tmp, add_spearmanr_conclusion(result)


def get_correlation_with_reading_ease(df_amazon_product_info: pd.DataFrame):
    tmp = df_amazon_product_info[['Score', 'product_description', 'ScorePolarizationIndex']].copy().dropna()
    tmp['flesch_reading_ease'] = tmp['product_description'].apply(textstat.flesch_reading_ease)
    tmp['flesch_kincaid_grade'] = tmp['product_description'].apply(textstat.flesch_kincaid_grade)
    tmp['gunning_fog'] = tmp['product_description'].apply(textstat.gunning_fog)
    tmp = tmp[(tmp.flesch_reading_ease != 0) & (tmp.flesch_reading_ease != 0) & (tmp.flesch_reading_ease != 0)]
    result = pd.DataFrame(
        [dict(ease_index=k, compare_with=s, **dict(zip(('corr', 'p_value'), spearmanr(tmp[s], tmp[k]))))
         for k in ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog']
         for s in ['Score', 'ScorePolarizationIndex']])

    fig, axes = plt.subplots(1, 3, figsize=(12, 2))
    axes[0].hist(tmp.flesch_reading_ease, bins=25, color='C0',
                 width=(tmp.flesch_reading_ease.max() - tmp.flesch_reading_ease.min()) / 35)
    axes[0].set_title("flesch_reading_ease")
    axes[1].hist(tmp.flesch_kincaid_grade, bins=25, color='C1',
                 width=(tmp.flesch_kincaid_grade.max() - tmp.flesch_kincaid_grade.min()) / 35)
    axes[1].set_title("flesch_kincaid_grade")
    axes[2].hist(tmp.gunning_fog, bins=25, color='C2', width=(tmp.gunning_fog.max() - tmp.gunning_fog.min()) / 35)
    axes[2].set_title("gunning_fog")
    fig.suptitle("Distribution of reading ease index", fontsize=12, fontweight='bold')
    fig.tight_layout()

    return fig, add_spearmanr_conclusion(result)


def get_correlation_with_description_item(df_amazon_product_info: pd.DataFrame) -> pd.DataFrame:
    tmp = df_amazon_product_info[['product_detail', 'important_information', 'Score', 'ScorePolarizationIndex']].copy()
    detail_keys = tmp.product_detail.apply(
        lambda x: [e for e in x.keys() if not e.startswith('#')]).explode().value_counts()
    info_keys = tmp.important_information.apply(
        lambda x: [e for e in x.keys() if not e.startswith('#')]).explode().value_counts()
    detail_keys = detail_keys[detail_keys.values > 10]
    data = []
    for cat, (key, n) in [*zip(['product_detail'] * len(detail_keys), detail_keys.items()),
                          *zip(['important_information'] * len(detail_keys), info_keys.items())]:
        tmp[key] = tmp[cat].apply(lambda x: 1 if x.get(key) else -1)
        for cmp in ['Score', 'ScorePolarizationIndex']:
            corr, p_value = spearmanr(tmp.dropna()[cmp], tmp.dropna()[key])
            data.append({"item": key, "category": cat.split('_')[-1],
                         "num_samples": n, 'compare_with': cmp, "corr": corr, "p_value": p_value})
    result = pd.DataFrame(data).sort_values(['item', 'compare_with', 'corr'])
    return add_spearmanr_conclusion(result)
