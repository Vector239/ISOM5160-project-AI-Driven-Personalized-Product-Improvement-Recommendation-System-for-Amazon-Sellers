# File: data_cleaning.py
# Author: CAO, Xi
# Student ID: 21271664
# Email: @connect.ust.hk
# Date:
# Description: The data cleaning code

import pandas as pd
import re
import nltk
import torch
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('punkt', quiet=True)  # 显示下载过程，确保资源成功获取
nltk.download('stopwords', quiet=True)  # 下载停用词资源


def data_cleaning(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 查看数据的基本信息
    print('数据基本信息：')
    print(df.info())
    # 删除 Text、Score 和 ProductId 列中存在缺失值的行
    original_rows = df.shape[0]
    df = df.dropna(subset=['Text', 'Score', 'ProductId'])
    rows_deleted = original_rows - df.shape[0]
    print(f"已删除 {rows_deleted} 行含有缺失值的数据")

    # 删除重复值(ProductId, UserId, Score 三列完全相同的行,只保留第一个)
    original_rows = df.shape[0]
    df = df.drop_duplicates(
        subset=['ProductId', 'UserId', 'Score'], keep='first')
    print(f"已删除 {original_rows - df.shape[0]} 行重复值数据")

    # 处理 HelpfulnessDenominator 和 HelpfulnessNumerator 列
    numerator_pos = df.columns.get_loc('HelpfulnessNumerator')
    df['Helpfulness index'] = df.apply(
        lambda row: round(row['HelpfulnessNumerator'] /
                          row['HelpfulnessDenominator'], 2)
        if row['HelpfulnessDenominator'] != 0 else 0, axis=1
    )
    # 将新列插入到原HelpfulnessNumerator的位置
    df.insert(numerator_pos, 'Helpfulness index',
              df.pop('Helpfulness index'))
    # 删除原来的两列
    df = df.drop(columns=['HelpfulnessNumerator', 'HelpfulnessDenominator'])

    # 处理text列(提取关键词，去除多余空格)
    df['Text_cleaned'] = df['Text'].str.lower()
    df['Text_cleaned'] = df['Text_cleaned'].apply(
        lambda x: re.sub(r'[^\w\s]', '', x) if pd.notna(x) else x
    )
    # 删除重复空格
    df['Text_cleaned'] = df['Text_cleaned'].apply(
        # strip()去除首尾空格
        lambda x: re.sub(r'\s+', ' ', x).strip() if pd.notna(x) else x
    )
    # 分词
    df['tokens'] = df['Text_cleaned'].apply(
        lambda x: word_tokenize(x) if pd.notna(x) and x != '' else []
    )
    # 5.2 提取关键词（直接从清洗后的文本生成，跳过tokens中间列）
    stop_words = set(stopwords.words('english'))
    df['keywords'] = df['Text_cleaned'].apply(
        lambda x: [word for word in word_tokenize(x) if word not in stop_words]
        if (pd.notna(x) and x != '') else []  # 空值/空文本返回空列表
    )
    # 5.3 删除原始文本列和清洗后的文本列
    df = df.drop(columns=['Text', 'tokens'])
    return df


def add_sentiment_score(df, text_col_name):
    """情感分析函数：仅添加情感分数（基于预训练模型）"""
    # 加载预训练模型和分词器（轻量级情感分析模型）
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model_cache_dir = "./sentiment_model_cache"  # 本地缓存目录
    # 如果本地有缓存模型则加载本地模型
    if os.path.exists(model_cache_dir) and len(os.listdir(model_cache_dir)) > 0:
        print(f"发现本地缓存模型，从 {model_cache_dir} 加载...")
        # 下载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(model_cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cache_dir)
    else:
        print("本地无缓存模型，正在下载并缓存...")
        # 下载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # 创建缓存目录并保存模型
        os.makedirs(model_cache_dir, exist_ok=True)
        tokenizer.save_pretrained(model_cache_dir)
        model.save_pretrained(model_cache_dir)
        print(f"模型已缓存至 {model_cache_dir}")

    # 如果GPU可用则使用GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # print("GPU 可用，正在使用 GPU 进行计算...")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # print("MPS 可用，正在使用 Metal Performance Shaders 进行计算...")
    else:
        device = torch.device("cpu")
        # print("GPU 不可用，正在使用 CPU 进行计算...")

    model.to(device)
    # 存储情感分数的列表
    sentiment_scores = []
    total = len(df)
    # 遍历文本列计算情感分数（仅保留正面情感概率作为分数）
    for idx, text in enumerate(tqdm(df[text_col_name], desc=f"Adding sentiment score (using {device.type})")):
        # if idx %1000 == 0 or idx == total - 1:
        #     print(f"Processing row {idx + 1}/{total}...  ")
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        positive_score = probabilities[0][1].item()
        sentiment_scores.append(positive_score)

    df['sentiment_score'] = sentiment_scores
    print("情感分数计算完成")
    return df


def main():
    print("hello")
    cleaned_data = data_cleaning('datasets/amazon_food_reviews.csv')
    cleaned_data.to_csv(
        'datasets/amazon_food_reviews_cleaned.csv', index=False)

    print("Adding sentiment score")
    # 2. 添加情感分数列
    result_df = add_sentiment_score(cleaned_data, 'Text_cleaned')
    # 按相对路径将结果保存到 /data/ 文件夹下
    result_df.to_csv('datasets/amazon_food_reviews_cleaned.csv', index=False)


if __name__ == "__main__":
    main()
