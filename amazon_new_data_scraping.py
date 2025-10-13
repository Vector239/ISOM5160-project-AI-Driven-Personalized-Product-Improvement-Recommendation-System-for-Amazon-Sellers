# File: amazon_new_data_scraping.py
# Author: ZHANG, Ziyang
# Student ID: 21266920
# Email: zzhangmc@connect.ust.hk
# Date: 2025-09-13
# Description: Scrape the new data from the Amazon website

import argparse
import json
import multiprocessing as mp
import os
from glob import glob

import numpy as np
import pandas as pd
import regex
import requests
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm

_amazon_headers_ = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "cache-control": "max-age=0",
    "Connection": "keep-alive",
    "Referer": "https://www.amazon.com/",
    "viewport-width": "1080",
}

new_data_path = "datasets/new_data"


def handle_none_value(value, handle_none, handle_not_none):
    return handle_none(value) if value is None else handle_not_none(value)


def parse_amazon_product_info(soup: BeautifulSoup) -> dict:
    product_title = soup.find('span', id='productTitle')
    if product_title is None:
        return {}
    product_title = product_title.text.strip()
    byline_info = handle_none_value(soup.find('a', id='bylineInfo'),
                                    lambda x: '', lambda x: x.text.strip())
    product_description = handle_none_value(soup.find('div', id='productDescription'),
                                            lambda x: '', lambda x: x.text.strip())

    category = soup.find('div', id='wayfinding-breadcrumbs_feature_div')
    category = [e.strip() for e in category.text.split('›')] if category is not None else []

    alt_images_prefix = 'https://m.media-amazon.com/images/I/'
    alt_images = soup.find('div', id='altImages')
    if alt_images is not None:
        alt_images = [e['src'] for e in alt_images.find_all('img')]
        alt_images = [f"{alt_images_prefix}{e.split('/')[-1].split('.')[0]}.jpg" for e in
                      filter(lambda x: x.startswith(alt_images_prefix) and x.endswith('jpg'), alt_images)]

    product_detail = soup.find('div', id='detailBullets_feature_div')
    if product_detail is not None:
        product_detail = product_detail.find_all('li')
        product_detail = [regex.sub(r"[\s\p{C}]+", " ", e.text).strip() for e in product_detail]
        product_detail = {e.split(':')[0].strip(): ':'.join([a.strip() for a in e.split(':')[1:]]) for e in
                          product_detail}

    important_information = soup.find('div', id='important-information')
    if important_information is not None:
        important_information = important_information.find_all('div', class_='a-section content')
        important_information = {e.span.text.strip(): ''.join([p.text for p in e.find_all('p')]).strip() for e in
                                 important_information if e.span is not None}

    rating = soup.find('div', id='cm_cr_dp_d_rating_histogram')
    if rating is not None:
        try:
            score = rating.find('span', class_='a-size-medium a-color-base').text
            dist = {
                ''.join(filter(lambda x: isinstance(x, NavigableString),
                               e.find('div',
                                      class_='a-section a-spacing-none a-text-left aok-nowrap').contents)).strip():
                    ''.join(filter(lambda x: isinstance(x, NavigableString),
                                   e.find('div',
                                          class_='a-section a-spacing-none a-text-right aok-nowrap').contents)).strip()
                for e in rating.find('ul', id='histogramTable').find_all('li')
            }
            rating = dict(score=score, dist=dist)
        except Exception as e:
            rating = None

    top_comments = soup.find('ul', id='cm-cr-dp-review-list')
    if top_comments is None:
        top_comments = []
    else:
        top_comments = [
            dict(
                date=handle_none_value(each.find('span', class_='review-date'),
                                       lambda x: "", lambda x: x.text.split('on')[-1].strip()),
                title=each.find('a', class_='review-title-content').find_all('span')[-1].text.strip(),
                score=each.find('a', class_='review-title-content').find('span', class_="a-icon-alt").text.strip(),
                text=handle_none_value(each.find('div', class_='review-text-content'),
                                       lambda x: "", lambda x: x.text.strip()),
                helpfulness=handle_none_value(each.find('span', class_='cr-vote-text'),
                                              lambda x: "0", lambda x: x.text.strip().split(' ')[0]),
            )
            for each in top_comments.find_all('li')]

    return dict(
        product_title=product_title,
        byline_info=byline_info,
        product_description=product_description,
        category=category,
        alt_images=alt_images if alt_images is not None else [],
        product_detail=product_detail if product_detail is not None else {},
        important_information=important_information if important_information is not None else {},
        rating=rating if type(rating) is dict else None,
        top_comments=top_comments,
    )


def scrape_data_with_product_id(product_id: str) -> dict:
    url = f"https://www.amazon.com/dp/{product_id}?language=en_US&currency=USD"
    r = requests.get(url, headers=_amazon_headers_)
    soup = BeautifulSoup(r.text, "html.parser")
    return parse_amazon_product_info(soup)


def scrape_data_to_json(product_id: str, replace: bool = False) -> bool:
    output_file = os.path.join(new_data_path, f"{product_id}.json")
    if os.path.exists(output_file) and not replace:
        return True
    try:
        data = scrape_data_with_product_id(product_id)
    except Exception as e:
        print(f"Error scraping product {product_id}: {str(e)}")
        return False
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error writing to file {output_file}: {str(e)}")
        return False
    return True


def parallel_scrape_amazon_product_info(product_id_list: list, replace: bool = False):
    cpu_count = mp.cpu_count()
    print(f"Using {cpu_count} CPU cores for parallel scraping")
    # Create a folder for new data
    os.makedirs(new_data_path, exist_ok=True)
    with mp.Pool(cpu_count) as pool:
        n_success = sum([e for e in tqdm(pool.imap_unordered(scrape_data_to_json, product_id_list),
                                         total=len(product_id_list), desc="Scraping amazon products")])
        print(f"Successfully scraped {n_success}/{len(product_id_list)} products")


def scrape_main():
    parser = argparse.ArgumentParser(description="Amazon Product Data Scraper for ISOM5160")
    parser.add_argument("--replace", type=bool, required=False, help="If set, scrape will replace the existing data")
    args = parser.parse_args()

    try:
        amazon_food_reviews = pd.read_csv('datasets/amazon_food_reviews.csv')
        product_id_list = list(amazon_food_reviews.ProductId.unique())
    except Exception:
        print('Error loading "amazon_food_reviews.csv", please put it in the same folder as this script.')
        exit(-1)

    print(f"Total number of products: {len(product_id_list)}")
    print("Start scraping data...")
    parallel_scrape_amazon_product_info(product_id_list, args.replace)
    print("Scraping finished.")


def esteban_ray_index(p, alpha=1.3):
    if p is None:
        return np.nan
    scores, p = np.arange(1, len(p) + 1), np.array(p) / sum(p)
    er = 0.0
    for i in range(len(p)):
        for j in range(len(p)):
            er += (p[i] ** (1 + alpha)) * p[j] * abs(scores[i] - scores[j])
    return er


def add_custom_columns(df_amazon_product_info: pd.DataFrame):
    df_amazon_product_info['CountAltImages'] = df_amazon_product_info.alt_images.apply(lambda x: len(x))
    df_amazon_product_info['Score'] = df_amazon_product_info.rating.apply(
        lambda x: float(x['score'].split(' ')[0]) if x is not None else np.nan)
    df_amazon_product_info['ScoreDistribution'] = df_amazon_product_info.rating.apply(lambda x: np.array(
        [float(x['dist'][k][:-1]) for k in
         ['1 star', '2 star', '3 star', '4 star', '5 star']] if x is not None else None))
    df_amazon_product_info['ScorePolarizationIndex'] = df_amazon_product_info.ScoreDistribution.apply(esteban_ray_index)
    df_amazon_product_info['NumRatings'] = df_amazon_product_info.product_detail.apply(
        lambda x: int(x.get('Customer Reviews').split(' ')[-2].replace(',', ''))
        if x.get('Customer Reviews') is not None else -1)
    df_amazon_product_info['IsFood'] = df_amazon_product_info.category.apply(
        lambda x: sum(['food' in e.lower() for e in x])) > 0
    return df_amazon_product_info.drop(columns=['rating'])


def load_json(filename):
    product_id = filename.split("/")[-1].split(".")[0]
    with open(filename, 'r') as f:
        item = json.load(f)
        item["product_id"] = product_id
    return item


def load_all_data_as_dataframe():
    df = pd.DataFrame(filter(
        lambda x: len(x) > 1,
        [load_json(each) for each in tqdm(glob(f"{new_data_path}/*.json"), desc="Loading product info")]))[[
        'product_id', 'product_title', 'byline_info', 'product_description', 'category',
        'alt_images', 'product_detail', 'important_information', 'rating', 'top_comments']]
    return add_custom_columns(df)


def extract_comments_from_product_info(df_amazon_product_info: pd.DataFrame):
    df = pd.DataFrame([dict(
        ProductId=row.product_id,
        HelpfulnessNumerator=int(cmt['helpfulness'].replace(',', '')) if cmt['helpfulness'] != 'One' else 1,
        Score=int(float(cmt['score'].split(' ')[0])),
        Time=cmt['date'],
        Summary=cmt['title'],
        Text=cmt['text']
    ) for _, row in df_amazon_product_info.iterrows() for cmt in row.top_comments])
    df['Time'] = pd.to_datetime(df.Time)
    return df


if __name__ == "__main__":
    scrape_main()
