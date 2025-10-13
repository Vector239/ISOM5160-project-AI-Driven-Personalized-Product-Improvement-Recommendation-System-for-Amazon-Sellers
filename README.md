# ISOM5160-Group7-Project



## Introduction


This is a course project for HKUST ISOM5160. It leverages Python to analyze over 20,000 real-world entries from an Amazon review dataset, with a built-in UI to facilitate intuitive queries. The project also delivers data-driven recommendations for product enhancements, bridging analytical insights with practical business improvements.


## Individual Contribution

| Name          | SID      | Contributions                                                                                                          |
|---------------|----------|------------------------------------------------------------------------------------------------------------------------|
| CAO, Xi       | 21271664 | Review Text Preprocessing; Sentiment Analysis; Review Keyword Extraction; Data Cleaning                                                                          |
| LI, Heyi      | 21265689 | Anomaly Analysis of Comments and Ratings (Based on Sentiment Analysis of Comments), PPT Coordination                                                                          |
| LIAO, Jingyu  | 21262106 | Analysis of Negative Review Reasons (Based on Review Keywords)                                                                                          |
| LIN, Chuwei   | 21237955 | Time Trend Analysis of Rating                                                                                                          |
| YE, Chenwei   | 21199517 | User Comment Weight Analysis (simple weighted & Bayesian smooth)                                                                                                        |
| ZHANG, Ziyang | 21266920 | 1. Data scraping: additional amazon product info <br/>2. Analyse: Correlation Between Ratings and Product Descriptions |

## Timeline

1. [9.21] Preprocessed data and supplementary data:
   - [x] Additional amazon product info
   - [x] Comment Text Processing; Sentiment Analysis
   - [x] Introduction to Text Processing

2. [10.3] Data Analysis and Conclusion Output: 
   1. Analysis of Negative Review Causes
   2. Analysis of Outlier Reviews and Ratings
   3. Analysis of Rating Time Trends
   4. Correlation Analysis Between Product Descriptions and Ratings
   5. User Review Weight Analysis

3. [10.10] PPT Consolidation and Output

## Data Source Introduction

We use ***BOTH*** datasets provided in course requirements and supplementary data w scraped from amazon.

1. `amazon_food_reviews.csv`: Original dataset downloaded from `Canvas`

| File Name               | No. of Rows | No. of Columns  |
|-------------------------|------------|-----------------|
| amazon_food_reviews.csv | 10,828     | 10              |


2. `new_data.zip`: Additional data scrapped from `amazon.com` 

| File Name               | No. of Rows | No. of Columns    |
|-------------------------|-------------|-------------------|
| new_data.zip | 3380  (After loaded)      | 29 (After loaded) |

- Why 3380 rows? We only scrapped info of products in `amazon_food_reviews.csv`

```
RangeIndex: 3381 entries, 0 to 3380
Data columns (total 29 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   product_id              3381 non-null   object 
 1   product_title           3381 non-null   object 
 2   byline_info             3381 non-null   object 
 3   product_description     3381 non-null   object 
 4   category                3381 non-null   object 
 5   alt_images              3381 non-null   object 
 6   product_detail          3381 non-null   object 
 7   important_information   3381 non-null   object 
 8   top_comments            3381 non-null   object 
 9   CountAltImages          3381 non-null   int64  
 10  Score                   3354 non-null   float64
 11  ScoreDistribution       3354 non-null   object 
 12  ScorePolarizationIndex  3354 non-null   float64
 13  NumRatings              3381 non-null   int64  
 14  IsFood                  3381 non-null   bool   
 15  sentiment_score         3381 non-null   float64
 16  mean_score              3377 non-null   float64
 17  num_reviews             3377 non-null   float64
 18  weighted_mean_simple    3377 non-null   float64
 19  weighted_mean_smooth    3358 non-null   float64
 20  bayesian_mean_smooth    3358 non-null   float64
 21  bayesian_mean_simple    3377 non-null   float64
 22  count_1star             3381 non-null   int64  
 23  count_2star             3381 non-null   int64  
 24  count_3star             3381 non-null   int64  
 25  count_4star             3381 non-null   int64  
 26  count_5star             3381 non-null   int64  
 27  diff_smooth_vs_simple   3358 non-null   float64
 28  diff_bayes_vs_mean      3358 non-null   float64
dtypes: bool(1), float64(11), int64(7), object(10)
memory usage: 743.0+ KB
```


## Documentation

### File Structure
- `datasets`: Raw data and processed data
  - `amazon_food_reviews.csv`: Original dataset downloaded from `Canvas`
  - `amazon_food_reviews_cleaned.csv`: Processed dataset
  - `new_data.zip`: Additional data scrapped from `amazon.com` (Folder with json files)
- `doc`: Related documents
- `slides`: LaTex project & output for keynote slides
- `*.py`: Source code files of the functional module
- `ISOM5160-GROUP7.ipynb`: Main code

### Code Running
- Main code: Execute `ISOM5160-GROUP7.ipynb` in sequence.
- Some code can execute independently.
  - `amazon_review_scraper.py`
  - `negative_review_analysis.py`
  - `data_cleaning.py`
