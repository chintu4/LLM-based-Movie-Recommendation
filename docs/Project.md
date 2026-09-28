* It can recognize both user intent and user sentiment.
* 
## In Notebook
https://github.com/chintu4/LLM-based-Movie-Recommendation/blob/main/data_exploration.ipynb
* Exploring the data check columns of `genre`.
* cleared null data
* `movie_missing` is columns where any one row has null value is called movie missing in desired feature columns `columns_of_interest`.
* 

https://github.com/chintu4/LLM-based-Movie-Recommendation/blob/main/sentiment_analysis.ipynb
* `movies_with_Genre.csv` is imported.
* emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
* Are emotions where humans have their emotions.
* `j-hartmann/emotion-english-distilroberta-base` model is used that predict all.
* suppose it was give sentence "I love you " it predicted the following.
  ```
  [[{'label': 'joy', 'score': 0.9771687984466553},
  {'label': 'surprise', 'score': 0.008528684265911579},
  {'label': 'neutral', 'score': 0.005764597095549107},
  {'label': 'anger', 'score': 0.004419783595949411},
  {'label': 'sadness', 'score': 0.002092392183840275},
  {'label': 'disgust', 'score': 0.001611993182450533},
  {'label': 'fear', 'score': 0.0004138521908316761}]]
  ```
* `Title` columns is added previous imported dataset and finally added .
* It finally saved data to `movies_with_emotions.csv`.

https://github.com/chintu4/LLM-based-Movie-Recommendation/blob/main/text_classification.ipynb
* we check what value_count we have of each Genre movies like
```
Drama	374
1	Comedy	328
2	Horror	203
3	Drama, Romance	194
4	Horror, Thriller	161
...	...	...
2085	Adventure, Fantasy, Drama, Mystery	1
2086	Science Fiction, Comedy, Fantasy	1
2087	Thriller, Science Fiction, Mystery, Horror	1
2088	Family, Animation, Adventure, Fantasy, Science...	1
2089	Family, TV Movie, Comedy, Science Fiction, Adv...
```

This notebook is **not training a new machine-learning model**. It is mainly using a pre-trained Hugging Face model to classify movies as either **“Fiction”** or **“Nonfiction”**, then saving the result.

## Overall workflow

```text
Load movie CSV
      ↓
Inspect movie genres
      ↓
Create a simplified Fiction/Nonfiction label
      ↓
Use BART zero-shot classification on movie overviews
      ↓
Compare model predictions with the manually created labels
      ↓
Fill missing labels
      ↓
Save movies_with_Genre.csv
```

## 1. Download and load the movie dataset

The notebook first tries to download `movies_cleaned.csv`:

```python
!curl https://raw.githubusercontent.com/chintu4/LLM-based-Movie-Recommendation/blob/main/movies_cleaned.csv -o movies_cleaned.csv
```

Then it loads the file using pandas:

```python
import pandas as pd

movies = pd.read_csv("movies_cleaned.csv")
```

The dataset contains columns such as:

- `Title`
- `Overview`
- `Genre`
- `Popularity`
- `Vote_Average`
- `Poster_Url`
- `tagged_description`

The `Overview` column contains the movie description, which is later given to the language model.

### Important issue

The download URL appears incorrect. A raw GitHub URL should normally be:

```python
!curl https://raw.githubusercontent.com/chintu4/LLM-based-Movie-Recommendation/main/movies_cleaned.csv -o movies_cleaned.csv
```

The `/blob/main/` part generally belongs to a normal GitHub webpage URL, not a raw file URL.

## 2. Analyze the existing genres

This code counts how many movies have each exact genre string:

```python
movies["Genre"].value_counts().reset_index()
```

For example, the output contains categories such as:

```text
Drama                  374
Comedy                 328
Horror                 203
Drama, Romance         194
Horror, Thriller       161
```

Because genres can contain combinations like `"Drama, Romance"`, the dataset has many unique genre values—around 2,090 combinations.

The next code shows only genre combinations appearing more than 50 times:

```python
movies["Genre"].value_counts().reset_index().query("count > 50")
```

This is exploratory data analysis. It helps the author understand the distribution of genres.

## 3. Create a simplified genre label

The notebook creates a new column called `simple_Genre`:

```python
movies["simple_Genre"] = "Fiction"
```

This means every movie is initially classified as `"Fiction"`.

Then it defines keywords associated with nonfiction:

```python
nonfiction_keywords = [
    "documentary",
    "biography",
    "history",
    "science",
    "literary criticism",
    "philosophy",
    "religion",
    "juvenile nonfiction"
]
```

For each keyword, it checks whether that keyword appears inside the original `Genre` column:

```python
for keyword in nonfiction_keywords:
    movies.loc[
        movies["Genre"].str.contains(keyword, case=False, na=False),
        "simple_Genre"
    ] = "Nonfiction"
```

Therefore:

- A movie containing `"Documentary"` in its genre becomes `"Nonfiction"`.
- A movie containing `"History"` becomes `"Nonfiction"`.
- Everything else remains `"Fiction"`.

For example:

```text
Genre                         simple_Genre
Documentary                   Nonfiction
Drama                         Fiction
Comedy, Romance               Fiction
Drama, History                Nonfiction
```

This is a **rule-based classification**, not a trained classifier.

## 4. Load the zero-shot classification model

The notebook loads this Hugging Face pipeline:

```python
from transformers import pipeline

fiction_Genre = ["Fiction", "Nonfiction"]

pipe = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0
)
```

The model is `facebook/bart-large-mnli`.

Zero-shot classification means the model was not specifically trained in this notebook. Instead, it receives:

1. A text sequence, such as a movie overview.
2. Candidate labels, such as `["Fiction", "Nonfiction"]`.

Example:

```python
pipe(
    "A documentary about the history of space exploration.",
    ["Fiction", "Nonfiction"]
)
```

The model returns scores for both labels and selects the label with the highest score.

## 5. Test the classifier on one movie

The notebook selects the overview of the first movie marked as Fiction:

```python
sequence = movies.loc[
    movies["simple_Genre"] == "Fiction",
    "Overview"
].reset_index(drop=True)[0]
```

Then it classifies that overview:

```python
pipe(sequence, fiction_Genre)
```

The model returns something like:

```python
{
    "labels": ["Nonfiction", "Fiction"],
    "scores": [0.72, 0.28],
    "sequence": "..."
}
```

The notebook gets the label with the highest score:

```python
max_index = np.argmax(pipe(sequence, fiction_Genre)["scores"])
max_label = pipe(sequence, fiction_Genre)["labels"][max_index]
```

## 6. Define a reusable prediction function

Instead of repeating the classification logic, it defines:

```python
def generate_predictions(sequence, Genre):
    predictions = pipe(sequence, Genre)
    max_index = np.argmax(predictions["scores"])
    max_label = predictions["labels"][max_index]
    return max_label
```

This function:

1. Sends the movie overview to BART.
2. Gets the scores.
3. Finds the highest score.
4. Returns either `"Fiction"` or `"Nonfiction"`.

## 7. Classify all movies

The notebook separates the movies into two groups:

```python
fiction_sequences = movies.loc[
    movies["simple_Genre"] == "Fiction",
    "Overview"
].reset_index(drop=True)
```

It then predicts every Fiction movie:

```python
for i in tqdm(range(0, len(fiction_sequences))):
    sequence = fiction_sequences[i]
    predicted_cats += [
        generate_predictions(sequence, fiction_Genre)
    ]
    actual_cats += ["Fiction"]
```

It repeats the same process for Nonfiction movies:

```python
nonfiction_sequences = movies.loc[
    movies["simple_Genre"] == "Nonfiction",
    "Overview"
].reset_index(drop=True)
```

The `tqdm` library displays progress while the predictions run. The recorded output shows that thousands of movie descriptions were processed, which took several minutes.

## 8. Compare predictions with the rule-based labels

The predictions are stored in a DataFrame:

```python
predictions_df = pd.DataFrame({
    "actual_Genre": actual_cats,
    "predicted_Genre": predicted_cats
})
```

Then the notebook marks each prediction as correct or incorrect:

```python
predictions_df["correct_prediction"] = np.where(
    predictions_df["actual_Genre"] ==
    predictions_df["predicted_Genre"],
    1,
    0
)
```

Finally, it computes:

```python
predictions_df["correct_prediction"].sum() / len(predictions_df)
```

This calculates the prediction accuracy.

However, this is not a true ground-truth accuracy measurement. The `"actual_Genre"` values were themselves created using simple keyword rules, so the model is being evaluated against those rules—not against verified movie labels.

## 9. Fill missing categories

The notebook later searches for rows where `simple_Genre` is missing:

```python
missing_cats = movies.loc[
    movies["simple_Genre"].isna(),
    ["Title", "Overview"]
].reset_index(drop=True)
```

It sends those missing movie overviews to the classifier and creates:

```python
missing_predicted_df = pd.DataFrame({
    "Title": isbns,
    "predicted_Genre": predicted_cats
})
```

Then it merges the predictions back into `movies`:

```python
movies = pd.merge(
    movies,
    missing_predicted_df,
    on="Title",
    how="left"
)

movies["simple_Genre"] = np.where(
    movies["simple_Genre"].isna(),
    movies["predicted_Genre"],
    movies["simple_Genre"]
)

movies = movies.drop(columns=["predicted_Genre"])
```

### But there is a logic problem

Because this line assigns `"Fiction"` to every row:

```python
movies["simple_Genre"] = "Fiction"
```

there should not be any missing values in `simple_Genre`.

Therefore:

```python
movies["simple_Genre"].isna()
```

will normally be false for every row, making `missing_cats` empty. The “fill missing categories” section is effectively unnecessary unless the notebook originally intended to initialize the column with `NaN`.

A better approach would be to initialize it like this:

```python
movies["simple_Genre"] = np.nan
```

Then apply the keyword rules, and classify only the rows that remain missing.

## 10. Save the result

At the end, the processed dataset is saved:

```python
movies.to_csv("movies_with_Genre.csv", index=False)
```

The final file contains the original movie information plus the new:

```text
simple_Genre
```

column.

## Important observations

### It is not really “text classification” training

There is no:

- train/test split
- loss function
- optimizer
- model fine-tuning
- neural network training
- learned movie-specific classifier

Instead, it uses a pre-trained BART model through **zero-shot classification**.

### It reduces many genres to only two categories

The original dataset has genres such as:

```text
Drama
Comedy
Horror
Action, Thriller
Drama, Romance
Animation, Family
```

But the notebook reduces all of these to only:

```text
Fiction
Nonfiction
```

This is a very broad and somewhat artificial distinction.

### The rule-based labels may be inaccurate

The notebook assumes that movies with genres such as `"History"` or `"Science"` are nonfiction. But a historical or science-fiction movie can still be fictional.

For example:

- `"Science Fiction"` is fiction, despite containing the word `"science"`.
- A historical drama may be fictional.
- A documentary may include `"Drama"` or other genres.

### `device=0` requires a GPU

This line:

```python
device=0
```

tells Hugging Face to use the first CUDA GPU. It will fail if no GPU is available.

The notebook metadata mentions a TPU, but the pipeline configuration is written for a GPU. In Google Colab, the runtime should be set to GPU, or the code should use:

```python
device=-1
```

for CPU execution.

## In simple terms

The notebook does this:

> “Look at each movie’s description and decide whether it sounds more like Fiction or Nonfiction using the pre-trained BART language model. Then compare those predictions with labels created using simple genre keywords and save the results.”
