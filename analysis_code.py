
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

df = pd.read_csv("/mnt/data/Customer_Review (1).csv")
df.columns = [c.strip() for c in df.columns]

def map_purchased(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in ['yes','y','true','1','t']: return 1
    if s in ['no','n','false','0','f']: return 0
    try: return int(float(s))
    except: return np.nan

df['Purchased_bin'] = df['Purchased'].apply(map_purchased)
df = df.dropna(subset=['Purchased_bin']).reset_index(drop=True)

df['Age_num'] = pd.to_numeric(df.get('Age', pd.Series()), errors='coerce')
df.loc[(df['Age_num'] < 10) | (df['Age_num'] > 100), 'Age_num'] = np.nan

df['Review_text'] = df['Review'].fillna('').astype(str)
df['review_len_chars'] = df['Review_text'].apply(len)
df['review_word_count'] = df['Review_text'].apply(lambda s: len(s.split()))
df['review_exclaim_count'] = df['Review_text'].apply(lambda s: s.count('!'))
df['review_allcaps_ratio'] = df['Review_text'].apply(lambda s: sum(1 for ch in s if ch.isupper()) / (len(s)+1e-6))

numeric_features = ['Age_num','review_len_chars','review_word_count','review_exclaim_count','review_allcaps_ratio']
categorical_features = [c for c in ['Gender','Education'] if c in df.columns]
text_feature = 'Review_text'
target = 'Purchased_bin'

X = df[numeric_features + categorical_features + [text_feature]].copy()
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), numeric_features),

        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))]), categorical_features),

        ('txt', Pipeline([('tfidf', TfidfVectorizer(max_features=150,
                                                   ngram_range=(1,2),
                                                   stop_words='english'))]), text_feature)
    ],
    remainder='drop',
    sparse_threshold=0
)

rf_clf = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_clf.fit(X_train, y_train)

perm = permutation_importance(rf_clf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1)

print("Permutation importance computed successfully.")
