
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# تنسيق CSS مخصص مع ألوان مختلفة لكل عنصر
st.markdown("""
<style>
 
    /* شعار الجامعة وتفاصيل الفريق */
    .university-header {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
   
    .university-logo {
        text-align: center;
        margin-bottom: 15px;
    }
   
    .team-section {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
        flex-wrap: wrap;
    }
   
    .team-right, .supervisor-left {
        background: rgba(255, 215, 0, 0.2);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
   
    .team-right {
        text-align: right;
        width: 48%;
    }
   
    .supervisor-left {
        text-align: left;
        width: 48%;
    }
   
    .main-header {
        font-size: 3rem;
        color: #E50914;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    /* أزرار التصنيفات - ذهبية */
    .genre-button {
        background-color: #FFD700;
        color: #000;
        border: none;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem;
        border-radius: 0.8rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .genre-button:hover {
        background-color: #FFC400;
        transform: scale(1.05);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    /* بطاقات الأفلام - أزرق */
    .movie-card {
        background: linear-gradient(135deg, #1E90FF 0%, #0077CC 100%);
        color: white;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border-left: 5px solid #FFD700;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    /* بطاقات الإحصائيات - أحمر */
    .stats-card {
        background: linear-gradient(135deg, #E50914 0%, #B20710 100%);
        color: white;
        border-radius: 1rem;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    /* العناوين الرئيسية - أحمر */
    .section-header {
        font-size: 2rem;
        color: #E50914;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #FFD700;
        padding-bottom: 0.5rem;
    }
    /* أقسام التقارير - أخضر */
    .report-section {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    /* بطاقات التنبؤات - بنفسجي */
    .prediction-card {
        background: linear-gradient(135deg, #9370DB 0%, #8A2BE2 100%);
        color: white;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    /* أزرار التنزيل - برتقالي */
    .download-button {
        background-color: #FF8C00;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        border-radius: 8px;
        margin: 10px 2px;
        cursor: pointer;
        border: none;
        font-weight: bold;
    }
    .download-button:hover {
        background-color: #FF7700;
    }
    /* أزرار المعلومات - أزرق فاتح */
    .info-button {
        background-color: #1E90FF;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        margin: 5px;
    }
    .info-button:hover {
        background-color: #0077CC;
    }
    /* أزرار البحث - أخضر */
    .search-button {
        background-color: #32CD32;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .search-button:hover {
        background-color: #28A428;
    }
    /* أزرار التصفية - بنفسجي فاتح */
    .filter-button {
        background-color: #9370DB;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        margin: 5px;
    }
    .filter-button:hover {
        background-color: #8A2BE2;
    }
    /* أزرار التحليل - وردي */
    .analysis-button {
        background-color: #FF69B4;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        margin: 5px;
    }
    .analysis-button:hover {
        background-color: #FF1493;
    }
    /* أزرار التنبؤ - أزرق داكن */
    .prediction-button {
        background-color: #000080;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        margin: 5px;
    }
    .prediction-button:hover {
        background-color: #0000CD;
    }
    /* بطاقات التوصيات - وردي */
    .recommendation-card {
        background: linear-gradient(135deg, #FF69B4 0%, #FF1493 100%);
        color: white;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    .recommendation-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


</style>
""", unsafe_allow_html=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# قم بإنشاء متغير يحمل الصورة المشفرة
# استبدل 'your_background_image.jpg' باسم ملف صورتك
background_image_path = "ali1.jpg"
background_image_base64 = get_base64_of_bin_file(background_image_path)

st.markdown(f"""
<style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{background_image_base64}");
        background-size: cover;
        background-position: center;
    }}
</style>
""", unsafe_allow_html=True)


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
# قسم شعار الجامعة والفريق
# استبدل 'SULTAN.PNG' باسم ملف الصورة الخاص بك
logo_path = "SULTAN.PNG"
logo_base64 = get_base64_of_bin_file(logo_path)
logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="شعار جامعة أزال" style="width:100px;height:auto; display: block; margin-left: auto; margin-right: auto;">'

st.markdown(f"""
<div class="university-header">
    <div class="university-logo">
        {logo_html}
        <h2 style="text-align: center;">جامعة آزال للعلوم والتكنولوجيا</h2>
        <p style="text-align: center;">Azal University for Science and Technology</p>
    </div>
</div>
   
    <div class="team-section">
        <div class="supervisor-left">
            <h1>تحت إشراف</h1>
            <p>الدكتور صفوان</p>
        </div>
       
        <div class="team-right">
            <h1>فريق العمل</h1>
            <p>علي الكوكباني</p>
            <p>حسين الزبيري</p>
            <p>نجم الدين الوريث</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.markdown('<h1 class="main-header">🎬 Netflix & IMDb Explorer</h1>', unsafe_allow_html=True)

# تحميل البيانات من ملف محلي
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('imdb.csv')
        st.sidebar.success("تم تحميل imdb.csv بنجاح!")
    except:
        try:
            df = pd.read_csv('imdb-processed.csv')
            st.sidebar.success("تم تحميل imdb-processed.csv بنجاح!")
        except:
            import os
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if csv_files:
                df = pd.read_csv(csv_files[0])
                st.sidebar.success(f"تم تحميل {csv_files[0]} بنجاح!")
            else:
                # إنشاء بيانات نموذجية لأغراض العرض
                st.warning("⚠️ لم يتم العثور على أي ملف CSV في المجلد، سيتم استخدام بيانات نموذجية.")
                data = {
                    'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
                             'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
                             'The Lord of the Rings: The Fellowship of the Ring',
                             'The Lord of the Rings: The Two Towers',
                             'The Lord of the Rings: The Return of the King',
                             'Fight Club', 'Goodfellas', 'The Silence of the Lambs',
                             'Interstellar', 'The Departed', 'Whiplash', 'Gladiator'],
                    'year': [1994, 1972, 2008, 1994, 1994, 2010, 1999, 2001, 2002, 2003,
                            1999, 1990, 1991, 2014, 2006, 2014, 2000],
                    'rating': [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.8, 8.7, 8.9,
                              8.8, 8.7, 8.6, 8.6, 8.5, 8.5, 8.5],
                    'genre': ['Drama', 'Crime,Drama', 'Action,Crime,Drama',
                             'Crime,Drama', 'Drama,Romance', 'Action,Adventure,Sci-Fi',
                             'Action,Sci-Fi', 'Adventure,Drama,Fantasy',
                             'Adventure,Drama,Fantasy', 'Adventure,Drama,Fantasy',
                             'Drama', 'Biography,Crime,Drama', 'Crime,Drama,Thriller',
                             'Adventure,Drama,Sci-Fi', 'Crime,Drama,Thriller',
                             'Drama,Music', 'Action,Adventure,Drama'],
                    'director': ['Frank Darabont', 'Francis Ford Coppola', 'Christopher Nolan',
                                'Quentin Tarantino', 'Robert Zemeckis', 'Christopher Nolan',
                                'Lana Wachowski, Lilly Wachowski', 'Peter Jackson',
                                'Peter Jackson', 'Peter Jackson', 'David Fincher',
                                'Martin Scorsese', 'Jonathan Demme', 'Christopher Nolan',
                                'Martin Scorsese', 'Damien Chazelle', 'Ridley Scott'],
                    'cast': ['Tim Robbins, Morgan Freeman', 'Marlon Brando, Al Pacino',
                            'Christian Bale, Heath Ledger', 'John Travolta, Uma Thurman',
                            'Tom Hanks, Robin Wright', 'Leonardo DiCaprio, Joseph Gordon-Levitt',
                            'Keanu Reeves, Laurence Fishburne', 'Elijah Wood, Ian McKellen',
                            'Elijah Wood, Ian McKellen', 'Elijah Wood, Ian McKellen',
                            'Brad Pitt, Edward Norton', 'Robert De Niro, Ray Liotta',
                            'Jodie Foster, Anthony Hopkins', 'Matthew McConaughey, Anne Hathaway',
                            'Leonardo DiCaprio, Matt Damon', 'Miles Teller, J.K. Simmons',
                            'Russell Crowe, Joaquin Phoenix'],
                    'description': ['Two imprisoned men bond over a number of years...',
                                   'The aging patriarch of an organized crime dynasty...',
                                   'When the menace known as the Joker wreaks havoc...',
                                   'The lives of two mob hitmen, a boxer, a gangster...',
                                   'The presidencies of Kennedy and Johnson, the events...',
                                   'A thief who steals corporate secrets through...',
                                   'A computer hacker learns from mysterious rebels...',
                                   'A meek Hobbit from the Shire and eight companions...',
                                   'While Frodo and Sam edge closer to Mordor...',
                                   'Gandalf and Aragorn lead the World of Men against Sauron...',
                                   'An insomniac office worker and a devil-may-care soapmaker...',
                                   'The story of Henry Hill and his life in the mob...',
                                   'A young F.B.I. cadet must receive the help of an incarcerated...',
                                   'A team of explorers travel through a wormhole in space...',
                                   'An undercover cop and a mole in the police attempt...',
                                   'A promising young drummer enrolls at a cut-throat music...',
                                   'A former Roman General sets out to exact vengeance...']
                }
                df = pd.DataFrame(data)
    return df

df = load_data()

if df.empty:
    st.stop()

# تنظيف البيانات الأساسية
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    df = df.dropna(subset=numeric_cols[:2], how='all')
# إعداد نظام التوصيات
@st.cache_data
def prepare_recommendation_system(_df):
    # إنشاء نسخة من البيانات للعمل عليها
    rec_df = _df.copy()
   
    # تنظيف النوع ليكون أكثر اتساقًا
    if 'genre' in rec_df.columns:
        rec_df['genre'] = rec_df['genre'].fillna('').apply(lambda x: re.sub(r'[^a-zA-Z,]', '', x))
   
    # إنشاء ميزات للتصفية بناءً على المحتوى
    rec_df['content_features'] = ''
   
    if 'genre' in rec_df.columns:
        rec_df['content_features'] += rec_df['genre'].fillna('') + ' '
   
    if 'director' in rec_df.columns:
        rec_df['content_features'] += rec_df['director'].fillna('').str.replace(',', ' ') + ' '
   
    if 'cast' in rec_df.columns:
        rec_df['content_features'] += rec_df['cast'].fillna('').str.replace(',', ' ') + ' '
   
    if 'description' in rec_df.columns:
        rec_df['content_features'] += rec_df['description'].fillna('') + ' '
   
    # الحل: استبدال أي قيم NaN في عمود content_features بسلسلة نصية فارغة
    rec_df['content_features'] = rec_df['content_features'].fillna('')
   
    # استخدام TF-IDF لتحويل النص إلى متجهات
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(rec_df['content_features'])
   
    # حساب مصفوفة التشابه
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
   
    return cosine_sim, rec_df

   
    # استخدام TF-IDF لتحويل النص إلى متجهات
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(rec_df['content_features'])
   
    # حساب مصفوفة التشابه
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
   
    return cosine_sim, rec_df

# إنشاء نظام التوصيات
cosine_sim, rec_df = prepare_recommendation_system(df)

# وظيفة للحصول على التوصيات
def get_recommendations(title, cosine_sim=cosine_sim, _df=rec_df, num_recommendations=5):
    if title not in _df['title'].values:
        return pd.DataFrame()
   
    # الحصول على индекс الفيلم
    idx = _df[_df['title'] == title].index[0]
   
    # الحصول على درجات التشابه للفيلم
    sim_scores = list(enumerate(cosine_sim[idx]))
   
    # ترتيب الأفلام بناءً على درجات التشابه
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   
    # الحصول على نتائج أكثر الأفلام تشابهًا (تخطي الأول لأنه الفيلم نفسه)
    sim_scores = sim_scores[1:num_recommendations+1]
   
    # الحصول على مؤشرات الأفلام
    movie_indices = [i[0] for i in sim_scores]
   
    # الحصول على درجات التشابه
    similarity_scores = [i[1] for i in sim_scores]
   
    # إرجاع أفضل التوصيات
    recommendations = _df[['title', 'year', 'rating', 'genre']].iloc[movie_indices].copy()
    recommendations['similarity_score'] = similarity_scores
    recommendations['similarity_percentage'] = recommendations['similarity_score'].apply(lambda x: f"{x*100:.1f}%")
   
    return recommendations

# وظيفة للحصول على توصيات بناءً على التفضيلات
def get_preference_based_recommendations(preferred_genres, min_rating=7.0, _df=df, num_recommendations=10):
    # تصفية الأفلام بناءً على التفضيلات
    filtered_df = _df.copy()
   
    if 'rating' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
   
    if preferred_genres and 'genre' in filtered_df.columns:
        genre_filter = filtered_df['genre'].str.contains('|'.join(preferred_genres), case=False, na=False)
        filtered_df = filtered_df[genre_filter]
   
    # ترتيب الأفلام حسب التقييم
    if 'rating' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('rating', ascending=False)
   
    return filtered_df.head(num_recommendations)

# وظيفة للحصول على صورة افتراضية بناءً على نوع الفيلم
def get_genre_emoji(genre_name):
    genre_emojis = {
        "Action": "🔫", "Comedy": "😂", "Drama": "🎭",
        "Thriller": "🔪", "Romance": "❤️", "Horror": "👻",
        "Adventure": "🗺️", "Sci-Fi": "🚀", "Documentary": "📽️",
        "Animation": "🐰", "Crime": "👮", "Fantasy": "🦄",
        "Biography": "📖", "Music": "🎵", "Mystery": "🕵️"
    }
   
    if pd.isna(genre_name):
        return "🎬"
   
    for genre, emoji in genre_emojis.items():
        if genre.lower() in str(genre_name).lower():
            return emoji
    return "🎬"  # رمز افتراضي

# الشريط الجانبي
st.sidebar.header("🎛️ خيارات التصفية")

# أزرار التصنيفات
st.sidebar.markdown("### 🎭 اختر أنواع الأفلام المفضلة")

# قائمة بالأنواع الشائعة (يمكن تعديلها حسب بياناتك)
popular_genres = [
    "Action", "Comedy", "Drama", "Thriller", "Romance",
    "Horror", "Adventure", "Sci-Fi", "Documentary", "Animation",
    "Crime", "Fantasy", "Biography", "Music", "Mystery"
]

# إنشاء أزرار للأنواع
selected_genres = st.sidebar.multiselect(
    "اختر أنواع الأفلام:",
    options=popular_genres,
    default=[]
)

# تصفية حسب السنة
year_col = None
if 'year' in df.columns or 'release_year' in df.columns:
    year_col = 'release_year' if 'release_year' in df.columns else 'year'
    min_year = int(df[year_col].min())
    max_year = int(df[year_col].max())
    year_range = st.sidebar.slider(
        '📅 اختر نطاق السنوات',
        min_year, max_year, (max_year-10, max_year)
    )

# تصفية حسب التقييم
rating_col = None
rating_cols = [col for col in df.columns if 'rating' in col.lower() or 'score' in col.lower()]
if rating_cols:
    rating_col = rating_cols[0]
    min_rating = float(df[rating_col].min())
    max_rating = float(df[rating_col].max())
    rating_range = st.sidebar.slider(
        '⭐ اختر نطاق التقييم',
        min_rating, max_rating, (7.0, max_rating),
        step=0.1
    )

# تطبيق التصفية
def apply_filters(df):
    filtered_df = df.copy()
   
    if year_col and 'year_range' in locals():
        filtered_df = filtered_df[(filtered_df[year_col] >= year_range[0]) &
                                (filtered_df[year_col] <= year_range[1])]
   
    if rating_col and 'rating_range' in locals():
        filtered_df = filtered_df[(filtered_df[rating_col] >= rating_range[0]) &
                                (filtered_df[rating_col] <= rating_range[1])]
   
    if selected_genres:
        genre_cols = [col for col in df.columns if 'genre' in col.lower()]
        if genre_cols:
            genre_col = genre_cols[0]
            genre_filter = filtered_df[genre_col].str.contains('|'.join(selected_genres), case=False, na=False)
            filtered_df = filtered_df[genre_filter]
   
    return filtered_df

filtered_df = apply_filters(df)
st.sidebar.markdown(f"### 🎯 عدد الأفلام بعد التصفية: **{len(filtered_df)}**")

# قسم التوصيات في الشريط الجانبي
st.sidebar.header("💡 نظام التوصيات")

# خيارات نظام التوصيات
recommendation_type = st.sidebar.radio(
    "نوع التوصيات:",
    ["بناءً على التفضيلات", "بناءً على فيلم معين"]
)

if recommendation_type == "بناءً على فيلم معين":
    # اختيار فيلم للحصول على توصيات
    movie_titles = df['title'].tolist()
    selected_movie = st.sidebar.selectbox("اختر فيلمًا:", movie_titles)
   
    if st.sidebar.button("الحصول على توصيات"):
        with st.spinner('جاري البحث عن توصيات...'):
            recommendations = get_recommendations(selected_movie)
           
            if not recommendations.empty:
                st.sidebar.success("تم العثور على توصيات!")
               
                # عرض التوصيات في الشريط الجانبي
                st.sidebar.markdown("### 🎬 أفلام مشابهة")
                for _, movie in recommendations.iterrows():
                    genre_emoji = get_genre_emoji(movie['genre'])
                    st.sidebar.markdown(f"""
                    <div style="background: rgba(255,105,180,0.2); padding: 10px; border-radius: 10px; margin: 5px 0;">
                        <strong>{genre_emoji} {movie['title']}</strong> ({movie['year']})<br>
                        ⭐ {movie['rating']} | التشابه: {movie['similarity_percentage']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.sidebar.warning("لم يتم العثور على توصيات لهذا الفيلم.")

# قسم التوصيات في الواجهة الرئيسية
st.markdown('<h2 class="section-header">💡 توصيات مخصصة</h2>', unsafe_allow_html=True)

# الحصول على التوصيات بناءً على التفضيلات
if selected_genres or (rating_col and rating_range[0] > 0):
    preference_recommendations = get_preference_based_recommendations(
        selected_genres,
        rating_range[0] if rating_col else 7.0,
      _df=filtered_df
    )
   
    if not preference_recommendations.empty:
        st.markdown(f"### 🎯 أفلام قد تعجبك بناءً على تفضيلاتك")
       
        # عدد الأعمدة للعرض
        cols_per_row = 3
        rows = [preference_recommendations[i:i+cols_per_row] for i in range(0, min(9, len(preference_recommendations)), cols_per_row)]
       
        for row in rows:
            cols = st.columns(cols_per_row)
            for idx, (_, movie) in enumerate(row.iterrows()):
                with cols[idx]:
                    # الحصول على بيانات الفيلم
                    title = movie.get('title', 'No title')
                    year = movie.get(year_col, 'N/A') if year_col else 'N/A'
                    rating = movie.get(rating_col, 'N/A') if rating_col else 'N/A'
                   
                    # الحصول على رمز النوع
                    genre_emoji = "🎬"
                    genre_cols = [col for col in movie.index if 'genre' in col.lower()]
                    if genre_cols and genre_cols[0] in movie:
                        genre_emoji = get_genre_emoji(movie[genre_cols[0]])
                   
                    # إنشاء بطاقة الفيلم الموصى به
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{genre_emoji} {title}</h4>
                        <p><strong>📅 السنة:</strong> {year}</p>
                        <p><strong>⭐ التقييم:</strong> {rating}</p>
                        <p><strong>🎭 النوع:</strong> {movie.get(genre_cols[0] if genre_cols and genre_cols[0] in movie else 'N/A', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("⚠️ لم نتمكن من العثور على توصيات بناءً على تفضيلاتك الحالية. حاول توسيع نطاق التصفية.")

# قسم "اكتشف أفلامًا مشابهة"
st.markdown("### 🔍 اكتشف أفلامًا مشابهة")
movie_to_compare = st.selectbox("اختر فيلمًا لاكتشاف أفلام مشابهة:", df['title'].tolist())

if st.button("اكتشف أفلامًا مشابهة"):
    with st.spinner('جاري البحث عن أفلام مشابهة...'):
        similar_movies = get_recommendations(movie_to_compare, num_recommendations=6)
       
        if not similar_movies.empty:
            st.success(f"أفلام مشابهة لـ **{movie_to_compare}**")
           
            # عرض الأفلام المشابهة في أعمدة
            cols = st.columns(3)
            for idx, (_, movie) in enumerate(similar_movies.iterrows()):
                with cols[idx % 3]:
                    genre_emoji = get_genre_emoji(movie['genre'])
                    st.markdown(f"""
                    <div class="movie-card">
                        <h4>{genre_emoji} {movie['title']}</h4>
                        <p><strong>📅 السنة:</strong> {movie['year']}</p>
                        <p><strong>⭐ التقييم:</strong> {movie['rating']}</p>
                        <p><strong>🎭 النوع:</strong> {movie['genre']}</p>
                        <p><strong>📊 التشابه:</strong> {movie['similarity_percentage']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("لم يتم العثور على أفلام مشابهة.")

# قسم تحليل الأنماط والتوصيات
st.markdown("### 📈 تحليل الأنماط والتوصيات")

# تحليل الأنواع الشائعة
if 'genre' in df.columns:
    genre_col = 'genre'
    df[genre_col] = df[genre_col].astype(str)
    all_genres = df[genre_col].str.split(',', expand=True).stack().str.strip()
    genre_counts = all_genres.value_counts().head(8)
   
    # عرض الأنواع الشائعة
    st.markdown("#### 🎭 الأنواع الأكثر شيوعًا")
    col1, col2 = st.columns(2)
   
    with col1:
        for genre, count in genre_counts.items():
            emoji = get_genre_emoji(genre)
            st.write(f"{emoji} **{genre}**: {count} فيلم")
   
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        genre_counts.plot(kind='barh', ax=ax, color=['#E50914', '#1E90FF', '#FFD700', '#32CD32', '#9370DB', '#FF69B4', '#FF8C00', '#000080'])
        ax.set_title('الأنواع الأكثر شيوعًا')
        ax.set_xlabel('عدد الأفلام')
        plt.tight_layout()
        st.pyplot(fig)

# توصيات بناءً على الاتجاهات
st.markdown("#### 📊 توصيات بناءً على الاتجاهات")

if year_col and rating_col:
    # تحليل اتجاهات التقييمات عبر السنوات
    yearly_stats = df.groupby(year_col).agg({rating_col: ['mean', 'count']}).reset_index()
    yearly_stats.columns = [year_col, 'avg_rating', 'movie_count']
   
    # العثور على السنوات ذات الأعلى تقييمًا
    top_years = yearly_stats.nlargest(5, 'avg_rating')
   
    col1, col2 = st.columns(2)
   
    with col1:
        st.markdown("**أفضل السنوات من حيث التقييم:**")
        for _, row in top_years.iterrows():
            st.write(f"📅 **{int(row[year_col])}**: ⭐ {row['avg_rating']:.2f} ({int(row['movie_count'])} فيلم)")
   
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(yearly_stats[year_col], yearly_stats['avg_rating'], s=yearly_stats['movie_count']*2,
                  alpha=0.6, c=yearly_stats['avg_rating'], cmap='RdYlGn')
        ax.set_xlabel('السنة')
        ax.set_ylabel('متوسط التقييم')
        ax.set_title('اتجاهات التقييمات عبر السنوات')
        plt.tight_layout()
        st.pyplot(fig)
   
    # تقديم توصيات بناءً على التحليل
    st.markdown("#### 💎 نصائح لاكتشاف الأفلام")
    st.markdown("""
    <div class="report-section">
        <ul>
            <li>🔍 <strong>استخدم خيارات التصفية</strong> للعثور على الأفلام التي تتناسب مع ذوقك</li>
            <li>🎯 <strong>جرب أنواعًا جديدة</strong> بناءً على التوصيات المقدمة</li>
            <li>⭐ <strong>انظر إلى تقييمات الأفلام</strong> ولكن لا تعتمد عليها completamente</li>
            <li>📅 <strong>اكتشف أفلامًا من سنوات مختلفة</strong> لتجربة متنوعة</li>
            <li>🤝 <strong>شارك تجربتك</strong> مع الآخرين واحصل على recommendations منهم</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# باقي الكود الأصلي يبقى كما هو...
# [يتبع باقي الكود الأصلي دون تغيير مع الحفاظ على جميع الوظائف الموجودة]

# ... [يتبع باقي الكود الأصلي] ...

# عرض الأفلام المصفاة
if len(filtered_df) > 0:
    st.markdown('<h2 class="section-header">🎬 الأفلام المختارة</h2>', unsafe_allow_html=True)
   
    # عدد الأعمدة للعرض
    cols_per_row = 3
    rows = [filtered_df[i:i+cols_per_row] for i in range(0, min(12, len(filtered_df)), cols_per_row)]
   
    for row in rows:
        cols = st.columns(cols_per_row)
        for idx, (_, movie) in enumerate(row.iterrows()):
            with cols[idx]:
                # الحصول على بيانات الفيلم
                title = movie.get('title', 'No title')
                year = movie.get(year_col, 'N/A') if year_col else 'N/A'
                rating = movie.get(rating_col, 'N/A') if rating_col else 'N/A'
               
                # الحصول على رمز النوع
                genre_emoji = "🎬"
                genre_cols = [col for col in movie.index if 'genre' in col.lower()]
                if genre_cols and genre_cols[0] in movie:
                    genre_emoji = get_genre_emoji(movie[genre_cols[0]])
               
                # إنشاء بطاقة الفيلم
                st.markdown(f"""
                <div class="movie-card">
                    <h3>{genre_emoji} {title}</h3>
                    <p><strong>📅 السنة:</strong> {year}</p>
                    <p><strong>⭐ التقييم:</strong> {rating}</p>
                    <p><strong>🎭 النوع:</strong> {movie.get(genre_cols[0] if genre_cols and genre_cols[0] in movie else 'N/A', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
               
                # زر لمشاهدة المزيد من المعلومات
                if st.button("المزيد من المعلومات", key=f"btn_more_{idx}_{title}"):
                    with st.expander(f"تفاصيل الفيلم: {title}"):
                        st.write(f"**📅 السنة:** {year}")
                        st.write(f"**⭐ التقييم:** {rating}")
                       
                        if genre_cols and genre_cols[0] in movie:
                            st.write(f"**🎭 النوع:** {movie[genre_cols[0]]}")
                       
                        if 'director' in movie:
                            st.write(f"**🎬 المخرج:** {movie['director']}")
                       
                        if 'cast' in movie:
                            st.write(f"**👥 طاقم التمثيل:** {movie['cast']}")
                       
                        if 'description' in movie:
                            st.write(f"**📝 الوصف:** {movie['description']}")
else:
    st.warning("⚠️ لم يتم العثور على أفلام تطابق معايير التصفية الخاصة بك.")

# إحصائيات وتحليلات
st.markdown('<h2 class="section-header">📊 الإحصائيات والتحليلات</h2>', unsafe_allow_html=True)

if len(filtered_df) > 0:
    col1, col2, col3 = st.columns(3)
   
    with col1:
        avg_rating = filtered_df[rating_col].mean() if rating_col else 0
        st.markdown(f"""
        <div class="stats-card">
            <h3>⭐ متوسط التقييم</h3>
            <h2>{avg_rating:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
   
    with col2:
        latest_year = filtered_df[year_col].max() if year_col else 'N/A'
        st.markdown(f"""
        <div class="stats-card">
            <h3>📅 أحدث سنة</h3>
            <h2>{latest_year}</h2>
        </div>
        """, unsafe_allow_html=True)
   
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <h3>🎬 عدد الأفلام</h3>
            <h2>{len(filtered_df)}</h2>
        </div>
        """, unsafe_allow_html=True)
   
    # رسم بياني لتوزيع التقييمات
    if rating_col:
        fig, ax = plt.subplots(figsize=(10, 6))
        filtered_df[rating_col].hist(bins=20, ax=ax, color='#E50914', edgecolor='black', alpha=0.7)
        ax.set_title('📈 توزيع التقييمات للأفلام المختارة', fontsize=16)
        ax.set_xlabel('التقييم')
        ax.set_ylabel('عدد الأفلام')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# قسم التحليلات المتقدمة
st.markdown('<h2 class="section-header">📈 تحليلات متقدمة</h2>', unsafe_allow_html=True)

if st.checkbox('عرض البيانات الخام', key='raw_data'):
    st.subheader('البيانات الخام')
    st.dataframe(df)

if st.checkbox('الإحصائيات الأساسية', key='basic_stats'):
    st.subheader('📊 الإحصائيات الأساسية')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("عدد الأفلام", df.shape[0])
    with col2:
        st.metric("عدد الأعمدة", df.shape[1])
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        st.metric("الأعمدة الرقمية", numeric_cols)
   
    st.write("الإحصائيات الوصفية:")
    st.dataframe(df.describe())

if st.checkbox('تحليل التقييمات', key='rating_analysis'):
    st.subheader('⭐ تحليل التقييمات')
    if rating_cols:
        rating_col = rating_cols[0]
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
       
        # توزيع التقييمات
        df[rating_col].hist(bins=30, ax=ax[0], color='lightgreen', edgecolor='black')
        ax[0].set_title('توزيع التقييمات')
        ax[0].set_xlabel('التقييم')
        ax[0].set_ylabel('عدد الأفلام')
       
        # boxplot
        df[rating_col].plot(kind='box', ax=ax[1])
        ax[1].set_title('مخطط الصندوق للتقييمات')
       
        st.pyplot(fig)
       
        # إحصائيات
        st.write("📈 إحصائيات التقييمات:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("المتوسط", f"{df[rating_col].mean():.2f}")
        with col2:
            st.metric("الوسيط", f"{df[rating_col].median():.2f}")
        with col3:
            st.metric("الأعلى", f"{df[rating_col].max():.2f}")
        with col4:
            st.metric("الأدنى", f"{df[rating_col].min():.2f}")

if st.checkbox('تحليل السنوات', key='year_analysis'):
    st.subheader('📅 تحليل السنوات')
    year_cols = [col for col in df.columns if 'year' in col.lower()]
   
    if year_cols:
        year_col = year_cols[0]
        fig, ax = plt.subplots(figsize=(12, 6))
        df[year_col].hist(bins=30, ax=ax, color='lightcoral', edgecolor='black')
        ax.set_title('توزيع سنوات الإصدار')
        ax.set_xlabel('السنة')
        ax.set_ylabel('عدد الأفلام')
        st.pyplot(fig)

if st.checkbox('تحليل الأنواع', key='genre_analysis'):
    st.subheader('🎭 تحليل أنواع الأفلام')
    genre_cols = [col for col in df.columns if 'genre' in col.lower()]
   
    if genre_cols:
        genre_col = genre_cols[0]
        df[genre_col] = df[genre_col].astype(str)
        all_genres = df[genre_col].str.split(',', expand=True).stack().str.strip()
        genre_counts = all_genres.value_counts().head(10)
       
        fig, ax = plt.subplots(figsize=(12, 8))
        genre_counts.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('أكثر 10 أنواع أفلام شيوعاً')
        ax.set_xlabel('عدد الأفلام')
        ax.invert_yaxis()
        st.pyplot(fig)

# قسم التنبؤات المستقبلية
st.markdown('<h2 class="section-header">🔮 التنبؤات المستقبلية</h2>', unsafe_allow_html=True)

if st.checkbox('عرض تنبؤات مستقبلية', key='future_predictions'):
    st.subheader('📈 تنبؤ تقييمات الأفلام المستقبلية')
   
    if rating_col and year_col:
        # تحضير البيانات للتنبؤ
        prediction_df = df.dropna(subset=[rating_col, year_col])
        X = prediction_df[year_col].values.reshape(-1, 1)
        y = prediction_df[rating_col].values
       
        if len(X) > 0:
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
           
            # تدريب النموذج
            model = LinearRegression()
            model.fit(X_train, y_train)
           
            # التنبؤ
            future_years = np.array([[2024], [2025], [2026], [2027], [2028]])
            predictions = model.predict(future_years)
           
            # عرض النتائج
            st.markdown("""
            <div class="prediction-card">
                <h3>التنبؤ بمتوسط تقييمات الأفلام المستقبلية</h3>
            </div>
            """, unsafe_allow_html=True)
           
            col1, col2, col3, col4, col5 = st.columns(5)
            cols = [col1, col2, col3, col4, col5]
           
            for i, (col, year, pred) in enumerate(zip(cols, [2024, 2025, 2026, 2027, 2028], predictions)):
                with col:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; background: #f0f8ff; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <h3>{year}</h3>
                        <h2 style="color: #E50914;">{pred:.2f}</h2>
                        <p>⭐</p>
                    </div>
                    """, unsafe_allow_html=True)
           
            # رسم بياني للتنبؤات
            fig, ax = plt.subplots(figsize=(12, 6))
            years = list(range(int(df[year_col].min()), 2029))
            future_preds = model.predict(np.array(years).reshape(-1, 1))
           
            # تقسيم البيانات إلى تاريخية وتنبؤية
            historical_years = [y for y in years if y <= 2023]
            future_years = [y for y in years if y > 2023]
           
            historical_preds = future_preds[:len(historical_years)]
            future_preds = future_preds[len(historical_years):]
           
            ax.plot(historical_years, historical_preds, 'b-', label='التقييمات التاريخية', linewidth=2)
            ax.plot(future_years, future_preds, 'r--', label='التنبؤات المستقبلية', linewidth=2)
            ax.set_xlabel('السنة', fontsize=12)
            ax.set_ylabel('التقييم المتوقع', fontsize=12)
            ax.set_title('تنبؤ تقييمات الأفلام المستقبلية', fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
           
            # دقة النموذج
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            st.write(f"**دقة النموذج:** متوسط الخطأ المطلق: {mae:.3f}")
           
        else:
            st.warning("لا توجد بيانات كافية للتنبؤ")
    else:
        st.warning("يجب وجود أعمدة السنة والتقييم للتنبؤ")

if st.checkbox('تنبؤ شعبية الأنواع المستقبلية', key='genre_prediction'):
    st.subheader('🎭 تنبؤ شعبية الأنواع المستقبلية')
   
    genre_cols = [col for col in df.columns if 'genre' in col.lower()]
    if genre_cols and year_col:
        genre_col = genre_cols[0]
       
        # تحليل تطور شعبية الأنواع
        genre_trends = {}
        for genre in popular_genres:
            genre_counts = []
            years = sorted(df[year_col].unique())
           
            for year in years:
                count = len(df[(df[year_col] == year) &
                             (df[genre_col].str.contains(genre, case=False, na=False))])
                genre_counts.append(count)
           
            if sum(genre_counts) > 0:  # فقط الأنواع الموجودة
                genre_trends[genre] = (years, genre_counts)
       
        # عرض أكثر الأنواع نمواً
        st.markdown("""
        <div class="prediction-card">
            <h3>الأنواع الأسرع نمواً</h3>
        </div>
        """, unsafe_allow_html=True)
       
        growth_rates = {}
        for genre, (years, counts) in genre_trends.items():
            if len(counts) >= 5:
                # حساب معدل النمو في آخر 3 سنوات
                recent_growth = 0
                if len(counts) >= 4:
                    recent_counts = counts[-4:]
                    growth_rates_recent = []
                    for i in range(1, len(recent_counts)):
                        if recent_counts[i-1] > 0:
                            growth = (recent_counts[i] - recent_counts[i-1]) / recent_counts[i-1] * 100
                            growth_rates_recent.append(growth)
                    if growth_rates_recent:
                        recent_growth = sum(growth_rates_recent) / len(growth_rates_recent)
               
                growth_rates[genre] = recent_growth
       
        # عرض أفضل 5 أنواع من حيث النمو
        top_growing = sorted(growth_rates.items(), key=lambda x: x[1], reverse=True)[:5]
       
        for genre, growth in top_growing:
            emoji = get_genre_emoji(genre)
            st.write(f"{emoji} **{genre}**: {growth:+.1f}% نمو سنوي")
           
            # رسم اتجاه النوع
            years, counts = genre_trends[genre]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(years, counts, 'o-', color='#E50914')
            ax.set_title(f'اتجاه نمو نوع: {genre}')
            ax.set_xlabel('السنة')
            ax.set_ylabel('عدد الأفلام')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
   
    else:
        st.warning("يجب وجود أعمدة السنة والنوع للتنبؤ")

# البحث في البيانات
st.sidebar.header("🔎 بحث متقدم")
search_term = st.sidebar.text_input("ابحث عن فيلم، ممثل، أو مخرج:")
if search_term:
    search_result = df[df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
    st.sidebar.write(f"تم العثور على {len(search_result)} نتيجة")
   
    if len(search_result) > 0:
        st.header(f"نتائج البحث عن: '{search_term}'")
        st.dataframe(search_result.head(10))

# قسم طباعة التقرير
st.markdown('<h2 class="section-header">📋 تقرير التحليل</h2>', unsafe_allow_html=True)

# إنشاء تقرير تفاعلي
with st.expander("📊 عرض تقرير التحليل الكامل", expanded=False):
    st.markdown('<div class="report-section">', unsafe_allow_html=True)
   
    st.subheader("📋 ملخص التقرير")
    col1, col2 = st.columns(2)
   
    with col1:
        st.markdown("**المعلومات العامة:**")
        st.write(f"- تاريخ إنشاء التقرير: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.write(f"- عدد الأفلام الإجمالي: {df.shape[0]}")
        st.write(f"- عدد الأعمدة: {df.shape[1]}")
        st.write(f"- حجم البيانات: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
   
    with col2:
        st.markdown("**إحصائيات التقييمات:**")
        if rating_col:
            st.write(f"- متوسط التقييم: {df[rating_col].mean():.2f}")
            st.write(f"- أعلى تقييم: {df[rating_col].max():.2f}")
            st.write(f"- أدنى تقييم: {df[rating_col].min():.2f}")
        if year_col:
            st.write(f"- أقدم فيلم: {int(df[year_col].min())}")
            st.write(f"- أحدث فيلم: {int(df[year_col].max())}")
   
    st.markdown("**نتائج التصفية:**")
    st.write(f"- الأنواع المختارة: {', '.join(selected_genres) if selected_genres else 'جميع الأنواع'}")
    if year_col and 'year_range' in locals():
        st.write(f"- نطاق السنوات: {year_range[0]} - {year_range[1]}")
    if rating_col and 'rating_range' in locals():
        st.write(f"- نطاق التقييم: {rating_range[0]:.1f} - {rating_range[1]:.1f}")
    st.write(f"- عدد الأفلام بعد التصفية: {len(filtered_df)}")
   
    st.markdown("**الأفلام الأعلى تقييمًا:**")
    if rating_col and len(filtered_df) > 0:
        top_movies = filtered_df.nlargest(5, rating_col)[['title', rating_col, year_col if year_col else '']]
        st.dataframe(top_movies)
   
    st.markdown("</div>", unsafe_allow_html=True)

# وظيفة لتحميل التقرير كملف نصي
def create_download_link(content, filename, title):
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" class="download-button">{title}</a>'
    return href

# إنشاء محتوى التقرير
report_content = f"""
تقرير تحليل بيانات الأفلام
تم إنشاؤه في: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{'='*50}

المعلومات العامة:
- عدد الأفلام الإجمالي: {df.shape[0]}
- عدد الأعمدة: {df.shape[1]}
- حجم البيانات: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

{'='*50}

إحصائيات التقييمات:
{ f"- متوسط التقييم: {df[rating_col].mean():.2f}" if rating_col else ""}
{ f"- أعلى تقييم: {df[rating_col].max():.2f}" if rating_col else ""}
{ f"- أدنى تقييم: {df[rating_col].min():.2f}" if rating_col else ""}
{ f"- أقدم فيلم: {int(df[year_col].min())}" if year_col else ""}
{ f"- أحدث فيلم: {int(df[year_col].max())}" if year_col else ""}

{'='*50}

نتائج التصفية:
- الأنواع المختارة: {', '.join(selected_genres) if selected_genres else 'جميع الأنواع'}
{ f"- نطاق السنوات: {year_range[0]} - {year_range[1]}" if year_col and 'year_range' in locals() else ""}
{ f"- نطاق التقييم: {rating_range[0]:.1f} - {rating_range[1]:.1f}" if rating_col and 'rating_range' in locals() else ""}
- عدد الأفلام بعد التصفية: {len(filtered_df)}

{'='*50}

التنبؤات المستقبلية:
"""

# إضافة التنبؤات إلى التقرير
if rating_col and year_col:
    try:
        prediction_df = df.dropna(subset=[rating_col, year_col])
        X = prediction_df[year_col].values.reshape(-1, 1)
        y = prediction_df[rating_col].values
       
        if len(X) > 0:
            model = LinearRegression()
            model.fit(X, y)
           
            future_years = [2024, 2025, 2026, 2027, 2028]
            predictions = model.predict(np.array(future_years).reshape(-1, 1))
           
            report_content += "\nتنبؤ تقييمات الأفلام المستقبلية:\n"
            for year, pred in zip(future_years, predictions):
                report_content += f"- عام {year}: {pred:.2f} ⭐\n"
    except:
        report_content += "\n(غير متوفر)\n"

report_content += f"""
{'='*50}

ملاحظات:
- تم إنشاء هذا التقرير تلقائيًا باستخدام Netflix & IMDb Explorer
- التاريخ: {datetime.now().strftime('%Y-%m-%d')}
"""

# عرض أزرار التحميل
st.markdown("### 💾 تحميل التقرير")
col1, col2 = st.columns(2)

with col1:
    st.markdown(create_download_link(report_content, "film_analysis_report.txt", "📥 تحميل التقرير نصي"), unsafe_allow_html=True)

with col2:
    # إنشاء تقرير HTML جميل
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>تقرير تحليل الأفلام</title>
        <style>
            body {{ font-family: Arial, sans-serif; direction: rtl; margin: 40px; background-color: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #E50914 0%, #B20710 100%); color: white; padding: 30px; text-align: center; border-radius: 15px; margin-bottom: 30px; }}
            .section {{ background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .stat-item {{ text-align: center; padding: 15px; background: #f8f9fa; border-radius: 10px; width: 22%; }}
            .footer {{ text-align: center; margin-top: 40px; color: #6c757d; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>تقرير تحليل بيانات الأفلام</h1>
            <p>تم إنشاؤه في: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
       
        <div class="section">
            <h2>المعلومات العامة</h2>
            <div class="stats">
                <div class="stat-item">
                    <h3>{df.shape[0]}</h3>
                    <p>عدد الأفلام الإجمالي</p>
                </div>
                <div class="stat-item">
                    <h3>{df.shape[1]}</h3>
                    <p>عدد الأعمدة</p>
                </div>
                <div class="stat-item">
                    <h3>{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB</h3>
                    <p>حجم البيانات</p>
                </div>
            </div>
        </div>
       
        <div class="section">
            <h2>نتائج التصفية</h2>
            <p><strong>الأنواع المختارة:</strong> {', '.join(selected_genres) if selected_genres else 'جميع الأنواع'}</p>
            <p><strong>عدد الأفلام بعد التصفية:</strong> {len(filtered_df)}</p>
        </div>
       
        <div class="footer">
            <p>تم إنشاء هذا التقرير تلقائيًا باستخدام Netflix & IMDb Explorer</p>
            <p>التاريخ: {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
    </body>
    </html>
    """
    st.markdown(create_download_link(html_report, "film_analysis_report.html", "📊 تحميل التقرير HTML"), unsafe_allow_html=True)

# تذييل الصفحة
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 2rem;">
    <p>تم تطوير هذا التطبيق باستخدام Streamlit.....و seaborn و Pandas و Matplotlib</p>
    <p>🎬 Netflix & IMDb Explorer © 2025</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.info("""
ℹ️ **معلومات عن التطبيق**
- يعمل بدون إنترنت
- البيانات محلية من ملف CSV
- يدعم التصفية المتقدمة
- واجهة تفاعلية جميلة
- إمكانية طباعة التقارير
- تنبؤات مستقبلية
- نظام توصيات متقدم
""")