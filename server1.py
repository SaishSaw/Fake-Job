import streamlit as st 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.corpus import words as nltk_words
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import pickle
import re

#Loading trained model from pickle file.
with open('model.pkl','rb') as file:
    model_rd = pickle.load(file)

# Defining function for creating model which gives output.
def model(loc_num,employ_vec,industry_vec,numeric_val,has_company_logo,has_questions):
    model_pred = model_rd.predict([[loc_num,employ_vec,industry_vec,numeric_val,has_company_logo,has_questions]])
    if model_pred == 0:
        return "Yippee!!! The job post appears to be genuine."
    else:
        return "Beware!! The job post seems to be FAKE."

#Preprocessing steps for text data.
le = LabelEncoder()
def load_preprocessing_model():
    return Preprocessing()

class Preprocessing:
    def __init__(self):
        pass

    def process(self, text):
        pattern = re.compile(r'\s\'|(?<=\s)\'|\'$|[^a-zA-Z\s\']+')
        stop = stopwords.words('english')
        eng_word = set(nltk_words.words())
        snow = SnowballStemmer('english')
        lmztr = WordNetLemmatizer()

        text_clean = pattern.sub(" ", text).lower()
        words = text_clean.split()
        without_stop = [word for word in words if word not in stop]
        meaning_words = [word for word in without_stop if word in eng_word]
        stemmed_words = [snow.stem(word) for word in meaning_words]
        lemmatized_words = [lmztr.lemmatize(word, pos='v') for word in stemmed_words]

        return lemmatized_words

def lsa_summary(text):
    num_topics = 2
    num_words = 5
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([" ".join(text)])
    
    if tfidf_matrix.shape[1] < 2:
        return []
    
    lsa = TruncatedSVD(n_components=num_topics, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)
    
    top_words_indices = lsa_matrix[0].argsort()[::-1][:num_words]
    terms = vectorizer.get_feature_names_out()
    summary = [(terms[i], tfidf_matrix[0, i]) for i in top_words_indices]
    
    return summary

#Streamlit app interface
def main():
    st.title('Fake Job Detector')
    html_temp = """
    <div style="background-color: black; padding: 20px">
        <h2 style="color: white; text-align: center;">Streamlit Fake Job Posting Detector</h2>
    </div>   
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    location = st.text_input('Location','Type Here (For eg India as IND)')
    employment_type = st.text_input('Employment Type','Type here')
    industry = st.text_input('Industry Type','Type here')
    has_company_logo = st.radio('Company Logo',[0,1], format_func= lambda x : 'In case company has no logo' if x == 0 else 'Incase of logo')
    has_questions = st.radio('Questions',[0,1],format_func = lambda x : "In case of no questions" if x == 0 else "Incase of any questions")
    job_info = st.text_input('Job information','This may include Job Description,benefits,requirements etc.')

    loc_num = np.array(location).reshape(1,-1)
    loc_num = le.fit_transform(loc_num).ravel()

    employ_vec = np.array(employment_type).reshape(1,-1)
    employ_vec = le.fit_transform(employ_vec).ravel()

    industry_vec = np.array(industry).reshape(1,-1)
    industry_vec = le.fit_transform(industry_vec).ravel()

    preprocessed = load_preprocessing_model()
    processor = preprocessed.process(job_info)
    numeric_val = lsa_summary(processor)

    for inner_val in numeric_val:
        number = inner_val[1]

    if st.button('Predict'):
        final_output = model(loc_num[0],employ_vec[0],industry_vec[0],number,has_company_logo,has_questions)
        st.success(final_output)

    if st.button('About'):
        st.text('Fake job posting web app')
        st.text('Built using streamlit')
        st.text('Created by Saish')

if __name__ == "__main__":
    main()

