# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:12:54 2020

@author: IAN
"""
import streamlit as st
import pandas as pd

import re
import pickle
from nltk.stem.snowball import SnowballStemmer
import joblib
from nltk.corpus import stopwords


def text_splitter(text):
    return text.split()


classifier = joblib.load('lr_with_more_title_weight.pkl')
tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

stem = SnowballStemmer('english')

stop_words = set(stopwords.words('english'))


def striphtml(question):
    cleanr = re.compile('<.*?>')
    cleanquestion = re.sub(cleanr, ' ', str(question))
    return cleanquestion

# Predicting one query point


@st.cache
def clean_question(title, question):
    # strip the code from the question
    question = re.sub('<code>(.*?)</code>', '', question,
                      flags=re.MULTILINE | re.DOTALL)
    # strip any html code that may be in the question
    question = striphtml(question.encode('utf-8'))

    title = title.encode('utf-8')

    question = str(title) + " " + str(title) + " " + \
        str(title) + " " + str(question)

    question = re.sub(r'[^A-Za-z]+', ' ', question)

    question = ' '.join(str(stem.stem(j)) for j in question.split(
    ) if j not in stop_words and (len(j) != 1 or j == 'c'))

    return question


@st.cache
def get_table_download_link(df):
    """
    Generates a link allowing the data 
    in a given dataframe to be dowloaded.
    input: dataframe
    output: link
    """
    import base64
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return '<a href="data:file/csv;base64,{}" download="StackOverflow_TagPrediction.csv">Download CSV file</a>'.format(b64)


def run():
    from PIL import Image
    image = Image.open('favicon.png')
    image_tag = Image.open('tag.png')

    st.image(image, use_column_width=False)

    add_select_box = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))

    st.sidebar.info(
        "This app is a web app for the Kaggle this competition https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction")

    st.sidebar.image(image_tag)
    st.title('Stack Overflow Tag Predictor')

    st.header('AI that predicts tags for a post')
    st.subheader('Title and Question for the Stack Overflow post.')

    if add_select_box == 'Online':
        title = st.text_input('Title')
        body = st.text_area('Body')
        output = ''
        input_dict = {'title': title, 'body': body}

        prediction_array = classifier.predict_proba(
            tfidf_vectorizer.transform([clean_question(title, body)]))[0]


        if st.button('Predict'):

            tags = [tfidf_vectorizer.get_feature_names()[i]
                    for i in prediction_array.argsort()[-3:][::-1]]

            output = ', '.join(tags)

        st.success('0utput: {}'.format(output))

    if add_select_box == 'Batch':
        file_upload = st.file_uploader(
            'Upload csv file for prediction with columns: Id, Title and Body', type=['csv'])

        if file_upload is not None:
            try:
                data = pd.read_csv(file_upload)
                list_of_questions = []
                for index in range(data.shape[0]):
                    row = data.iloc[index]
                    title, body = row[1], row[2]
                    question = clean_question(title, body)
                    list_of_questions.append(question)
                list_of_questions_vec = tfidf_vectorizer.transform(
                    list_of_questions)
                prediction_array = classifier.predict_proba(list_of_questions_vec)
                
                tags = [list(i.argsort()[-3:][::-1]) for i in prediction_array]
                list_of_tags = []
                for lst in tags:
                    tags_for_each = []
                    for el in lst:
                        tags_for_each.append(
                            tfidf_vectorizer.get_feature_names()[el])
                    list_of_tags.append(tags_for_each)
                # Append the real tags to the data input
                data['Tags'] = [', '.join(i) for i in list_of_tags]
                st.dataframe(data.style.highlight_max(axis=0))
                st.markdown(get_table_download_link(data), unsafe_allow_html=True)
            except:
                st.write('The file uploaded was not of the required format. Make sure it has columns Id, Title and Body only in that order then upload again.')


if __name__ == '__main__':
    run()
