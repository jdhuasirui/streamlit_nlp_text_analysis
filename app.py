from pandas.io import html
import streamlit as st
import streamlit.components.v1 as stc

# Load EDA pkgs
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("Agg")

# import extrernal pkgs
from app_utils import *

# define function to get downloads
def make_downloadable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "nlp_results_{}_.csv".format(timestr)
    st.markdown("### **Download CVS file**")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download csv file</a>'
    st.markdown(href, unsafe_allow_html=True)


# wordcloud
from wordcloud import WordCloud


def plot_wordcloud(my_text):
    my_wordcloud = WordCloud().generate(my_text)
    fig = plt.figure()
    plt.imshow(my_wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)


def main():
    st.title("NLP App with Streamlit")
    menu = ["Home", "NLP(Files)", "About"]

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home: Text Analysis")
        raw_text = st.text_area("Enter Text Here")
        num_of_most_common = st.sidebar.number_input("Most Common Tokens", 5, 15)
        st.write(raw_text)
        if st.button("Analyze"):

            with st.expander("Original Text"):
                st.write(raw_text)

            with st.expander("Text Analysis"):
                token_result_df = text_analyzer(raw_text)
                st.dataframe(token_result_df)

            with st.expander("Entities"):
                entity_result = render_entities(raw_text)
                stc.html(entity_result, height=1000, scrolling=True)

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("Word Stats"):
                    st.info("Word Statistics")
                    docx = nt.TextFrame(raw_text)
                    st.write(docx.word_stats())

                with st.expander("Top Keywords"):
                    st.info("Top Keywords/Tokens")
                    processed_text = nfx.remove_stopwords(raw_text)
                    keywords = get_most_common_tokens(
                        processed_text, num=num_of_most_common
                    )
                    st.write(keywords)

                with st.expander("Sentiment"):
                    sent_result = get_sentiment(raw_text)
                    st.write(sent_result)

            with col2:
                with st.expander("Plot Word Freq"):
                    fig = plt.figure()
                    keywords_frequence = get_most_common_tokens(
                        processed_text, num=num_of_most_common * 2
                    )
                    plt.bar(keywords_frequence.keys(), keywords_frequence.values())
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                with st.expander("Plot Part of Speech"):
                    try:
                        fig = plt.figure()
                        sns.countplot(token_result_df["PoS"])
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    except:
                        st.warning("Insufficient Data: Must be more than 2")

                with st.expander("Plot Wordcloud"):
                    try:
                        plot_wordcloud(raw_text)
                    except:
                        st.warning("Insufficient Data: Must be more than 2")


            with st.expander("Download Text Analysis"):
                make_downloadable(token_result_df)

    elif choice == "NLP(Files)":
        st.subheader("NLP Task")

        text_file = st.file_uploader("Upload Files", type=["pdf", "docx", "txt"])
        num_of_most_common = st.sidebar.number_input("Most Common Tokens", 5, 15)

        if text_file is not None:
            if text_file.type == "application/pdf":
                raw_text = read_pdf(text_file)
            elif text_file.type == "text/plain":
                raw_text = str(text_file.read(), "utf-8")
            else:
                raw_text = docx2txt.process(text_file)

            with st.expander("Original Text"):
                st.write(raw_text)

            with st.expander("Text Analysis"):
                token_result_df = text_analyzer(raw_text)
                st.dataframe(token_result_df)

            with st.expander("Entities"):
                entity_result = render_entities(raw_text)
                stc.html(entity_result, height=1000, scrolling=True)

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("Word Stats"):
                    st.info("Word Statistics")
                    docx = nt.TextFrame(raw_text)
                    st.write(docx.word_stats())

                with st.expander("Top Keywords"):
                    st.info("Top Keywords/Tokens")
                    processed_text = nfx.remove_stopwords(raw_text)
                    keywords = get_most_common_tokens(
                        processed_text, num=num_of_most_common
                    )
                    st.write(keywords)

                with st.expander("Sentiment"):
                    sent_result = get_sentiment(raw_text)
                    st.write(sent_result)

            with col2:
                with st.expander("Plot Word Freq"):
                    fig = plt.figure()
                    keywords_frequence = get_most_common_tokens(
                        processed_text, num=num_of_most_common * 2
                    )
                    plt.bar(keywords_frequence.keys(), keywords_frequence.values())
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                with st.expander("Plot Part of Speech"):
                    try:
                        fig = plt.figure()
                        sns.countplot(token_result_df["PoS"])
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    except:
                        st.warning("Insufficient Data: Must be more than 2")

                with st.expander("Plot Wordcloud"):
                    try:
                        plot_wordcloud(raw_text)
                    except:
                        st.warning("Insufficient Data: Must be more than 2")

            with st.expander("Download Text Analysis"):
                make_downloadable(token_result_df)
    else:
        st.subheader("About")


if __name__ == "__main__":
    main()
