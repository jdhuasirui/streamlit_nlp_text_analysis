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

from datetime import datetime
# import open files pkgs/metadata extraction
import os
# images
from PIL import Image
import exifread
# audio
import mutagen
# pdf
from PyPDF2 import PdfFileReader

# HTML
matadata_wiki = """
Metadata is defined as the data providing infomation about one or more aspects of the data.
"""

HTML_BANNER = """
    <div style="background-color:#464e5f;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">MetaData Extractor App </h1>
    </div>
"""
# define load image function
def load_image(image_file):
    img = Image.open(image_file)
    return img

# import extrernal pkgs
from app_utils import *

# define function to get downloads
def make_downloadable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "results_{}_.csv".format(timestr)
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

def text_analysis(raw_text, num_of_most_common=5):
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

# define function to get human readable time
def get_readable_time(mytime):
    return datetime.fromtimestamp(mytime).strftime("%Y-%m-%d-%H:%M")


def main():
    st.title("Streamlit Apps")
    menu = ["📚NLP", "📚NLP(Upload File)", "Metadata Extractor Home", "Image Metadata Extractor", "Audio Metadata Extractor", "DocumentFiles Metadata Extractor", "About"]

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "📚NLP":
        st.subheader("Home: Text Analysis")
        raw_text = st.text_area("Enter Text Here")
        num_of_most_common = st.sidebar.number_input("Most Common Tokens", 5, 15)
        if st.button("Analyze"):
            text_analysis(raw_text=raw_text, num_of_most_common=num_of_most_common)

    elif choice == "📚NLP(Upload File)":
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

            text_analysis(raw_text=raw_text, num_of_most_common=num_of_most_common)
    
    elif choice == "Metadata Extractor Home":
        st.subheader('Image Metadata Extractor')
        # description
        st.write(matadata_wiki)
        # create 3 columns
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.expander("Get Image Metadata"):
                st.info("Image Metadata")
                st.text("Upload JPEG, JPG, PNG Images")
        with col2:
            with st.expander("Get Audio Metadata"):
                st.info("Audio Metadata")
                st.text("Upload MP3, Ogg Audios")
        with col3:
            with st.expander("Get Document Metadata"):
                st.info("Document Files Metadata")
                st.text("Upload PDF, Docx Files")

    elif choice == "Image Metadata Extractor":
        st.subheader('Image Metadata Extractor')
        image_file = st.file_uploader("Upload Image",type=["png","jpg","jpeg"])
        if image_file is not None:
            with st.expander("File Stats"):
                file_details = {"FileName":image_file.name,
                                "FileSize":image_file.size,
                                "FileType":image_file.type}
                st.write(file_details)

                statsinfo = os.stat(image_file.readable())
                st.write(statsinfo)
                stats_details = {"Accessed_Time":get_readable_time(statsinfo.st_atime),
                                 "Creation_Time":get_readable_time(statsinfo.st_ctime),
                                 "Modefication_Time":get_readable_time(statsinfo.st_mtime)}
                st.write(stats_details)

                # combine all file details
                file_details_combined = {"FileName":image_file.name,
                                         "FileSize":image_file.size,
                                         "FileType":image_file.type,
                                         "Accessed_Time":get_readable_time(statsinfo.st_atime),
                                         "Creation_Time":get_readable_time(statsinfo.st_ctime),
                                         "Modefication_Time":get_readable_time(statsinfo.st_mtime)}
                
                # convert to DataFrame
                df_file_details_combined = pd.DataFrame(file_details_combined.items(),columns=['Meta Tag','Value']).astype(str)
                st.dataframe(df_file_details_combined)
            
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("View Image"):
                    img = load_image(image_file)
                    st.image(img, width=250)
            with col2:
                with st.expander("Default(JPEG)"):
                    st.info("Using Pillow")
                    img = load_image(image_file)
                    #st.write(dir(img))
                    img_details = {"format":img.format,
                                   "format_description":img.format_description,
                                   "filename":img.filename,
                                   "size":img.size,
                                   "height":img.height,
                                   "width":img.width,
                                   "info":img.info}
                    df_img_details = pd.DataFrame(img_details.items(),columns=['Meta Tag','Value']).astype(str)
                    st.dataframe(df_img_details)   
            # layout for forensic
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                with st.expander("Exifread Tool"):
                    meta_tags = exifread.process_file(image_file)
                    df_meta_tags = pd.DataFrame(meta_tags.items(),columns=['Meta Tag','Value']).astype(str)
                    st.dataframe(df_meta_tags) 
            with fcol2:
                pass
            
            with st.expander("Download Results"):
                final_df= pd.concat([df_file_details_combined, df_img_details, df_meta_tags])
                make_downloadable(final_df)

    elif choice == "Audio Metadata Extractor":
        st.subheader('Audio Metadata Extractor')
        audio_file = st.file_uploader("Upload Audio",type=["mp3","ogg","wav","m4a"])
        if audio_file is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.audio(audio_file.read())
            with col2:
                with st.expander("File Stats"):
                    file_details = {"FileName":audio_file.name,
                                "FileSize":audio_file.size,
                                "FileType":audio_file.type}
                    st.write(file_details)

                    statsinfo = os.stat(audio_file.readable())
                    st.write(statsinfo)
                    stats_details = {"Accessed_Time":get_readable_time(statsinfo.st_atime),
                                 "Creation_Time":get_readable_time(statsinfo.st_ctime),
                                 "Modefication_Time":get_readable_time(statsinfo.st_mtime)}
                    st.write(stats_details)

                    # combine all file details
                    file_details_combined = {"FileName":audio_file.name,
                                         "FileSize":audio_file.size,
                                         "FileType":audio_file.type,
                                         "Accessed_Time":get_readable_time(statsinfo.st_atime),
                                         "Creation_Time":get_readable_time(statsinfo.st_ctime),
                                         "Modefication_Time":get_readable_time(statsinfo.st_mtime)}
                
                    # convert to DataFrame
                    df_file_details_combined = pd.DataFrame(file_details_combined.items(),columns=['Meta Tag','Value']).astype(str)
                    st.dataframe(df_file_details_combined)
            
            with st.expander("Metadata with Mutagen"):
                meta_tags = mutagen.File(audio_file)
                df_audio_details = pd.DataFrame(file_details.items(), columns=['Meta Tag', "Value"]).astype(str)
                st.dataframe(df_audio_details)

            with st.expander("Download Results"):
                final_df = pd.concat([df_file_details_combined, df_audio_details])
                make_downloadable(final_df)

    elif choice == "DocumentFiles Metadata Extractor":
        st.subheader('DocumentFiles Metadata Extractor Metadata Extractor')
        text_file= st.file_uploader("Upload File", type=['PDF'])

        if text_file is not None:
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("File Stats"):
                    file_details = {"FileName":text_file.name,
                                "FileSize":text_file.size,
                                "FileType":text_file.type}
                    st.write(file_details)

                    statsinfo = os.stat(text_file.readable())
                    st.write(statsinfo)
                    stats_details = {"Accessed_Time":get_readable_time(statsinfo.st_atime),
                                 "Creation_Time":get_readable_time(statsinfo.st_ctime),
                                 "Modefication_Time":get_readable_time(statsinfo.st_mtime)}
                    st.write(stats_details)

                    # combine all file details
                    file_details_combined = {"FileName":text_file.name,
                                         "FileSize":text_file.size,
                                         "FileType":text_file.type,
                                         "Accessed_Time":get_readable_time(statsinfo.st_atime),
                                         "Creation_Time":get_readable_time(statsinfo.st_ctime),
                                         "Modefication_Time":get_readable_time(statsinfo.st_mtime)}
                
                    # convert to DataFrame
                    df_file_details_combined = pd.DataFrame(file_details_combined.items(),columns=['Meta Tag','Value']).astype(str)
                    st.dataframe(df_file_details_combined)

            # Extraction
            with col2:
                with st.expander("Metadata with Mutagen"):
                    pdf_file = PdfFileReader(text_file)
                    pdf_info = pdf_file.getDocumentInfo()
                    df_pdf_info = pd.DataFrame(pdf_info.items(),columns=['Meta Tag','Value']).astype(str)
                    st.dataframe(df_pdf_info)

            with st.expander("Download Results"):
                final_df = pd.concat([df_file_details_combined, df_pdf_info])
                make_downloadable(final_df)



    else:
        st.subheader("About")


if __name__ == "__main__":
    main()
