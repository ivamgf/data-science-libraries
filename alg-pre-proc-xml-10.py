import os
import nltk
import xml.etree.ElementTree as ET
import spacy
from gensim.models import Word2Vec
import streamlit as st

nltk.download('punkt')

path = "C:\\Dataset-TRT"
files = os.listdir(path)

nlp = spacy.load("pt_core_news_sm")

def replace_words(text):
    word_replacements = {
        "LimitaÃ§Ã£o da condenaÃ§Ã£o aos valores dos pedidos": "Limitacao da condenacao aos valores dos pedidos",
        "AssÃ©dio moral": "Assedio moral",
        "HonorÃ¡rios sucumbenciais": "Honorarios sucumbenciais",
        "Estabilidade acidentÃ¡ria": "Estabilidade acidentaria",
        "DoenÃ§a ocupacional": "Doenca ocupacional"
    }
    for old_word, new_word in word_replacements.items():
        text = text.replace(old_word, new_word)
    return text

def replace_expression(text):
    expressions = {
        "LimitaÃ§Ã£o da condenaÃ§Ã£o aos valores dos pedidos": "Limitacao da condenacao aos valores dos pedidos",
        "AssÃ©dio moral": "Assedio moral",
        "HonorÃ¡rios sucumbenciais": "Honorarios sucumbenciais",
        "Estabilidade acidentÃ¡ria": "Estabilidade acidentaria",
        "DoenÃ§a ocupacional": "Doenca ocupacional"
    }
    for expression, replacement in expressions.items():
        text = text.replace(expression, replacement)
    return text

st.title("Dataset Cleaning and Word Embeddings")

# Print the file list
st.subheader("Arquivos encontrados no diretório:")
for file in files:
    if file.endswith(".xml"):
        st.write(file)

        xml_file = os.path.join(path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        st.subheader("Conteúdo do arquivo " + file + ":")
        sentences = []  # List to store all sentences
        for element in root.iter("webanno.custom.Judgmentsentity"):
            if (
                "sofa" in element.attrib and
                "begin" in element.attrib and
                "end" in element.attrib and
                "Instance" in element.attrib and
                "Value" in element.attrib
            ):
                sofa = element.attrib["sofa"]
                begin = element.attrib["begin"]
                end = element.attrib["end"]
                instance = element.attrib["Instance"]
                value = element.attrib["Value"]

                instance = replace_words(instance)
                value = replace_words(value)

                for child in element:
                    if child.text:
                        text = child.text.strip().lower()
                        text = replace_words(text)
                        text = replace_expression(text)
                        doc = nlp(text)
                        tokens = [token.text for token in doc]
                        sentences.append(tokens)

                        model = Word2Vec(min_count=1, workers=2)
                        model.build_vocab(sentences)
                        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

                        st.subheader("Vocabulary:")
                        for word in model.wv.key_to_index.keys():
                            st.write("Token:" + word)
                            st.write("Instance:" + instance.encode("utf-8").decode("utf-8"))
                            st.write("Value:" + value.encode("utf-8").decode("utf-8"))
                            st.write("Embedding:")
                            st.write(model.wv[word])
