# Algorithm 2 - dataset cleaning and pre-processing texts in PDF

# Imports
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import PyPDF2

# Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Define the list of stopwords for the Portuguese language
stop_words = set(stopwords.words('portuguese'))

# Use double backslashes in file or directory path
path = "C:\\Dataset2"

# Get a list of files in the directory
files = os.listdir(path)

# Print the file list
print("Files found in the directory:")
for file in files:
    print(file)

    # Open the PDF file
    with open(os.path.join(path, file), 'rb') as c:
        # Read the PDF content
        pdfReader = PyPDF2.PdfFileReader(c)
        num_pages = pdfReader.numPages
        content = ""
        for page in range(num_pages):
            pageObj = pdfReader.getPage(page)
            content += pageObj.extractText()

        # Tokenize the content into sentences
        sentences = sent_tokenize(content)

        # Print sentences without stop words
        print("Sentences in the file " + file + " without stop words:")
        for sentence in sentences:
            # Flag for tokenizing and stemming the sentences after the keyword "ACORDAM"
            if "RELATÓRIO" in sentence:
                tokenize_after_keyword = True
            else:
                tokenize_after_keyword = False

            if tokenize_after_keyword:
                # Tokenize each sentence into words
                words = word_tokenize(sentence.lower())

                # Remove stop words and punctuation
                filtered_words = [
                    word
                    for word in words
                    if word not in stop_words and
                       word not in punctuation
                ]

                # Put the filtered words together into a sentence again
                filtered_sentence = ' '.join(filtered_words)

                # Print the sentences
                print(filtered_sentence)
