import os
import nltk
import PyPDF2
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

# Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Use double backslashes in file or directory path
path = "C:\\Dataset2"

# Get a list of files in the directory
files = os.listdir(path)

# Define the list of stopwords for the Portuguese language and punctuations
stop_words = set(stopwords.words('portuguese') + list(string.punctuation))

# Print the file list
print("Files found in the directory:")
for filename in files:
    print(filename)
    if filename.endswith('.pdf'):

        # Open the PDF file in binary mode
        with open(os.path.join(path, filename), 'rb') as f:
            # Read the PDF content
            pdf_reader = PyPDF2.PdfFileReader(f)

            # Iterate through PDF file pages and extract text
            text = ""
            for page_num in range(pdf_reader.getNumPages()):
                page = pdf_reader.getPage(page_num)
                text += page.extractText()

            # Tokenize the content into sentences
            sentences = sent_tokenize(text)

            # Variable to indicate if the sentence with the word VISTOS was found
            found_vistos = False

            # Iterate through the sentences and tokenize the words
            for sentence in sentences:
                words = word_tokenize(sentence.lower(), language='portuguese')

                # Filter stop words and scores
                filtered_words = [word for word in words if
                                  word not in stop_words and word not in string.punctuation]

                # Put the filtered words back together into sentences
                filtered_sentence = ' '.join(filtered_words)

                # Check if the sentence contains the word "VISTOS"
                if 'vistos' in filtered_sentence:
                    found_vistos = True

                # Check if the sentence was found and extract the case number
                if found_vistos and 'processo' in filtered_sentence:
                    process_number = filtered_sentence.split()[-1]
                    print(process_number)
                    found_vistos = False

                # Print the remaining data after the sentence with the word "VISTOS" and the case number
                if found_vistos:
                    print(filtered_sentence)
