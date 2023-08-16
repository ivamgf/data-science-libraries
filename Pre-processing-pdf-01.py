# Algorithm 01 pre-processing PDF files

# Imports
import os
import nltk
import PyPDF2
from nltk.tokenize import sent_tokenize, word_tokenize

# Downloads
nltk.download('punkt')

# Use double backslashes in file or directory path
path = "C:\\Dataset"

# Get a list of files in the directory
files = os.listdir(path)

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

                # Put the filtered words back together into sentences
                filtered_sentence = ' '.join(words)

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
