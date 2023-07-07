# Algorithm 1 - dataset cleaning and pre-processing

# Imports
import os
import nltk
from nltk.tokenize import sent_tokenize

# Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Use double backslashes in file or directory path
path = "C:\\Dataset"

# Get a list of files in the directory
files = os.listdir(path)

# Print the file list
print("Files found in the directory:")
for file in files:
    print(file)

    # Open the file and read the contents
    with open(os.path.join(path, file), 'r', encoding='utf8') as c:
        content = c.read()

        # Tokenize the content in the sentences
        sentences = sent_tokenize(content)

        # Print sentences without stop words
        print("Sentences in the file " + file + ":")
        for sentence in sentences:

                # Put the filtered words together into a sentence again
                filtered_sentence = ' '.join(sentences)

                # Print the sentences
                print(filtered_sentence)
