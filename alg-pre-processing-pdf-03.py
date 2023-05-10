# Algorithm 3 - dataset cleaning and pre-processing texts in PDF

# Imports
import os
import PyPDF2

# Use double backslashes in file or directory path
path = "C:\\Dataset2"

# Get a list of files in the directory
files = os.listdir(path)

# loop through all PDF files in the directory
for filename in os.listdir(path):
    print(filename)

    if filename.endswith('.pdf'):
        filepath = os.path.join(path, filename)
        with open(filepath, 'rb') as pdf_file:
            # create a PDF reader object
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)

            # get the number of pages in the PDF file
            num_pages = pdf_reader.getNumPages()

            # loop through all pages in the PDF file
            for page_num in range(num_pages):
                # get the text content of the page
                page = pdf_reader.getPage(page_num)
                text = page.extractText()

                # do something with the extracted text
                print(text)



