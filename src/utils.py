import re
import pandas as pd


import textwrap

# to display the text in certain width
def word_wrap(text, width=70):

    return textwrap.fill(text, width=width)



# Read the text from the .docx file
#text = docx2txt.process("your_document.docx")

# Function to extract context based on page numbers
def extract_context(doc_text, section_pages):
    context_dict = {}
    for section, page in section_pages.items():
        print(page,type(page))
        start_index = doc_text.find("Page " + str(page))
        end_index = doc_text.find("Page " + str(int(page) + 1)) if str(int(page) + 1) in section_pages.values() else None
        context = doc_text[start_index:end_index].strip() if end_index else doc_text[start_index:].strip()
        context_dict[section] = context
    return context_dict
'''
def create_QA(file_path):
    # Read the text from the .docx file
    text = docx2txt.process(file_path)
    # Define regex patterns for identifying questions, answers and sections
    question_pattern = re.compile(r"Q\d+: (.+?)\n")
    answer_pattern = re.compile(r"A\d+: (.+?)\n")
    

    # Find all matches of questions, answers and sections
    questions = question_pattern.findall(text)
    answers = answer_pattern.findall(text)
    #sections = section_pattern.findall(text)

    # Create a dictionary to store the data
    data = {'Question': questions, 'Answer': answers}

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data)
    

    # Read the text from the second .docx file to extract context information
    # Read the text from the second .docx file to extract context information
    #toc_text = docx2txt.process(context_path)

    # Extract section numbers and corresponding page numbers from the Table of Contents
    #section_pages = dict(re.findall(r"Section ([\d\.]+).*?(\d+)", toc_text))

    # Extract context based on page numbers
    #context_text = docx2txt.process(context_path)
    #context_dict = extract_context(context_text, section_pages)

    #print(context_dict)
    # Merge context data with the question-answer DataFrame
    #df['Context'] = df['Section'].map(context_dict)




    return df
'''

def remove_special_characters(input_string):
    # Define a regex pattern to match the special characters
    pattern = r'["\t●\n\[\]]'
    # Use re.sub() to replace matches of the pattern with an empty string
    cleaned_string = re.sub(pattern, ' ', input_string)
    return cleaned_string
