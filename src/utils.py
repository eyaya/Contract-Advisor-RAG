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

def create_QA(file_path):
    import docx2txt
    import re
    import pandas as pd

    # Read the text from the .docx file
    text = docx2txt.process("your_document.docx")

    # Define regex patterns for identifying questions and answers
    question_pattern = re.compile(r"Q\d+: (.+?)\n")
    answer_pattern = re.compile(r"A\d+: (.+?)\n")

    # Find all matches of questions and answers
    questions = question_pattern.findall(text)
    answers = answer_pattern.findall(text)

    # Create a dictionary to store the data
    data = {'question': questions, 'ground_truth': answers}

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data)

    return df
def remove_special_characters(input_string):
    # Define a regex pattern to match the special characters
    pattern = r'["\t●\n\[\]]'
    # Use re.sub() to replace matches of the pattern with an empty string
    cleaned_string = re.sub(pattern, ' ', input_string)
    return cleaned_string
