import pandas as pd
from eval import eval_across_domains
from load import load_prediction_and_gold
import os
import json
import openai
openai.api_key_path = "openai"



# read in json file
validation = pd.read_json("../data/public_dat/validation.json")
print(validation)
validation_dummy = validation[['id']].copy()
prompt_start = "Here are the definitions of different labels. Recap summarizes the manuscript: 'The paper proposes a new method for...' Strength points out the merits of the work: 'It is very well written and the contribution is significant.' Weakness points out a limitation: 'However, the data is not publicly available, making the work hard to reproduce' Todo suggests the ways a manuscript can be improved: 'Could the authors devise a standard procedure to obtain the data?' Other contains additional information, e.g. reviewer's thoughts, background knowledge and performative statements: 'Few examples from prior work: [1], [2], [3]', 'Once this is clarified, the paper can be accepted.' Structure is used to organize the reviewing report: 'Typos:' " \
               "  The data is labeled in json format like this example: " \
               "{sentences: [{sentence : 'A very good attemp to present the Indian COVID-19 scenario by the authors.', label: 'Strength'}," \
               "{sentence : 'I congratulate them on their work', label: 'Strength'}," \
               "{sentence : 'However a few queries', label: 'Structure'}," \
               "{sentence : 'The data analysis has been performed on 1161 patients.', label: 'Recap'}," \
               "{sentence : 'To project it for such a large population has limited scope', label: 'Weakness'}," \
               "{sentence : 'In COVID, most of the patients recover in due course.', label: 'Other'}," \
               "{sentence : 'If possible, the SEIR model could habe been used for a better picture', label: 'Todo'}]}" \
               "  Please label the following sentences and bring them into json format: "
MODEL = "gpt-3.5-turbo"


final_pred = {}
# put every entry after a prompt
output_dir = 'gpt_output_validation'
os.makedirs(output_dir, exist_ok=True)
for index, row in validation.iterrows():
    print(index)
    #if (index<9):
    #    continue
    prediction = []
    #print(row)
    sentences = row['sentences']
    prompt = prompt_start + str(sentences)
    #print(prompt)
    essay_id = row['id']
#    print(sentences)
#    print(len(sentences))
    # send it to the openai API
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0,
    )
#    print(type(response))
#    print(response)
    content_string = response['choices'][0]['message']['content']
#    print(content_string)
    with open(output_dir + '/' + str(index) + "_out.txt", "w") as text_file:
        text_file.write(content_string)





