import os
import json
import openai
openai.api_key_path = "openai"



prompt ="Here are the definitions of different labels. Recap summarizes the manuscript: 'The paper proposes a new method for...' Strength points out the merits of the work: 'It is very well written and the contribution is significant.' Weakness points out a limitation: 'However, the data is not publicly available, making the work hard to reproduce' Todo suggests the ways a manuscript can be improved: 'Could the authors devise a standard procedure to obtain the data?' Other contains additional information, e.g. reviewer's thoughts, background knowledge and performative statements: 'Few examples from prior work: [1], [2], [3]', 'Once this is clarified, the paper can be accepted.' Structure is used to organize the reviewing report: 'Typos:' " \
        "  The data is labeled in json format like this example: {sentences: ['A very good attemp to present the Indian COVID-19 scenario by the authors.','I congratulate them on their work','However a few queries','The data analysis has been performed on 1161 patients.','To project it for such a large population has limited scope','In COVID, most of the patients recover in due course.','If possible, the SEIR model could habe been used for a better picture', labels:['Strength', 'Structure','Recap','Weakness','Other','Todo']}" \
        "  Please label the following sentences and bring them into json format: ['Reviewer response for version 1', 'As pointed out by this review, the evidence for hypokalemia in COVID-19 is weak, promulgated by a single unpublished article that was not corroborated by larger series studies. ', 'My chief concern about the review, is that it amplifies the myth. ', 'Furthermore, the frequent use of loop diuretics in ARDS, which cause hypokalemia, makes a causal connection between the development of hypokalemia and COVID-19 precarious.', 'At the very least, the title should be changed to minimize the connection between hypokalemia and COVID-19.', 'Paragraph 1 is somewhat misleading. ', 'Although COVID-19 does have an array of different sequela, acute respiratory distress syndrome (ARDS) seems to be the most common serve form of the disease. ', 'The first paragraph should provide an incidence of this and the other sequela. ', 'Reference to large studies from China, Europe and North American should be cited.', 'Paragraph 2 ', 'The statement “SARS-CoV-2 may affect the kidneys directly and indirectly by both renal and renin-angiotensin-aldosterone system (RAS) involvement,” has no factual basis. ', 'There are a few small studies that demonstrate that SARS-CoV-2 can infect the kidney but it is presently unclear if this is a key mechanism of AKI in COVID19.', 'Paragraph 3 Provide a reference to Mas receptor on type 2 alveolar epithelial cells as stated.', 'Paragraph 3 last sentence. ', 'This is evidence for SARS-COV, not SARS-COV2', 'Provide a reference to “hyperaldosteronism does not always cause hypokalaemia.” ', 'Although “Switch” activation has been a popular explanation for this, it has never been proven. ', 'Perhaps, the authors could cite clinical studies, demonstrating the development of severe hypokalemia in normokalemic PA patients following thiazide administration as evidence.', 'The limitations of TTKG should be articulated.']"

MODEL = "gpt-3.5-turbo"
response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": prompt},
    ],
    temperature=0,
)

print(type(response))
print(response)
content_string =  response['choices'][0]['message']['content']
content = json.loads(content_string)
print(content)
print(type(content))
labels = content['labels']
print(type(labels))
print(labels)

#content = "{"sentences": ["Reviewer response for version 1", "As pointed out by this review, the evidence for hypokalemia in COVID-19 is weak, promulgated by a single unpublished article that was not corroborated by larger series studies. ", "My chief concern about the review, is that it amplifies the myth. ", "Furthermore, the frequent use of loop diuretics in ARDS, which cause hypokalemia, makes a causal connection between the development of hypokalemia and COVID-19 precarious.", "At the very least, the title should be changed to minimize the connection between hypokalemia and COVID-19.", "Paragraph 1 is somewhat misleading. ", "Although COVID-19 does have an array of different sequela, acute respiratory distress syndrome (ARDS) seems to be the most common serve form of the disease. ", "The first paragraph should provide an incidence of this and the other sequela. ", "Reference to large studies from China, Europe and North American should be cited.", "Paragraph 2 ", "The statement “SARS-CoV-2 may affect the kidneys directly and indirectly by both renal and renin-angiotensin-aldosterone system (RAS) involvement,” has no factual basis. ", "There are a few small studies that demonstrate that SARS-CoV-2 can infect the kidney but it is presently unclear if this is a key mechanism of AKI in COVID19.", "Paragraph 3 Provide a reference to Mas receptor on type 2 alveolar epithelial cells as stated.", "Paragraph 3 last sentence. ", "This is evidence for SARS-COV, not SARS-COV2", "Provide a reference to “hyperaldosteronism does not always cause hypokalaemia.” ", "Although “Switch” activation has been a popular explanation for this, it has never been proven. ", "Perhaps, the authors could cite clinical studies, demonstrating the development of severe hypokalemia in normokalemic PA patients following thiazide administration as evidence.", "The limitations of TTKG should be articulated."], "labels": []}
#"