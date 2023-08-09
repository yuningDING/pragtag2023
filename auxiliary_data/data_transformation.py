import pandas as pd
import os
import json


LABEL_MAPPING = {'paper_summary': 'Recap', 'summary_of_strengths':'Strength', 'summary_of_weaknesses':'Weakness', 'comments,_suggestions_and_typos':'Todo'}


if __name__ == "__main__":
    directory = "ARR-22/data"
    output_data = []
    for sub_dir in os.listdir(directory):
        #print(sub_dir)
        if sub_dir.endswith('.txt') or sub_dir.endswith('.json'):
            continue
        file = "/".join([directory, sub_dir,"v1/reviews.json"])

        original_data = json.loads(open(file).read())

        for review in original_data:
            #print(review)
            review_item = {}
            review_item['id'] = review['rid']
            review_item['domain'] = 'arr'
            review_sentences = []
            review_labels = []
            for section in review['report']:
                current_label = LABEL_MAPPING.get(section)
                sentences_indexs = review['meta']['sentences'][section]
                for segment in sentences_indexs:
                    sentence = review['report'][section][segment[0]:segment[1]]
                    review_sentences.append(sentence)
                    review_labels.append(current_label)
            review_item['sentences'] = review_sentences
            review_item['labels'] = review_labels
            output_data.append(review_item)
    print('#Reviews:',len(output_data))

    json_data = json.dumps(output_data)
    with open("arr.json", "w") as outfile:
        outfile.write(json_data)


