from glob import glob
import json
import os
import sys
from nltk.probability import FreqDist
from pathlib import Path

aux_data_path = 'data/F1000-22/data'
types = set()
gw_list = []
gateway_dir = 'gateways'

neither = 0
both = 0
num_case = 0

gateway_lists = {}


# Point from link to gateway
url_dict = {}
for file in os.listdir(gateway_dir):
    gw_name = file[:file.index('_')]
    with open(os.path.join(gateway_dir, file)) as gw_links:
        for link in gw_links:
            url_dict[str(Path(link)).strip()] = gw_name

print(url_dict.keys())


# Gab all files named 'meta.json'
for filename in glob(aux_data_path + '/**/meta.json', recursive=True):
    # print(filename) 


    with open(filename) as meta_file:

        json_file = json.load(meta_file)
        is_case_report = False
        if json_file.get('atype') == 'case-report':
            is_case_report = True
        
        link = json_file.get('url')
        if link is not None:

            link = Path(link)
            link = str(link.parent)
            
            gateway = url_dict.get(link, None)
            gw_list.append(gateway)

            if is_case_report and (gateway is not None):
                # print("IS BOTH A CASE REPORT AND ", is_case_report, gateway)
                both += 1
                # sys.exit(0)

            elif not is_case_report and (gateway is None):
                # print("IS NEITHER CASE REPORT NOT ANY OF THE GATEWAYS", is_case_report, gateway)
                neither += 1
                # sys.exit(0)
            
            elif is_case_report:
                num_case += 1
            
            else:
                print(filename, is_case_report, gateway)
            
            # Do not include these
            if is_case_report and (gateway is not None):
                pass
        
            else:
                if is_case_report:
                    gateway = 'case'         

                current_list = gateway_lists.get(gateway, [])
                current_list.append(link[link.rindex('/')+1:])
                gateway_lists[gateway] = current_list

        else:
            print("SKIPPING", filename)


print("NUM NEITHER", neither)
print("NUM BOTH", both)
print("NUM CASE REPORTS", num_case)
# print(types)

fd = FreqDist(gw_list)
for key in fd.keys():
    print(key, ',', fd[key])

with open('gw_lookup.json', 'w') as outfile:
    json.dump(gateway_lists, outfile)