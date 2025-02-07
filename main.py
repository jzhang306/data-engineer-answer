import pyarrow.parquet as pq
import json as json
import requests
import time


# Download the test_data.parquet file from the link.
url = 'https://huggingface.co/api/datasets/AdaptLLM/finance-tasks/parquet/Headline/test/0.parquet'
response = requests.get(url)
with open('test_data.parquet', 'wb') as f:
    f.write(response.content)
    
# start the timer
start_time = time.time()

# Read the parquet file
table = pq.read_table('test_data.parquet')

# Convert the table to a pandas DataFrame
df = table.to_pandas()

# extract elements from each row separately.
def process_row(row):
    qid = row['id']
    questions = row['input']
    answers = row['options']
    gold_index = row['gold_index']

    selected_answer = answers[gold_index]

#give the json format of one row.
    return{
        "id": qid,
        "Question": questions,
        "Answer": selected_answer
    }
#extract elements from each row separately.
qa_pairs =  []
for index, row in df.iterrows():
    qa_pairs.append(process_row(row))

# write the extracted data to a json file.
with open('./formatted_data.json', 'w', encoding = 'utf-8') as f:
    json.dump(qa_pairs,f,ensure_ascii=False,indent=4)
# end the timer
end_time = time.time()
total_time = end_time - start_time
print(f"Total question-answer pairs extracted: {len(qa_pairs)}, and the total time taken: {total_time} seconds.")