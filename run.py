import pandas as pd
import time
import random
from eval2 import ImprovedSTEMEvaluator

api_keys = ["AIzaSyA-Z5H4oq9oxACY6LlE_0sgL-5O6End8eA" , "AIzaSyD6dV0eY_tTTyIYKVUuRdi4FYSyzsOktU0" , "AIzaSyDsxp7rzoRiTbvwOodnCQhNnMl9dV5cND0"]

path = "datasets/sat-maths-rag-phi4-3B.csv"
data = pd.read_csv(path)

def process(row):
    question = row['question']
    options = row['options']

    ques = question + " options : \n" + options

    response = row["phi4:3.7B_response_rag"]
    ground_truth = row["answer"]
    domine = row["content_domain"]
    skill = row["skill"]
    subject = domine + " " + skill

    max_retries = 10  # Set a maximum number of retries to avoid infinite loops
    retry_count = 0
    
    while True:
        try:
            api_key = random.choice(api_keys)
            evaluator = ImprovedSTEMEvaluator(api_key=api_key)
            results = evaluator.comprehensive_evaluation(
                question=ques,
                explanation=response,
                reference=ground_truth,
                subject=subject,
                target_audience="students"
            )
            return results  # Return the results if successful
        except Exception as e:
            retry_count += 1
            print(f"Error occurred: {e}. Retry attempt {retry_count}")
            if retry_count >= max_retries:
                print(f"Maximum retries ({max_retries}) reached. Returning error.")
                return {"error": str(e)}  # Return error information after max retries
            time.sleep(30)  # Wait for 30 seconds before retrying


data['eval_res']=data.apply(process, axis=1)
data.to_csv(f"eval_res_phi4_rag.csv", index=False)
data.to_csv(f"results/eval_res_phi4_rag.csv", index=False)
