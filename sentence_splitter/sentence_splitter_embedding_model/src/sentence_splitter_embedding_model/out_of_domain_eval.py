from transformers import pipeline
import urllib3

BASE_URL = "https://raw.githubusercontent.com/RacheleSprugnoli/Sentence_Splitting_Manzoni/refs/heads/main/test-novels/"
http = urllib3.PoolManager()

def load_text(url):
    response = http.request('GET', url)
    return response.data.decode('utf-8')

def load_lines_of_text(url):
    response = load_text(url)
    return response.splitlines()

def test_out_of_domain_eval():
    trained_model_name = "bert-base-cased-sentence-splitter"
    model_checkpoint = "fax4ever/" + trained_model_name
    inference_pipeline = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple")

    cuore_test = load_text(BASE_URL + "Cuore-test.txt")
    cuore_lines = load_lines_of_text(BASE_URL + "Cuore-GOLD.txt")

    cuore_inference = inference_pipeline(cuore_test)
    print(cuore_lines)
    print(cuore_inference)

if __name__ == "__main__":
    test_out_of_domain_eval()
    