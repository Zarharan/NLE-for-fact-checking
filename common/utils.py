import requests
import openai


PROMPT_TEMPLATES = {
    "PubHealth": {
        "veracity" : {
            "basic": ("Context: {}\nClaim: {}\nclaim is {}\n")
        },
        "explanation": {
            "basic": ("Context: {}\nClaim: {}\nclaim is {}\nWhy? {}\n")
            }
    }
}

# ToDo: Remove tokens before making the code publicly available.
HF_TOKEN= "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"
OAI_API_KEY = "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"


class Summarization():
    '''
    The Summarization object is responsible for implementing different methods to summarize a text (e.g. main text of the news).
    '''
    def __init__(self):
        pass

    @staticmethod
    def gpt3(text_for_summary, max_tokens=200):
        ''' This function gets a text and summarizes it by generating at most max_tokens by using GPT-3.

        :param text_for_summary: The input text 
        :type text_for_summary: str
        :param max_tokens: The maximum number of tokens for generated summary
        :type max_tokens: str

        :returns: The generated summary
        :rtype: str
        '''

        openai.api_key= OAI_API_KEY
        text_for_summary+= " TL;DR: "
        response = openai.Completion.create(engine="text-davinci-002", prompt=text_for_summary
            , temperature=0.2,max_tokens=max_tokens, top_p=1, frequency_penalty=0, presence_penalty=0)

        return response.choices[0].text

    @staticmethod
    def bart_large_cnn(text_for_summary, max_tokens=200):
        ''' This function gets a text and summarizes it by generating at most max_tokens by using bart-large-cnn-samsum from HiggingFace.

        :param text_for_summary: The input text 
        :type text_for_summary: str
        :param max_tokens: The maximum number of tokens for generated summary
        :type max_tokens: str

        :returns: The generated summary
        :rtype: str
        '''

        API_URL = "https://api-inference.huggingface.co/models/philschmid/bart-large-cnn-samsum"
        headers = {"Authorization": "Bearer " + HF_TOKEN}

        payload= {
            "inputs": text_for_summary,
            "truncation":True,
            "max_length": max_tokens
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()[0]["summary_text"]


SUMMARIZATION_KEY_VAL= {"false": None, "gpt3": Summarization.gpt3, "bart":Summarization.bart_large_cnn}