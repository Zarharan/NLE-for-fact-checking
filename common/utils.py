import requests
import openai
from rouge_score import rouge_scorer
from openai.error import RateLimitError
import backoff


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

    :param max_tokens: The maximum number of tokens for generated summary
    :type max_tokens: str
    :param temperature: To set the randomness of generated text (between 0 and 1, with 0 being the most predictable and 1 being the most random)
    :type temperature: float
    :param model_name: The target model to generate summary
    :type model_name: string
    '''
    def __init__(self, max_tokens= 300, temperature= 0.5, model_name= "bart"):
        self.max_tokens= max_tokens
        self.temperature= temperature
        self.model_name= model_name

    
    def get_summary(self, text_for_summary):
        if self.model_name== "bart":
            return self.__bart_large_cnn(text_for_summary)
        elif self.model_name == "gpt3":
            return self.__gpt3(text_for_summary)


    @backoff.on_exception(backoff.expo, RateLimitError)
    def __gpt3(self, text_for_summary):
        ''' This function gets a text and summarizes it by generating at most max_tokens by using GPT-3.

        :param text_for_summary: The input text 
        :type text_for_summary: str

        :returns: The generated summary
        :rtype: str
        '''

        openai.api_key= OAI_API_KEY
        text_for_summary+= "\nTL;DR:\n"
        response = openai.Completion.create(engine="text-davinci-003", prompt=text_for_summary
            , temperature= self.temperature,max_tokens=self.max_tokens, top_p=1, frequency_penalty=0, presence_penalty=0)

        return response.choices[0].text


    def __bart_large_cnn(self, text_for_summary):
        ''' This function gets a text and summarizes it by generating at most max_tokens by using bart-large-cnn-samsum from HiggingFace.

        :param text_for_summary: The input text 
        :type text_for_summary: str

        :returns: The generated summary
        :rtype: str
        '''

        API_URL = "https://api-inference.huggingface.co/models/philschmid/bart-large-cnn-samsum"
        headers = {"Authorization": "Bearer " + HF_TOKEN}

        payload= {
            "inputs": text_for_summary,
            "truncation":True,
            "max_length": self.max_tokens,
            "temperature": self.temperature
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()[0]["summary_text"]