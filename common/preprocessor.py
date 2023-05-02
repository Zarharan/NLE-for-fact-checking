import re
from bs4 import BeautifulSoup


class Preprocessor():
  '''
  The Preprocessor object is responsible for cleaning the text of instances.

  :ivar emoji_pattern: emoji patterns to omit from the text
  :vartype emoji_pattern: regex
  :ivar pattern_list: list of patterns to omit from the text
  :vartype pattern_list: list
  '''

  def __init__(self):

    self.emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", flags=re.UNICODE)

    self.pattern_list = ["\\u200c|\\u200d|\\u200e|\\u200b|\\u2067|\\u2069|\\u016b|\\u03b2|\\u2033"]


  def clean_text(self, text):
    '''this function gets a text and after removing HTML tags, URLs, and specific patterns returns clean text

    :param text: The target text
    :type text: str

    :returns: Cleaned text
    :rtype: str
    '''

    # remove html tags
    text= self.__remove_html_tags_url(text)

    for pattern in self.pattern_list:
      text = re.sub(pattern, " ", text)


    text = self.__remove_emoji(text)
    # remove hashtags
    text = self.__remove_hashtags(text)

    return text
  

  def __remove_emoji(self, text):
    '''this function gets a text and removes emoji icons from it

    :param text: The target text
    :type text: str

    :returns: Cleaned text
    :rtype: str
    '''    
    return self.emoji_pattern.sub(r'', text)    


  def __remove_html_tags_url(self, text):
    '''this function gets a text and removes html tags and Urls

    :param text: The target text
    :type text: str

    :returns: Cleaned text
    :rtype: str
    '''    
    
    text= BeautifulSoup(text, features="html.parser").get_text()
    text= re.sub(r"http\S+", "", text)

    return text


  def __remove_hashtags(self, text):
    '''this function gets a text and removes hashtags

    :param text: The target text
    :type text: str

    :returns: Cleaned text
    :rtype: str
    '''    
    clean_tokens = [word.strip("#").replace('_', ' ') if word.startswith("#") else word for word in text.split()]
    clean_tokens = ' '.join(clean_tokens)
    return clean_tokens