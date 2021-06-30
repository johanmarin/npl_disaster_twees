import re

sp_characters = [r"\x89Û_", r"\x89ÛÒ", r"\x89ÛÓ", r"\x89ÛÏ", r"\x89ÛÏ", r"\x89Ûª",
                 r"\x89Û÷", r"\x89Ûª", r"\x89Û\x9d", r"å_", r"\x89Û¢", r"\x89Û¢åÊ",
                 r"åÊ", r"åÊ", r"åÈ", r"Ì_", r"Ì©", r"å¨", r"SuruÌ¤", r"åÇ", r"å£",
                 r"åÀ", r"Ì¼", r"&gt;", r"&lt;", r"&amp;", r"<3"
                 ]



hastag = re.compile(r'#[a-zA-Z0-9À-ÿ\u00f1\u00d1]+_*[a-zA-Z0-9À-ÿ\u00f1\u00d1]+_*[a-zA-Z0-9À-ÿ\u00f1\u00d1]*_*[a-zA-Z0-9À-ÿ\u00f1\u00d1]*')
tag    = re.compile(r'@[a-zA-Z0-9À-ÿ\u00f1\u00d1]+_*[a-zA-Z0-9À-ÿ\u00f1\u00d1]+_*[a-zA-Z0-9À-ÿ\u00f1\u00d1]*_*[a-zA-Z0-9À-ÿ\u00f1\u00d1]*')
url    = re.compile(r"https?:\/\/t.co\/[A-Za-z0-9]+")
emoji  = re.compile(r'['  u'\U0001F600-\U0001F64F'  # emoticons
                          u'\U0001F300-\U0001F5FF'  # symbols & pictographs
                          u'\U0001F680-\U0001F6FF'  # transport & map symbols
                          u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                          u'\U00002702-\U000027B0'
                          u'\U000024C2-\U0001F251'
                          ']+', flags=re.UNICODE)


web_error = re.compile('|'.join(sp_characters))