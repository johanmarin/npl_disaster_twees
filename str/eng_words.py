letter_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i",
               "j", "k","l", "m", "n", "o", "p", "q", "r", 
               "s", "t", "u", "v", "w", "x", "y", "z"
               ]

article_list = ["a", "an", "the"]

pronoun_list = ["i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "us", "them", 
                "my", "your", "his", "its", "our", "their",
                "this", "that", "these", "those",
                "any", "another", "anyone", "anything",
                "some", "something", "someone", 
                "every", "everyone", "everything", "nothing"
                ]

modal_list = ["can", "could", "may", "might", "will", "would", 
              "must", "shall", "should", "ought", "cannot"
              ]

commom_verbs = ["want", "hope", "like", "get", "got", "gotten",
                "gonna", "gotta", "going", "give", "gave", "given",
                "go", "goes", "went", "gonne", "gotta", "gets", "just"
                ]

auxiliar_list = ["be", "am", "is", "are", "been", "was", "were",
                 "being","do", "does", "did", "doing", "done",
                 "have",  "has", "had", "having"
                 ] 

adverb_list = [# time adv
               "today", "yesterday", "tomorrow", "late", "early", "now", "next",
               # frecuenccy adv
               "never", "always", "often", "sometimes", "frequently","seldom", 
               "rarely", "daily", "weekly", "montly", "yearly","again", "ever",
               # location adv
               "here", "there", "everywhere", "nowhere", "somewhere", "inside", "up",
               "outside", "upstairs", "downstairs", "out", 
               # so adv
               "well", "badly", "quickly", "slowly", "softly", "loudly",
               #comp adv
               "more", "least", "even", "much", "half", "behind", "instead", "less", 
               # realtion adv
               "near", "back", "front", "too", "also",  "as", "than",
               #other
               "no", "up", "not", "yes", "about", "very", "really", "actually", 
               "almost", "already", "totally", "maybe", "only"
               ]   

preposition_list = ["after", "since", "before", "till", "until", "end",
                    "in", "on", "over", "into", "under", "across",
                    "through", "outside", "off ", "against", "around",
                    "for", "from", "of", "by", "to", "at","with"
                    ]

adjetive_list = ["latest", "last", "many", "middle", "best", "all",
                 "due", "possible", "most", "own", "whole "
                 ]

conjuntion_list = ["but", "and", "or", "so", "yet", "however", "both",
                   "then", "therefore", "if", "while", "because"
                   ]

whq_list = ["what", "why", "when","where", "who", "which", "how"]

contraction_list = ["im"]    

contraction_list = [r'\s',r'(n\'t)',r'\'m',r'(\'ll)',r'(\'ve)',r'(\'s)',r'(\'re)',r'(\'d)']

punctuation = ['!', "'", '#', '$', '%', '&', '\\', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

stopwords = letter_list + article_list + pronoun_list + commom_verbs + auxiliar_list + adverb_list + adjetive_list + preposition_list + conjuntion_list + whq_list +modal_list


