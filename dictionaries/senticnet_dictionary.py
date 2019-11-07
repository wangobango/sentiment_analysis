from senticnet.senticnet import SenticNet

"""Class which uses Senticnet API dictionary. Full documentation here: https://sentic.net/api/"""


class Dictionary:

    def __init__(self):
        self.sn = SenticNet()

    """
        Input : String 
        Output : "positive" or "negative"
    """

    def get_word_polarity(self, word):
        value = "empty"
        try:
            value = self.sn.polarity_value(word.lower())
        except:
            print('An error occurred. Word: ' + word + 'is not known.')

        return value

    """
        Input : String
        Output : Int [-1 : 1]    
    """

    def get_word_polarity_numerical_value(self, word):
        value = "empty"
        try:
            value = self.sn.polarity_intense(word.lower())
        except:
            print('An error occurred. Word: ' + word + 'is not known.')
        return value

