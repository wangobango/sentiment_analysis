from .senticnet.senticnet import SenticNet

"""Class which uses Senticnet API dictionary. Full documentation here: https://sentic.net/api/"""


class Dictionary:

    def __init__(self):
        self.sn = SenticNet()
    """
        Input : String 
        Output : "positive" or "negative"
    """

    def get_word_polarity(self, word, binary=False):
        return self.sn.polarity_value(word)

    def get_word_polarity_numerical_value(self, word):
        return self.sn.polarity_intense(word)


def main():
    sn = Dictionary()

    print(sn.get_word_polarity('Amazing'))


if __name__ == "__main__":
    main()
