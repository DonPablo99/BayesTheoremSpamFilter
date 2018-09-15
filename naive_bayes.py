# Define the classifier
class NaiveBayesClassifier():
    """Simple NaiveBayes model for spam detection"""
    def __init__(self, laplace=1):
        self.spam_words = {}
        self.spam_total = 0
        self.ham_words = {}
        self.ham_total = 0
        self.pA = 0
        self.pNotA = 0
        self.laplace = laplace  # Laplace Smoothing

    def __repr__(self):
        return ("NaiveBayesClassifier(laplace=" + str(self.laplace) + ")")

    # Read and count words from a specific email
    def processEmail(self, body, label):
        bow = self.cleanEmail(body)
        for word in bow:
            if label == 1:
                self.spam_words[word] = self.spam_words.get(word, 0) + 1
                self.spam_total += 1
            else:
                self.ham_words[word] = self.ham_words.get(word, 0) + 1
                self.ham_total += 1

    def train(self, train):
        total = 0
        num_spams = 0
        for i in train.index.values:
            email = train.loc[i]
            if email.spam == 1:
                num_spams += 1
            total += 1
            self.processEmail(email.body, email.spam)
        self.pA = num_spams / total  # Prior probability
        self.pNotA = (total - num_spams) / total  # 1 - pA

    # Clean the email body and return Bag Of Words
    def cleanEmail(self, body):
        clean_bow = []
        for w in body.split(" "):
            word = ""
            for l in w:
                if l.isalpha():
                    word += l
            if word != '':
                clean_bow.append(word.lower())
        return clean_bow

    # Gives the conditional probability p(B|A)
    def conditionalEmail(self, body, spam):
        result = 1
        bow = self.cleanEmail(body)
        for word in bow:
            result *= self.conditionalWord(word, spam, len(bow))
        return result

    # Laplace Smoothing for the words not present in the training set
    # gives the conditional probability p(B|A) with smoothing
    def conditionalWord(self, word, spam, num_words):
        if spam:
            return (self.spam_words.get(word, 0) + self.laplace) / (self.spam_total + self.laplace * num_words)
        return (self.ham_words.get(word, 0) + self.laplace) / (self.ham_total + self.laplace * num_words)

    # Classifies a new email
    def classify(self, email):
        isSpam = self.pA * self.conditionalEmail(email, True)  # P (A | B)
        notSpam = self.pNotA * self.conditionalEmail(email, False)  # P(¬A | B)
        return isSpam > notSpam