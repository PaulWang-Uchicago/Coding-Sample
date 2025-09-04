from mrjob.job import MRJob
from mrjob.step import MRStep
import re
import nltk
from nltk.corpus import stopwords

# Make sure stopwords are downloaded
nltk.download('stopwords', quiet=True)

# Use NLTK's English stopwords
STOPWORDS = set(stopwords.words('english'))

# simple word regex
WORD_RE = re.compile(r"\b[a-z']+\b")

class MRMostUsedWords(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper_tokenize,
                   combiner=self.combiner_count,
                   reducer=self.reducer_count),
            MRStep(mapper=self.mapper_prep_top10,
                   reducer=self.reducer_top10)
        ]

    def mapper_tokenize(self, _, line):
        """Emit (word, 1) for each non-stopword token."""
        for w in WORD_RE.findall(line.lower()):
            if w not in STOPWORDS:
                yield w, 1

    def combiner_count(self, word, counts):
        """Local aggregation to cut network I/O."""
        yield word, sum(counts)

    def reducer_count(self, word, counts):
        """Global word counts."""
        yield word, sum(counts)

    def mapper_prep_top10(self, word, count):
        """Send everything to a single reducer, keyed by None."""
        yield None, (count, word)

    def reducer_top10(self, _, count_word_pairs):
        """Sort all pairs descending and take the top 10."""
        top10 = sorted(count_word_pairs, reverse=True)[:10]
        for count, word in top10:
            yield word, count

if __name__ == '__main__':
    MRMostUsedWords.run()
