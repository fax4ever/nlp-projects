
class MinervaLabels:
  def __init__(self, minerva_output:str, words:list):
    self.words = words
    self.words_joined = "".join(words)
    self.minerva_output = minerva_output
    self.sentences = self._create_sentences()
    self.aligned_sentences = self._aligned_sentences()
    self._check_alinged_sentences() # Sanity check

  def _create_sentences(self):
    import re
    sentences = []
    for line in self.minerva_output.split("\n"):
      # Look for lines that start with number followed by dot and space
      match = re.match(r'^(\d+)\.\s+(.*)', line)
      if match:
        sentence = match.group(2)  # Extract the sentence part after "NUMBER. "
        # Clean up Minerva control tokens like <|eot_id|>
        sentence = re.sub(r'<\|[^|]+\|>', '', sentence)
        sentences.append(sentence)
    return sentences

  def _aligned_sentences(self):
    return [self._align_sentence(sentence) for sentence in self.sentences]

  def _align_sentence(self, sentence):
    sentence = sentence.replace(" ", "")
    sentence = sentence.replace("...", "…")
    if sentence in self.words_joined:
      return sentence
    else:
      raise ValueError(f"Sentence {sentence} not found in {self.words_joined}")

  def _check_alinged_sentences(self):
    aligned_join = "".join(self.aligned_sentences)
    if not aligned_join == self.words_joined:
      raise ValueError(f"Aligned sentences {self.aligned_sentences} do not match words {self.words_joined}")

  def aligned_labels(self, golden_labels):
    labels = [0] * len(self.words)

    index = 0
    split_indexes = set()
    for sentence in self.aligned_sentences:
      length = len(sentence)
      index += length
      split_indexes.add(index)
    
    index = 0
    for i, word in enumerate(self.words):
      length = len(word)
      index += length
      if index in split_indexes:
        labels[i] = 1

    # The last word can be a 1 or a 0,
    # the model is not asked to predict the last word, so we use the golden labels
    labels[-1] = golden_labels[-1]
    
    return labels