
class MinervaLabels:
  def __init__(self, minerva_output:str, words:list):
    self.words = words
    self.words_joined = "".join(words)
    self.minerva_output = minerva_output
    self.sentences = self._create_sentences()
    self.aligned_sentences = self._aligned_sentences()
    self.check_alinged_sentences()

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

  def check_alinged_sentences(self):
    aligned_join = "".join(self.aligned_sentences)
    if not aligned_join == self.words_joined:
      raise ValueError(f"Aligned sentences {self.aligned_sentences} do not match words {self.words_joined}")

  def aligned_labels(self):
    """
    Align the extracted sentences with the given tokens to produce labels.
    Returns a list where 1 indicates the end of a sentence (sentence boundary).
    """
    # Initialize all labels to 0
    labels = [0] * len(self.words)
    
    index = 0
    for sentence in self.sentences:
      length = len(sentence)
      index += length
      # labels[index - 1] = 1
    
    return labels