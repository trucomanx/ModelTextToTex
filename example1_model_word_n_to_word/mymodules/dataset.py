from datasets import load_dataset

def split_long_sentence(sentence, max_words_by_phrase=64):
    words = sentence.split(" ")
    if len(words) <= max_words_by_phrase:
        return [sentence]
    
    chunks = []
    for i in range(0, len(words), max_words_by_phrase):
        chunks.append(" ".join(words[i:i+max_words_by_phrase]))
    
    return chunks

def load_wiki_dataset(max_len=50000, max_words_by_phrase=64):
    dataset = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)
    text = []
    
    for data in dataset['train']:
        # Substitui a pontuação pelos delimitadores definidos
        data_text = data['text']
        data_text = data_text.replace(". ", " . |")
        data_text = data_text.replace(".\n", " .\n|")
        data_text = data_text.replace("; ", " ; |")
        data_text = data_text.replace(", ", " , |")
        data_text = data_text.replace(": ", " : |")
        data_text = data_text.replace("! ", " ! |")
        data_text = data_text.replace("? ", " ? |")
        
        # Divide o texto usando o delimitador "|"
        sentences = data_text.split("|")
        
        # Remove espaços em branco antes e depois de cada sentença e divide em pedaços menores
        for sentence in sentences:
            sentence = sentence.strip()  # Remove os espaços em branco antes e depois
            if sentence:  # Garante que a sentença não esteja vazia
                text.extend(split_long_sentence(sentence, max_words_by_phrase))
    
    return text[:max_len] if max_len < len(text) else text

