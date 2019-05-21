import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_lg")

def same_cluster(word):
    queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
    # by_cluster = filter(lambda x: x.cluster == word.cluster, queries)
    return queries

def most_similar(word):
    cluster = same_cluster(word)
    by_similarity = sorted(cluster, key=lambda w: word.similarity(w), reverse=True)
    # print([w for w in map(lambda x: [x.orth_,x.similarity(word)],by_similarity[:10])])
    return [w.orth_ for w in filter(lambda x: x.similarity(word) > 0.7,by_similarity[:10])]

handled_poss = ["NOUN", "ADJ", 'ADV']

def get_synonim(token):
    if token.pos_ in handled_poss:
        synonim = next((x for x in most_similar(token) if x != token.orth_), False)
        if synonim:
            return synonim
        return token.orth_
    return token.orth_

def get_paraphrase(sentence):
    doc = nlp(sentence)
    return ' '.join(map(get_synonim, doc))


while True:
    sent = input('Give me a sentence to paraphrase:   ')
    print(get_paraphrase(sent))