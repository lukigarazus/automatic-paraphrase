import spacy

test_cases = [
    "This tree looks beautiful!",
    "I am very tired.",
    "You should buy a new shirt.",
    "Nancy ate 20 hot dogs yesterday.",
]

handled_poss = ["NOUN", "ADJ", 'ADV']

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_lg")

def get_lemma_from_lexeme(lexeme):
    return nlp(lexeme.orth_)[0].lemma

def avoid_token_plurals_in_lexemes(lexemes, token):
    return filter( lambda x: get_lemma_from_lexeme(x) != token.lemma, lexemes )

def filter_lexemes_by_similarity_to_token(lexemes, token, treshold):
    return filter( lambda x: x.similarity(token) > treshold, lexemes )

def same_cluster(word):
    queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
    return queries

def most_similar(word):
    cluster = same_cluster(word)
    by_similarity = sorted(cluster, key=lambda w: word.similarity(w), reverse=True)
    return [w.orth_ for w in avoid_token_plurals_in_lexemes(filter_lexemes_by_similarity_to_token(by_similarity[:10], word, 0.5), word)]

def get_synonim(token):
    if token.pos_ in handled_poss:
        synonim = next((x for x in most_similar(token) if x != token.orth_), False)
        synonim_doc = nlp(synonim) # Avoid plurals
        synonim_token = synonim_doc[0]
        if synonim and synonim_token.lemma != token.lemma:
            return synonim
    return token.orth_ # If nothing found return the same token

def get_paraphrase(sentence):
    doc = nlp(sentence)
    para = ' '.join(map(get_synonim, doc))
    para_doc = nlp(para)
    # print("Paraphrase similarity by document:", para_doc.similarity(doc))
    return para


while True:
    test_results = []

    for test in test_cases:
        test_results.append([test, get_paraphrase(test)])

    print("Test cases")
    for result in test_results:
        print("Case: ", result[0])
        print("Result: ", result[1])
    
    sent = input('Give me a sentence to paraphrase:   ')
    print(get_paraphrase(sent))