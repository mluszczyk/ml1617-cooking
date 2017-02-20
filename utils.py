def preprocess_to_chars(ingredients):
    return preprocess_chars(ingredients, ord)


def preprocess_chars(ingredients, func):
    return list(filter(lambda x: x is not None, map(func, (';'.join(ingredients)))))


def preprocess_to_hashes(ingredients):
    return preprocess_words(ingredients, lambda x: hash(x) % 8000)


def preprocess_words(ingredients, func):
    return list(filter(lambda x: x is not None, map(func, (' ; '.join(ingredients)).split())))


class Encoder:
    def __init__(self):
        self.word_mapping = {}

    def transform(self, x):
        num = self.word_mapping.get(x)
        if num is not None:
            return num
        else:
            num = len(self.word_mapping)
            self.word_mapping[x] = num
            return num


def augment_permutations(ingredient_list, aug_max, shuffle):
    s = set()
    
    for _ in range(aug_max):
        ingredient_list = ingredient_list[:]
        shuffle(ingredient_list)
        s.add(tuple(ingredient_list))
    try:
        s.remove(tuple(ingredient_list))
    except KeyError:
        pass
    return list(s)
