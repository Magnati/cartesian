import json
import logging
import operator
import pathlib
import re

from cartesian.algorithm import oneplus
from cartesian.cgp import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def group(s):
    trans = str.maketrans({"[": "", "]": "", "*": "", "+": "", "$": "", "^": ""})
    return f"[{s.translate(trans)}]"


def or_(a, b):
    if a == b:
        return a
    return f"{a}|{b}"


def add(a, b):
    return a.replace("$", "") + b.replace("^", "")


primitives = [
    Primitive("add", add, 2),
    Primitive("star", lambda s: f"{s}*".replace("**", "*").replace("+*", "*"), 1),
    Primitive("plus", lambda s: f"{s}+".replace("++", "+").replace("*+", "+"), 1),
    Primitive("group", group, 1),
    Primitive("or", or_, 2),
    # Primitive("begin", lambda s: f"^{s.replace('^','')}", 1),
    # Primitive("end", lambda s: f"{s.replace('$', '')}$", 1),
    Symbol(","),
] + list(Symbol(f"{i}") for i in range(10))

pset = PrimitiveSet.create(primitives)


# TODO: #columns must be the longest match

MyCartesian = Cartesian("MyCartesian", pset, n_rows=10, n_columns=32, n_out=1, n_back=4)


def compile_regex(program):
    stack = []
    out = program._out_idx[0]
    for _, node in reversed(list(program._iter_subgraph(program[out]))):
        if node.arity > 0 and node.arity <= len(stack):
            args = [stack.pop() for _ in range(node.arity)]
            stack.append(node.function(*args))
        elif node.arity == 0:
            stack.append(node.name)
        else:
            raise ValueError("Incorrect program.")
    return re.compile(stack[-1])


data = ["0", "00,", "001,", "002,", "001,001,", "002,001,", "001,00"]


def evaluate(program):
    regex = compile_regex(program)
    return 1 / (1 + sum(1 for x in data if re.match(regex, "".join(reversed(x)))))


res = oneplus(evaluate, cls=MyCartesian, maxiter=10, n_jobs=1)
print(res)
print(compile_regex(res.ind))


class RegexFuncWrapper:

    examples = []

    @classmethod
    def init_examples(cls):
        json_dir = pathlib.Path(__file__).parent / "small_data.json"
        with open(json_dir) as json_file:
            cls.examples = json.load(json_file)["examples"]

    @staticmethod
    def exact_matches(regex):
        for e in RegexFuncWrapper.examples:
            pass

    @staticmethod
    def evaluate(individual):
        # print("evaluate-test: len=", len(RegexFuncWrapper.examples))
        regex = compile_regex(individual)
        # print("regex='{}'".format(str(regex)))

        # Confusion Matrix (without 'true negative')
        positive = 0
        negative = 0
        true_positive = 0
        # true_negative = 0
        false_negative = 0
        false_positive = 0

        all_wanted = 0

        for e in RegexFuncWrapper.examples:
            example_string = e["string"]
            wanted = e["match"]
            all_wanted += len(wanted)

            last_wanted = 0
            for matched in regex.finditer(example_string):
                positive += 1
                for w in wanted[last_wanted:]:
                    if w["start"] == matched.start() and w["end"] == matched.end():
                        true_positive += 1
                        last_wanted = wanted.index(w)
                        break
                    else:
                        if matched.end() > w["end"]:
                            negative += 1
                            false_negative += 1
                            last_wanted += 1
                            continue
                else:
                    false_positive += 1

            negative += len(wanted[last_wanted:])
            false_negative += len(wanted[last_wanted:])

        try:
            assert all_wanted == true_positive + false_negative
            assert positive == true_positive + false_positive
        except AssertionError as e:
            logger.exception(e)

        # Edge case
        if true_positive == 0 and (false_positive > 0 or false_negative > 0):
            precision, recall, f_score = 0, 0, 0
        else:
            # Precision (PPV):
            precision = true_positive / (true_positive + false_positive)
            # Recall (TPR):
            recall = true_positive / (true_positive + false_negative)

            # F-Measure:
            f_score = 2 * (precision * recall) / (precision + recall)

        # TODO: partial matches

        # length (needed?)
        length = len(regex.pattern)

        # TODO: weighting

        return 1 / (1 + f_score)


RegexFuncWrapper.init_examples()

res = oneplus(RegexFuncWrapper.evaluate, cls=MyCartesian, mutation_method="active", maxiter=10000, n_jobs=1)
print(res)
print(compile_regex(res.ind))
