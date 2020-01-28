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
    def confusion_matrix(regex):
        """Returns all partiall scoring values which are obtained by iterating through
        RegexFuncWrapper.examples.
        
        Args:
            regex ([Regular Expression object]): To be scored.
        
        Returns:
            f_Score: [description]
            precision: [description]
            recall: [description]
            true_positive_character_rate: [description]
            true_negative_character_rate: [description]
        """
        # Confusion Matrix (without 'true negative')
        positive = 0
        # negative = 0
        true_positive = 0
        # true_negative = 0
        false_negative = 0
        false_positive = 0

        true_positive_character_rate = 0
        true_negative_character_rate = 0

        all_wanted = 0
        length_all_wanted = 0
        length_all_matched_wanted = 0
        length_all_unwanted = 0
        length_all_unmatched_unwanted = 0

        for e in RegexFuncWrapper.examples:
            example_string = e["string"]
            wanted = e["match"]
            unwanted = e["unmatch"]
            all_wanted += len(wanted)
            length_all_wanted += sum(w["end"] - w["start"] for w in wanted)
            length_all_unwanted += sum(u["end"] - u["start"] for u in unwanted)
            all_unmatched_indexes = []
            last_match_end = 0

            last_wanted = 0
            for matched in regex.finditer(example_string):
                positive += 1
                all_unmatched_indexes.append((last_match_end, matched.start()))
                last_match_end = matched.end()
                for w in wanted[last_wanted:]:

                    if matched.end() > w["start"]:  # not before
                        if matched.start() < w["end"]:
                            # partial match
                            length_all_matched_wanted += min(matched.end(), w["end"]) - max(
                                matched.start(), w["start"]
                            )
                            if w["start"] == matched.start() and w["end"] == matched.end():
                                true_positive += 1
                                last_wanted = wanted.index(w) + 1
                                break
                            else:
                                # partial but not exact
                                pass
                                # false_negative += 1

                        else:
                            # is behind
                            last_wanted = wanted.index(w) + 1
                            false_negative += 1

                else:
                    false_positive += 1
            else:
                if last_match_end < len(example_string):
                    all_unmatched_indexes.append((last_match_end, len(example_string)))
                elif last_match_end > len(example_string):
                    raise IndexError("last_match_end has illogical value, higher, than len(example).")

            if not all_unmatched_indexes:  # == no match for regex in example_string
                all_unmatched_indexes.append((last_match_end, len(example_string)))
                # false_negative += len(wanted[last_wanted:])

            # negative += len(wanted[last_wanted:])
            false_negative += len(wanted[last_wanted:])

            last_unwanted = 0
            for unmatched in all_unmatched_indexes:
                for u in unwanted[last_unwanted:]:
                    if unmatched[1] > u["start"]:  # not before
                        if unmatched[0] < u["end"]:
                            length_all_unmatched_unwanted += min(unmatched[1], u["end"]) - max(
                                unmatched[0], u["start"]
                            )
                        else:
                            last_unwanted += 1

        true_positive_character_rate = length_all_matched_wanted / length_all_wanted
        true_negative_character_rate = length_all_unmatched_unwanted / length_all_unwanted

        try:
            assert all_wanted == true_positive + false_negative
            assert positive == true_positive + false_positive
        except AssertionError as e:
            logger.exception(e)
            logger.error(f"all_wanted={all_wanted}")
            logger.error(f"true_positive={true_positive}")
            logger.error(f"false_negative={false_negative}")
            logger.error(f"false_positive={false_positive}")
            logger.error(f"positive={positive}")
            logger.error(f"regex={regex}")

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

        return f_score, precision, recall, true_positive_character_rate, true_negative_character_rate

    @staticmethod
    def evaluate(individual):
        # print("evaluate-test: len=", len(RegexFuncWrapper.examples))
        regex = compile_regex(individual)

        (
            f_score,
            precision,
            recall,
            true_positive_character_rate,
            true_negative_character_rate,
        ) = RegexFuncWrapper.confusion_matrix(regex)

        # character oriented accuracy
        accuracy = (true_positive_character_rate + true_negative_character_rate) / 2.0

        # length (needed?)
        length = len(regex.pattern)

        # TODO: weighting
        f_score_weight = 4
        precision_weight = 4
        accuracy_weight = 1
        tpcr_weight = 1
        tncr_weight = 1

        score = 1 / (0.1 + f_score_weight * f_score + accuracy_weight * accuracy)
        # score = 1 / (
        #     0.1
        #     + precision_weight * precision
        #     + tpcr_weight * true_positive_character_rate
        #     + tncr_weight * true_negative_character_rate
        # )

        print(
            "regex='{}' : {} : fm={}*{} : acc={}*{} : tpcr={}*{} : tncr={}*{}".format(
                str(regex),
                str(score),
                str(f_score_weight),
                str(f_score),
                str(accuracy_weight),
                str(accuracy),
                str(tpcr_weight),
                str(true_positive_character_rate),
                str(tncr_weight),
                str(true_negative_character_rate),
            )
        )
        return score


RegexFuncWrapper.init_examples()

res = oneplus(RegexFuncWrapper.evaluate, cls=MyCartesian, mutation_method="active", maxiter=1000, n_jobs=1)
print(res)
print(compile_regex(res.ind))
