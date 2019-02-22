
from random import choice
import nltk
from nltk.grammar import Nonterminal
from nltk.parse.generate import generate


# def generate_sentence(grammar, items=[Nonterminal("S")]):
#   frags = []
#   if len(items) == 1:
#     if isinstance(items[0], Nonterminal):
#       for prod in grammar.productions(lhs=items[0]):
#         frags.append(generate_sentence(grammar, prod.rhs()))
#     else:
#       frags.append(items[0])
#   else:
#     # This is where we need to make our changes
#     chosen_expansion = choice(items)
#     frags.append(generate_sentence, chosen_expansion)
#   return frags


def load_grammar(filename):
  with open(filename) as fh:
    return nltk.CFG.fromstring(fh.read())
