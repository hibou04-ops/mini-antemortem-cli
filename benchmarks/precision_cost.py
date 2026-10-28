"""Known development cases and exact duplicate-work audit, no live calls.

The expected labels are declared before the classifier is run. A selected
pool is checked only when the installed API exposes that additive input.
"""
import argparse
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

parser=argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path)
args=parser.parse_args()
if args.source_root: sys.path.insert(0,str(args.source_root/"src"))

from omegaprompt.domain.dataset import Dataset, DatasetItem
from omegaprompt.domain.judge import Dimension, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from mini_antemortem_cli import traps

cases=[
    ("actual_input", "same", "yes", "same", "yes", "REAL"),
    ("same_pair", "question", "answer", "question", "answer", "REAL"),
    ("label_yes", "one", "yes", "two", "yes", "GHOST"),
    ("label_positive", "one", "positive", "two", "positive", "GHOST"),
    ("label_zero", "one", "0", "two", "0", "GHOST"),
    ("lexical_overlap", "alpha beta", "x", "alpha gamma", "z", "GHOST"),
    ("case_identifier", "KEY_A", "x", "key_a", "z", "GHOST"),
]
results=[]
for name,shot_input,shot_output,input_text,reference,expected in cases:
    findings=traps.analytical_preflight(target_provider="openai",target_model="a",judge_provider="anthropic",judge_model="b",
        train_dataset=Dataset(items=[DatasetItem(id="train",input="unrelated")]),
        test_dataset=Dataset(items=[DatasetItem(id="test",input=input_text,reference=reference)]),
        rubric=JudgeRubric(dimensions=[Dimension(name="q",description="coherent",weight=1)]),
        variants=PromptVariants(system_prompts=["Answer","Explain carefully"],few_shot_examples=[{"input":shot_input,"output":shot_output}]))
    actual=next(f.label for f in findings if f.trap_id=="few_shot_leakage_into_test")
    results.append({"case":name,"expected":expected,"actual":actual,"correct":actual==expected})

counter={"tokenizations":0,"union_allocations":0,"intersections":0}
class Tokens(set):
    def __or__(self,other):
        counter["union_allocations"]+=1
        return super().__or__(other)
    def __and__(self,other):
        counter["intersections"]+=1
        return super().__and__(other)
original=traps._tokenize_for_jaccard
def tokenize(text):
    counter["tokenizations"]+=1
    return Tokens(original(text))
with patch.object(traps,"_tokenize_for_jaccard",tokenize):
    maximum=traps._max_pairwise_jaccard(["same words"]*1000)
print(json.dumps({"kind":"known development cases, not blinded/general population evidence",
    "cases":results,"correct":sum(r["correct"] for r in results),"total":len(results),
    "duplicate_workload":{"prompts":1000,"exact_maximum":maximum,**counter}},indent=2))
