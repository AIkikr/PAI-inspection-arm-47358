"""Compare preserved observations without using them as generated predictions."""
import copy
import unittest
from dataclasses import replace
from inspection_baseline import BASELINE, baseline_comparison
from inspection_ai import planner_request
from inspection_core import Settings, evaluation_cases


class BaselineTests(unittest.TestCase):
    def test_actual_baseline_identity_and_counts(self):
        result = baseline_comparison({'records':copy.deepcopy(BASELINE['records'])})
        self.assertTrue(result['available'])
        self.assertEqual([r['before'] for r in result['metrics']], [10,10,10,9])
        self.assertEqual([r['total'] for r in result['metrics']], [14,14,10,10])
        self.assertEqual([r['delta'] for r in result['metrics']], [0,0,0,0])
        self.assertEqual(result['changed_cases'], [])

    def test_regression_changes_do_not_modify_old_results(self):
        rows=copy.deepcopy(BASELINE['records'])
        target=next(r for r in rows if r['id']=='n03')
        target.update(language_exact=True,end_to_end_success=True)
        result=baseline_comparison({'records':rows})
        self.assertEqual(result['metrics'][1]['delta'],1)
        self.assertFalse(next(r for r in BASELINE['records'] if r['id']=='n03')['end_to_end_success'])
        self.assertEqual(result['changed_cases'],[{'id':'n03','before_success':False,'after_success':True}])

    def test_changed_cases_or_subsets_cannot_be_compared(self):
        for kind in ['subset','text','expected','order']:
            rows=copy.deepcopy(BASELINE['records'])
            if kind=='subset': rows.pop()
            if kind=='text': rows[0]['instruction']='Bを点検'
            if kind=='expected': rows[0]['expected']=['B']
            if kind=='order': rows.reverse()
            with self.subTest(kind=kind):
                self.assertFalse(baseline_comparison({'records':rows})['available'])

    def test_user_expected_answer_never_enters_model_contract(self):
        first=Settings(user_expected_targets=['A'])
        second=replace(first,user_expected_targets=['B','A','B'])
        self.assertEqual(planner_request(first),planner_request(second))

    def test_all_fixed_evaluation_cases_match_preserved_case_definitions(self):
        for case,row in zip(evaluation_cases(),BASELINE['records']):
            self.assertEqual(case['id'],row['id'])
            self.assertEqual(case['text'],row['instruction'])
            self.assertEqual(case.get('expected'),row.get('expected'))
            self.assertEqual(case.get('expected_reason'),row.get('expected_reason'))

    def test_custom_expected_targets_preserve_repetitions_and_reject_bad_values(self):
        Settings(user_expected_targets=['A','B','A']).validate()
        for value in [[],['C'],['a'],['A']*5,'A,B']:
            with self.subTest(value=value),self.assertRaises(ValueError):
                Settings(user_expected_targets=value).validate()
