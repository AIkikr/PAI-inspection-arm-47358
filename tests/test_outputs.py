"""Regression tests for publication integrity, not AI/physics performance."""
import copy
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

from inspection_core import Settings, save_json
from inspection_store import file_hash, read_json, digest
from inspection_outputs import (verify_record, semantic_label, make_checks,
                                create_source_archive, create_submission, contained,
                                verify_planner_origin, rejection_label, comparison_label,
                                result_card_lines, select_evidence)


def complete_evidence():
    evaluation=[]
    for split in ["development","holdout"]:
        for i in range(7):
            evaluation.append({"kind":"normal","split":split,"accepted":True,"success":True,
                               "end_to_end_success":True,"language_exact":True,"failure":""})
        for i in range(5):
            evaluation.append({"kind":"reject","split":split,"accepted":False,"success":False,
                               "reason_correct":True,"failure":""})
    demos=[{"expected":["A","B"],"task_success":True,"success":True,"failure":""},
           {"expected":None,"task_success":None,"success":True,"failure":""},
           {"expected":None,"expected_reason":"forbidden_target:C","task_success":True,
            "success":False,"accepted":False,"failure":""},
           {"expected":["A","B"],"task_success":True,"success":True,"failure":""}]
    comparisons=[{"failure":"","scenario":s,"mode":m,"trace_hash":s}
                 for s in ["normal","outage","delayed","reordered"] for m in ["off","receipt","age"]]
    return {"diagnostic":False,"stages":{
        "evaluation":{"payload":{"records":evaluation}},
        "comparison":{"payload":{"available":True,"records":comparisons}},
        "demonstrations":{"payload":{"records":demos}}}}


class ReadinessTests(unittest.TestCase):
    def check(self,evidence=None,video=None,**kwargs):
        return make_checks(evidence or complete_evidence(),Settings(account="test_id"),
                           "診断用の記入", "診断用の考察",
                           video or {"duration_s":60,"frames":1200,"sha256":"x"},
                           kwargs.get("pdf_ok",True),kwargs.get("source_ok",True),kwargs.get("hashes_ok",True))

    def test_every_required_check_gates_readiness(self):
        self.assertTrue(self.check()["ready_for_manual_review"])
        for kwargs in [{"pdf_ok":False},{"source_ok":False},{"hashes_ok":False}]:
            self.assertFalse(self.check(**kwargs)["ready_for_manual_review"])

    def test_190_seconds_is_not_ready(self):
        checks=self.check(video={"duration_s":190,"frames":3800,"sha256":"x"})
        self.assertFalse(checks["required"]["video_60_to_180_seconds"])
        self.assertFalse(checks["ready_for_manual_review"])

    def test_wrong_order_completion_is_not_instruction_achievement(self):
        evidence=complete_evidence()
        demo=evidence["stages"]["demonstrations"]["payload"]["records"][0]
        demo.update(task_success=False,semantic_match=False)
        self.assertIn("不一致",semantic_label(demo))
        self.assertNotEqual(semantic_label(demo),"指示達成")
        self.assertFalse(self.check(evidence=evidence)["ready_for_manual_review"])

    def test_refusal_has_its_own_semantic_criterion(self):
        demo={"expected":None,"expected_reason":"forbidden_target:C","task_success":True,
              "accepted":False,"success":False,"validation_reason":"forbidden_target:C"}
        self.assertIn("期待した理由",semantic_label(demo))
        evidence=complete_evidence()
        evidence["stages"]["demonstrations"]["payload"]["records"][2]["task_success"]=False
        self.assertFalse(self.check(evidence=evidence)["ready_for_manual_review"])

    def test_unregistered_instruction_not_claimed_correct(self):
        label=semantic_label({"expected":None,"success":True})
        self.assertIn("一致は未判定",label)

    def test_custom_registered_expectation_with_dropped_target_is_not_ready(self):
        evidence = complete_evidence()
        custom = evidence['stages']['demonstrations']['payload']['records'][1]
        custom.update(expected=['B','A'],planned_targets=['B'],completed_targets=['B'],
                      success=True, task_success=False)
        self.assertIn('指示達成は不一致', semantic_label(custom))
        self.assertFalse(self.check(evidence=evidence)['ready_for_manual_review'])

    def test_exceptional_or_diagnostic_trials_not_ready(self):
        evidence=complete_evidence()
        evidence["diagnostic"]=True
        self.assertFalse(self.check(evidence=evidence)["ready_for_manual_review"])
        evidence=complete_evidence()
        evidence["stages"]["evaluation"]["payload"]["records"][0]["failure"]="capture_exception"
        self.assertFalse(self.check(evidence=evidence)["ready_for_manual_review"])

    def test_missing_comparison_not_ready(self):
        evidence=complete_evidence()
        evidence["stages"]["comparison"]["payload"]={"available":False,"records":[]}
        self.assertFalse(self.check(evidence=evidence)["ready_for_manual_review"])

    def test_blocked_wrong_reason_is_not_described_as_executed(self):
        self.assertEqual(rejection_label({'accepted':False,'reason_correct':False}),
                         '実行抑止／理由不一致')
        self.assertEqual(rejection_label({'accepted':True,'reason_correct':False}),
                         '実行許可／拒否失敗')

    def test_missing_comparison_cannot_claim_completed_ablation_in_video(self):
        evidence = complete_evidence()
        evidence['stages']['comparison']['payload'] = {'available':False,'records':[], 'reason':'未実施'}
        lines = result_card_lines(evidence)
        self.assertIn('比較未実施', lines[2])
        self.assertNotIn('12/12', lines[2])
        self.assertNotIn('3方式に再生', '\n'.join(lines))
        comparison = complete_evidence()['stages']['comparison']['payload']
        self.assertIn('12/12条件を実行', comparison_label(comparison))
        self.assertIn('AI精度とは別評価', comparison_label(comparison))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.run=Path(self.tmp.name)
        self.ep=self.run/"episodes"/"attempt_2"
        self.ep.mkdir(parents=True)
        # Old output deliberately remains next to this new refusal result.
        (self.ep/"capture_A.png").write_bytes(b"stale old picture")

    def tearDown(self):
        self.tmp.cleanup()

    def record(self,**kwargs):
        result={"status":"rejected","success":False,"accepted":False,
                "captures":[],"artifact_hashes":{},"diagnostic":False,**kwargs}
        save_json(self.ep/"result.json",result)
        return dict(result,result_path="episodes/attempt_2/result.json",result_sha256=file_hash(self.ep/"result.json"))

    def test_old_capture_is_not_discovered(self):
        verified=verify_record(self.run,self.record())
        self.assertEqual(verified["captures"],[])
        self.assertNotIn("episodes/attempt_2/capture_A.png",verified["files"])

    def test_capture_must_be_in_manifest(self):
        with self.assertRaisesRegex(ValueError,"Capture missing"):
            verify_record(self.run,self.record(captures=["capture_A.png"]))

    def test_modified_result_or_claimed_success_is_rejected(self):
        record=self.record()
        record["success"]=True
        with self.assertRaisesRegex(ValueError,"disagreement"):
            verify_record(self.run,record)
        record=self.record()
        save_json(self.ep/"result.json",{"success":True})
        with self.assertRaisesRegex(ValueError,"changed"):
            verify_record(self.run,record)

    def test_failed_new_attempt_does_not_publish_old_success(self):
        verified=verify_record(self.run,self.record(status="failed",failure="segment_timeout"))
        self.assertFalse(verified["result"]["success"])
        self.assertEqual(verified["captures"],[])
        with self.assertRaisesRegex(ValueError,"cannot claim success"):
            verify_record(self.run,self.record(status="failed",success=True,failure="segment_timeout"))
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            verify_record(self.run,self.record(status="interrupted"))

    def test_modified_artifact_is_rejected(self):
        capture=self.ep/"capture_A.png"
        record=self.record(captures=[capture.name],artifact_hashes={capture.name:file_hash(capture)})
        capture.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError,"changed"):
            verify_record(self.run,record)

    def test_diagnostics_forbidden(self):
        with self.assertRaisesRegex(ValueError,"Diagnostic"):
            verify_record(self.run,self.record(diagnostic=True))

    def planner_record(self, planner, metadata):
        save_json(self.ep/'plan.json', {'planner':planner})
        record = self.record(artifact_hashes={'plan.json':file_hash(self.ep/'plan.json')}, **metadata)
        return record, read_json(self.ep/'result.json')

    def test_reference_plan_is_valid_only_in_comparison(self):
        from inspection_reference import make_control_reference, reference_metadata
        reference = make_control_reference()
        metadata = reference_metadata(reference)
        record, actual = self.planner_record(reference, metadata)
        payload = {'reference_plan':reference, **metadata}
        verify_planner_origin(self.run, 'comparison', record, actual, payload, {})
        for name in ['evaluation', 'demonstrations']:
            with self.assertRaisesRegex(ValueError, 'actual LLM'):
                verify_planner_origin(self.run, name, record, actual, payload, {})
        changed = dict(payload, reference_plan_hash='0'*64)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            verify_planner_origin(self.run, 'comparison', record, actual, changed, {})

    def test_llm_episode_must_match_verified_inference(self):
        planner = {'source':'llm', 'raw_output':'diagnostic fixture'}
        record, actual = self.planner_record(planner, {'planner_source':'llm'})
        record['id'] = 'n01'
        verify_planner_origin(self.run, 'evaluation', record, actual, {}, {'n01':planner})
        with self.assertRaisesRegex(ValueError, 'differs'):
            verify_planner_origin(self.run, 'evaluation', record, actual, {}, {'n01':dict(planner,raw_output='different')})

    def test_path_cannot_escape(self):
        for value in ["../outside.png","episodes\\attempt_2\\capture_A.png"]:
            with self.assertRaises(ValueError):
                contained(self.run,value)

    def test_source_archive_excludes_unselected_old_images(self):
        source=self.run/"source"
        source.mkdir()
        (source/"example.py").write_text("print('snapshot')",encoding="utf-8")
        notebook=self.run/"notebook.ipynb"
        notebook.write_text("{}",encoding="utf-8")
        evidence={"manifest":{"source_hashes":{"example.py":file_hash(source/"example.py")}},
                  "files":verify_record(self.run,self.record())["files"],"selected_inputs":{}}
        archive=self.run/"source.zip"
        create_source_archive(self.run,archive,Settings(),evidence,notebook)
        with zipfile.ZipFile(archive) as z:
            self.assertNotIn("results/episodes/attempt_2/capture_A.png",z.namelist())
            self.assertIn("results/episodes/attempt_2/result.json",z.namelist())
            self.assertEqual(z.read("example.py"),b"print('snapshot')")


class SelectedEvidenceOriginTests(unittest.TestCase):
    """Select the whole staged set, so provenance gates cannot be bypassed."""
    def setUp(self):
        from inspection_reference import make_control_reference, reference_metadata
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.contract = {'revision':'a'*40}
        self.planner = {'source':'llm','raw_output':'test-fixture-only'}
        self.plans = {'n01':self.planner}
        self.reference = make_control_reference()
        metadata = reference_metadata(self.reference)
        self.manifest = {'run_id':'fixture_run','diagnostic':False}
        for name in ['manifest.json','manifest.sha256','planner_contract.json','planner_contract.sha256',
                     'settings.json','environment.json','environment.sha256']:
            save_json(self.run/name, {})
        save_json(self.run/'plans'/'n01.json', self.planner)
        save_json(self.run/'plans_manifest.json', {'files':{'n01':file_hash(self.run/'plans'/'n01.json')}})
        self.stages={}
        for name in ['evaluation','comparison','demonstrations']:
            ep=self.run/name/'episode'
            ep.mkdir(parents=True)
            planner=self.reference if name=='comparison' else self.planner
            origin=metadata if name=='comparison' else {'planner_source':'llm'}
            save_json(ep/'plan.json', {'planner':planner})
            actual={'status':'complete','success':True,'diagnostic':False,'captures':[],
                    'artifact_hashes':{'plan.json':file_hash(ep/'plan.json')},**origin}
            save_json(ep/'result.json',actual)
            row=dict(actual,id='n01',source_plan_id='n01',
                     result_path=(ep/'result.json').relative_to(self.run).as_posix(),
                     result_sha256=file_hash(ep/'result.json'))
            payload={'records':[row]}
            if name=='comparison':
                payload.update(reference_plan=self.reference,**metadata)
            self.stages[name]={'status':'complete','run_id':'fixture_run','stage_id':name,
                               'contract_hash':digest(self.contract),'plan_set_hash':digest(self.plans),
                               'payload':payload}

    def select(self):
        for name,stage in self.stages.items():
            path=self.run/name/'stage.json'
            save_json(path,{k:v for k,v in stage.items() if not k.startswith('_')})
            stage.update(_path=path.relative_to(self.run).as_posix(),_sha256=file_hash(path))
        with patch('inspection_outputs.assert_run', return_value=self.manifest), \
             patch('run_project.load_contract', return_value=self.contract), \
             patch('run_project.load_plans', return_value=self.plans), \
             patch('inspection_outputs.load_stage', side_effect=lambda run,name:self.stages[name]):
            return select_evidence(self.run,self.run,Settings())

    def test_valid_control_reference_is_selected_but_mutation_is_rejected(self):
        self.assertFalse(self.select()['diagnostic'])
        self.stages['comparison']['payload']['reference_plan'] = dict(self.reference,raw_output='changed')
        with self.assertRaises(ValueError):
            self.select()

    def test_falsely_labelled_llm_cannot_hide_control_reference(self):
        ep=self.run/'evaluation'/'episode'
        save_json(ep/'plan.json',{'planner':self.reference})
        actual=read_json(ep/'result.json')
        actual['artifact_hashes']['plan.json']=file_hash(ep/'plan.json')
        save_json(ep/'result.json',actual)
        row=self.stages['evaluation']['payload']['records'][0]
        row.update(actual,result_sha256=file_hash(ep/'result.json'))
        with self.assertRaisesRegex(ValueError,'actual LLM'):
            self.select()


class PublicationRegressionTests(unittest.TestCase):
    def setUp(self):
        from test_store import copy_project
        from inspection_store import new_run
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = copy_project(Path(self.tmp.name) / 'project')
        self.settings = replace(Settings(),user_instruction='Aだけを点検してください。',stale_threshold=1.0)
        self.run = new_run(self.root,self.settings)

    def test_failed_publication_validation_invalidates_old_success(self):
        from inspection_store import start_stage, seal_stage, load_stage, read_json
        old = start_stage(self.run,'publication','old_contract','old_plans')
        seal_stage(self.run,'publication',old,{'diagnostic_fixture':True})
        with patch('inspection_outputs.select_evidence',side_effect=ValueError('deliberate evidence mismatch')):
            with self.assertRaisesRegex(ValueError,'deliberate evidence mismatch'):
                create_submission(self.root,self.run,self.settings,'test','test')
        with self.assertRaises(ValueError):
            load_stage(self.run,'publication')
        current = read_json(self.run/'current.json')['publication']
        self.assertEqual(current['status'],'failed')
        self.assertNotEqual(current['path'],old.relative_to(self.run).as_posix())

    def test_exported_notebook_preserves_settings_and_resolved_model(self):
        import json
        from notebook_builder import build_notebook
        notebook = build_notebook(self.run/'source',self.run/'diagnostic_export.ipynb',
                                  settings=self.settings,model_revision='a'*40)
        document = json.loads(notebook.read_text(encoding='utf-8'))
        config = next(''.join(c['source']) for c in document['cells']
                      if c['cell_type']=='code' and 'BASE_SETTINGS = json.loads(' in ''.join(c['source']))
        namespace = {}
        exec(compile(config,'generated_config','exec'),namespace)
        self.assertEqual(namespace['USER_INSTRUCTION'],self.settings.user_instruction)
        self.assertEqual(namespace['STALE_THRESHOLD'],1.0)
        self.assertEqual(namespace['MODEL_REVISION'],'a'*40)
        self.assertEqual(namespace['BASE_SETTINGS']['model_revision'],'a'*40)
        self.assertTrue(all(not c.get('outputs') for c in document['cells']))

    def test_publication_identity_can_change_without_rewriting_experiment(self):
        import json
        from types import SimpleNamespace
        evidence = complete_evidence()
        evidence.update(manifest=read_json(self.run/'manifest.json'),
                        contract={'revision':'a'*40}, contract_hash='b'*64,
                        plan_set_hash='c'*64, selected_inputs={},
                        files={name:file_hash(self.run/name) for name in ['manifest.json','settings.json']})
        before = {name:file_hash(self.run/name) for name in ['manifest.json','settings.json']}
        rendered_accounts=[]
        def video(run, path, settings, evidence):
            rendered_accounts.append(settings.account)
            path.write_bytes(b'publication-test-fixture')
            return {'duration_s':60,'frames':1200,'sha256':file_hash(path)}
        def report(run, path, settings, *args):
            rendered_accounts.append(settings.account)
            path.write_bytes(b'x'*2000)
            return path
        with patch('inspection_outputs.select_evidence', return_value=evidence), \
             patch('inspection_outputs.create_submission_video', side_effect=video), \
             patch('inspection_outputs.create_report', side_effect=report), \
             patch('pypdf.PdfReader', return_value=SimpleNamespace(pages=[None]*4)):
            result=create_submission(self.root,self.run,self.settings,'変更内容','考察',submission_account='student_123')
        self.assertEqual(rendered_accounts,['student_123','student_123'])
        self.assertEqual(self.settings.account,'YOUR_OMNICAMPUS_ID')
        self.assertEqual(before,{name:file_hash(self.run/name) for name in before})
        self.assertEqual(Path(result['bundle']).name,'PAI_submission_student_123.zip')
        identity=read_json(Path(result['bundle']).parent/'publication_inputs.json')
        self.assertEqual(identity['experiment_account'],'YOUR_OMNICAMPUS_ID')
        self.assertEqual(identity['publication_account'],'student_123')
        with zipfile.ZipFile(result['source_zip']) as archive:
            identity=json.loads(archive.read('results/publication_identity.json'))
            self.assertEqual(identity['publication_account'],'student_123')
            self.assertEqual(identity['experiment_account'],'YOUR_OMNICAMPUS_ID')
            self.assertEqual(json.loads(archive.read('results/settings.json'))['account'],'YOUR_OMNICAMPUS_ID')
            notebook=json.loads(archive.read('PAI_Inspection_Colab_v3.ipynb'))
            config=next(''.join(c['source']) for c in notebook['cells']
                        if c['cell_type']=='code' and 'BASE_SETTINGS = json.loads(' in ''.join(c['source']))
            namespace={}
            exec(compile(config,'exported_config','exec'),namespace)
            self.assertEqual(namespace['ACCOUNT'],'student_123')
            self.assertEqual(namespace['USER_INSTRUCTION'],self.settings.user_instruction)
            self.assertEqual(namespace['STALE_THRESHOLD'],1.0)

    def test_invalid_publication_identity_is_rejected(self):
        with patch('inspection_outputs.select_evidence', return_value={}):
            for invalid in ['../escape','bad/name','a@example.com',42]:
                with self.assertRaises(ValueError):
                    create_submission(self.root,self.run,self.settings,'変更','考察',submission_account=invalid)


if __name__=="__main__":
    unittest.main()
