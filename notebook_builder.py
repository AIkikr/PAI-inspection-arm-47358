"""Build a portable, output-free Colab notebook from a source snapshot."""
from __future__ import annotations
from inspection_provenance import register_module
register_module(__name__, __file__)
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

def build_notebook(root, output, settings=None, model_revision=''):
    from inspection_core import Settings
    root, output = Path(root), Path(output)
    config = asdict(settings or Settings())
    if model_revision: config['model_revision'] = model_revision
    cells=[]
    def add(kind,text):
        cell={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True),
              'id':hashlib.sha256((str(len(cells))+text).encode()).hexdigest()[:12]}
        if kind=='code': cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    def md(text): add('markdown',text)
    def code(text): add('code',text)
    md('''# Physical AI 応用1 最終課題：自然言語と情報鮮度で動く設備点検アーム v3

**このipynbだけを新しいGoogle Colab GPUランタイムへアップロードして、上から実行します。**
Qwenによる指示解釈、MuJoCoの4自由度アーム、通信条件の比較、動画・PDF・GitHub用ZIPの生成まで含みます。外部リポジトリ、APIキー、ROS、Dockerは不要です。

AIは対象と順序の解釈を担当し、観測の採用・保留・再開は独立した規則で判断します。点検は位置到達・静止・画像保存まで。故障診断や実機安全保証は含みません。

**v3のQwen実推論とColab全セル完走は、提供時点では未確認です。** 旧版の実行結果は変更前の基準として保持し、新しい結果と混ぜません。v3の成績は、このノートブックで実測してから記載します。

まずランタイム→ランタイムのタイプを変更→T4等のGPUを選択してください。制御テストを確認後に推論へ進みます。比較実験は工夫を示すための本実装の追加要素で、講座の必須要件であるという意味ではありません。

**所要時間の目安は初回30～90分、作業枠は1～2時間です。Colabでの実測値ではありません。** モデル取得・GPU割当・CPUによるOSMesa描画・Drive保存で変わります。セル6で手元の描画速度を測定します。

**記入する場所は2か所です。** 最初に「1. 記入① アカウント名と指示」を記入し、1～10を実行します。結果を確認してから「11. 記入② 本人の変更内容・考察・提出物生成」を記入・実行し、12でZIPを保存します。最初からすべてのセルを一括実行せず、10までの結果を見て考察してください。

**v3は旧版とは別の作業先・Drive保存先を使います。** 初回も再開も、通常はセル5のAUTOを使ってください。旧版の結果は上書きせず、v3の新しいrunを作ります。指示や閾値などの実験条件を変更する場合はNEWを選びます。

**Google Driveへ途中経過を自動保存します。** 再接続後は同じノートブックを上から実行し、AUTOで最後に検証済みの保存地点から再開します。計算はColabの一時領域で行い、推論1件・エピソード1件の完了ごとにDriveへ保存します。Google Driveへの接続許可が必要です。
''')
    md('''## 1. 記入① アカウント名と指示
`ACCOUNT`にはOmniCampusのアカウント名を入力してください。氏名・所属・メールは入力しません。未記入のままではセル5で停止します。

初回は`USER_INSTRUCTION`と`USER_EXPECTED_TARGETS`を既定値のままにしてください。指示を変更する場合は、人が期待する点検順序も`B,A`のようにカンマ区切りで入力します。A/Bを1～4回指定でき、繰り返しは`B,A,B`と残します。この期待値は採点だけに使い、LLMへ渡したりAI出力を修正したりしません。

`STALE_THRESHOLD`と`MODEL_REVISION`は初回は変更しません。モデルSHAは旧版と同じものを既定にし、プロンプト変更前後を比較します。ACCOUNTだけの誤記は、完了後でもこのセルを修正・実行してから11・12を再実行すれば、計算をやり直さずに提出物を再生成できます。その場合セル5は再実行しません。
''')
    config_code = '#@title 1. 記入① アカウント名と指示\nimport json\nBASE_SETTINGS = json.loads('+repr(json.dumps(config,ensure_ascii=False))+')\n'
    for variable,key,kind in [('ACCOUNT','account','string'),('USER_INSTRUCTION','user_instruction','string')]:
        config_code += f'{variable} = {config[key]!r} #@param {{type:"{kind}"}}\n'
    expected_text = ','.join(config.get('user_expected_targets', ['B', 'A']))
    config_code += f'USER_EXPECTED_TARGETS = {expected_text!r} #@param {{type:"string"}}\n'
    config_code += '# 以下は初回は変更しないでください。\n'
    for variable,key,kind in [('STALE_THRESHOLD','stale_threshold','number'),('MODEL_REVISION','model_revision','string')]:
        config_code += f'{variable} = {config[key]!r} #@param {{type:"{kind}"}}\n'
    config_code += '# USER_EXPECTED_TARGETSは人が指定する採点用の順序で、LLMへの入力には使いません。\n'
    code(config_code)
    md('''## 2. 専用ランタイムの準備
OSMesa・FFmpeg・日本語フォントを導入します。テキストモデルで使わないtorchvision/torchaudio/librosaは、この専用Colabランタイムから削除します。Colab標準のtorch/numpyは置換しません。QwenはPyTorchだけを使用し、不要なTensorFlow等を読み込まない設定をimport前に適用します。手元のPCでこのセルを実行しないでください。
''')
    code((root/'setup_colab.py').read_text(encoding='utf-8'))
    md('## 3. ソース展開\n以下は実行する全ソースです。依存ライブラリとモデル以外のダウンロードは不要です。')
    names=[p.relative_to(root).as_posix() for p in sorted(root.glob('*.py'))]
    names += [p.relative_to(root).as_posix() for p in sorted((root/'tests').glob('*.py'))]
    names += [n for n in ['README.md','操作手順.md','requirements.txt','THIRD_PARTY_NOTICES.md','LICENSE','VALIDATION.md'] if (root/n).exists()]
    for name in names:
        code('%%writefile '+name+'\n'+(root/name).read_text(encoding='utf-8'))
    md('## 4. 単体・回帰テスト\nテスト用代替や合成入力は診断専用です。AIの正解率や物理性能の実績には含めません。')
    code('''import subprocess, sys, importlib
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
# 再実行時にメモリ上の旧ソースを混ぜない。ソースを変更した場合はランタイム再起動を推奨。
for name in list(sys.modules):
    if name.startswith('inspection_') or name in ('run_project', 'notebook_builder'):
        del sys.modules[name]
importlib.invalidate_caches()
''')
    md('''## 5. 初回開始／再開
設定・ソース・評価指示をrun開始時に固定します。通常はAUTOのままで実行してください。ローカルに実験があれば使用し、消えていればDriveの検証済みチェックポイントを復元します。どちらにも保存がない初回だけ新しいrunを作ります。破損・条件不一致はエラーにして止めます。

v3では旧版と保存先を分けています。旧版のrunは復元せず、初回のAUTOでv3の新しいrunを作ります。指示・期待順序・閾値などを変えて取り直す場合だけNEWを選択します。ACCOUNTだけの訂正は、セル1と11・12で行えます。

Google Driveの接続画面で許可してください。保存先はMyDrive/PAI_Inspection_Checkpoints_v3です。計算中のデータは/contentに置き、完了した単位ごとにZIPと検証情報をDriveへ保存します。保存に失敗した場合は後続へ進みません。各runの直近3件の検証済みチェックポイントを保持します。

再開時は保存済み設定を使います。設定変更やコード更新で新しい実験を行う場合はNEWを選んでください。過去の特定runはRESUME_RUN_IDで指定できます。旧版とソースが違うバックアップは自動移行しません。
''')
    code('''#@title 5. 初回開始／再開
RUN_MODE = "AUTO" #@param ["AUTO", "NEW", "RESUME_LOCAL", "RESTORE_DRIVE", "RESTORE_ZIP"]
CHECKPOINT_VAULT = "/content/drive/MyDrive/PAI_Inspection_Checkpoints_v3" #@param {type:"string"}
RESUME_RUN_ID = "" #@param {type:"string"}
BACKUP_ZIP = "/content/PAI_backup.zip" #@param {type:"string"}
from pathlib import Path
from google.colab import drive
from inspection_core import Settings
from inspection_store import new_run, load_latest, restore_backup
from inspection_checkpoint import save_checkpoint, latest_checkpoint, restore_checkpoint
import re
if ACCOUNT == 'YOUR_OMNICAMPUS_ID' or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', ACCOUNT):
    raise ValueError('セル1のACCOUNTにOmniCampusアカウント名を入力・実行してから、このセルを再実行してください。氏名やメールは入力しません。')
ROOT = Path('/content/pai_inspection_final_v3')
drive.mount('/content/drive')
VAULT = Path(CHECKPOINT_VAULT)
VAULT.mkdir(parents=True, exist_ok=True)
requested_run = RESUME_RUN_ID.strip() or None
restored = False
if RUN_MODE == 'AUTO':
    if (ROOT/'latest_run.json').exists() and requested_run is None:
        settings, RUN = load_latest(ROOT)
        restored = True
    else:
        saved = latest_checkpoint(VAULT, run_id=requested_run)
        if saved is not None:
            settings, RUN, receipt = restore_checkpoint(ROOT, VAULT, run_id=requested_run)
            restored = True
        elif requested_run is not None:
            raise FileNotFoundError('指定runのチェックポイントがありません: '+requested_run)
        else:
            RUN_MODE = 'NEW'
elif RUN_MODE == 'RESUME_LOCAL':
    if requested_run is not None:
        raise ValueError('特定runの指定にはRESTORE_DRIVEを使用してください。')
    settings, RUN = load_latest(ROOT)
    restored = True
elif RUN_MODE == 'RESTORE_DRIVE':
    settings, RUN, receipt = restore_checkpoint(ROOT, VAULT, run_id=requested_run)
    restored = True
elif RUN_MODE == 'RESTORE_ZIP':
    restore_backup(ROOT, BACKUP_ZIP)
    settings, RUN = load_latest(ROOT)
    restored = True
elif RUN_MODE != 'NEW':
    raise ValueError('不明なRUN_MODE: '+RUN_MODE)
if not restored:
    expected_targets = [part.strip().upper() for part in USER_EXPECTED_TARGETS.split(',')]
    if not expected_targets or any(not re.fullmatch(r'[A-Z]', target) for target in expected_targets):
        raise ValueError('USER_EXPECTED_TARGETSはB,AやB,A,Bのように、対象IDをカンマで区切ってください。')
    values = dict(BASE_SETTINGS, account=ACCOUNT, user_instruction=USER_INSTRUCTION,
                  user_expected_targets=expected_targets,
                  stale_threshold=float(STALE_THRESHOLD), model_revision=MODEL_REVISION)
    settings = Settings(**values)
    settings.validate()
    RUN = new_run(ROOT, settings)
else:
    print('保存済みの実験条件で再開します。指示・期待順序・閾値などの変更を適用するにはNEWを選んでください。')
    print('提出物に表示するACCOUNTはセル1の値を使い、実験開始時の設定記録は残します。')

def checkpoint(event):
    receipt = save_checkpoint(ROOT, RUN, VAULT, event=event, keep=3)
    print('Drive保存・検証済み:', event, receipt['created_utc'],
          f"{receipt['archive_bytes']/1024**2:.2f} MB", receipt['archive_path'])
    return receipt

checkpoint('resume_initialized' if restored else 'run_initialized')
print('ローカル作業先:', RUN)
print('Drive保存先:', VAULT)
print('今回の設定:', settings.user_instruction, '鮮度閾値:', settings.stale_threshold)
print('自分の指示の採点用順序:', settings.user_expected_targets)
''')
    md('## 6. 6秒の制御・描画テスト\nAIを使用しない診断です。到達誤差・描画・画像保存と、このランタイムの実行速度を確認します。再開時も環境確認のため実行します。失敗した場合は推論へ進まず、エラーを確認してください。')
    code('''from run_project import control_preflight
from IPython.display import Video, display
preflight = control_preflight(ROOT, RUN, settings)
checkpoint('control_preflight_complete')
print('最終位置誤差 [mm]:', preflight['final_error_m']*1000)
if 'wall_time_s' in preflight:
    print('制御テストの実時間 [秒]:', round(preflight['wall_time_s'], 1))
if preflight.get('render_frames_per_wall_s', 0) > 0:
    render_fps = preflight['render_frames_per_wall_s']
    print('描画を含む処理速度 [frames/実時間秒]:', round(render_fps, 2))
    print('1000フレーム分の描画処理の目安 [分]:', round(1000/render_fps/60, 1))
    print('これはモデル取得・推論・動画編集・Drive保存を含まない参考値です。')
display(Video(preflight['clip'], embed=True, width=800))
''')
    md('''## 7. Qwen
旧版と同じ24指示と、自分の指示1件の計25件を推論します。順序・反復の取りこぼしと曖昧指示への対応を見直したプロンプトですが、改善したかは実測で判定します。生出力を正解へ補正しません。採点用の期待順序はLLMへ渡しません。

24指示は旧版の失敗分析に使った既知の回帰テストです。元のdevelopment/holdoutという区分名は記録上残りますが、今回は未使用データへの汎化性能とは説明しません。モデルSHA・配置等も揃え、旧版に対して何が変わったかを確認します。

1件完了ごとにDriveへ保存します。途中までの推論は、モデル・プロンプト・生成条件が一致するときだけ再利用します。モデル重みはバックアップに含めないため、未完の推論がある新しいランタイムでは再取得・再読込が必要です。全25件が保存済みならモデルの読込を省略します。

制御・描画テストの後に起きた `LLVM ... registered more than once` は、LLVMの登録が重複してプロセスが終了したエラーです。TensorFlowと描画ライブラリの組合せが疑われますが、原因となる組合せは未確定です。この版ではQwenをPyTorch専用の新しい子プロセスで動かし、描画処理と分離して回避を図ります。警告を非表示にする変更ではありません。

モデル読込の上限は20分、1件の推論の上限は3分です。通常の所要時間を表す値ではありません。開始時にログの保存先が表示されます。子プロセスが異常終了した場合や時間切れの場合は、このセルにエラーとログの場所を表示して停止します。ログを確認し、原因を解消して推論セルを再実行してください。保存済みの推論は条件を検査して再利用します。
''')
    code('''#@title 7. Qwen
from run_project import generate_plans
plans = generate_plans(ROOT, RUN, settings, checkpoint=checkpoint)
print('固定済み推論:', len(plans), '件')
''')
    md('## 8. 言語解釈・タスク達成と変更前後の比較\n各エピソード終了後にDriveへ保存します。resume=Trueでは同じ条件の検証済み結果を再利用し、中断したエピソードから再実行します。物理計算の途中ステップからの復帰ではありません。独立した再試行を行う場合は、意図的にresume=Falseへ変更してください。\n\n旧版ZIPから保存した基準と、今回の同じ24指示の成績を表示します。正常指示の達成と異常指示の拒否は分けて見ます。入力や条件が違う場合は、同条件の改善比較として扱わないでください。')
    code('''#@title 8. 評価と変更前後の比較
from run_project import evaluate
from inspection_baseline import baseline_comparison
evaluation = evaluate(ROOT, RUN, settings, plans, resume=True, checkpoint=checkpoint)
print(json.dumps(evaluation['summary'], ensure_ascii=False, indent=2))
print('旧版と今回の比較（既知の24指示による回帰テスト）')
print(json.dumps(baseline_comparison(evaluation), ensure_ascii=False, indent=2))
if evaluation['summary']['end_to_end_success_n'] == 0:
    print('正常タスクの指示達成が0件です。完成品とは扱わず、出力・エラーを分析してください。')
''')
    md('''## 9. 同じ通信系列で3方式×4条件を比較
監視なし／受信途絶監視／取得時刻による鮮度監視を、正常・途絶・継続的遅延・復旧時の順序逆転の4条件で比較します。対象は静止し、通信時計はシミュレーション時刻に統一します。

**このセルは、あらかじめ決めたA→Bの参照計画を使う制御・監視の比較実験です。** AIが出した計画でも、AIの誤りを補正する代替計画でもありません。セル8のAI成績とは別に記録します。特定のAI指示が失敗しても、監視方式の比較は実行できます。

全方式で同じ観測採用規則を使います。無効・重複・逆順パケットは採用しません。開始・再開は有効な観測と必要な低速継続を確認します。保留の効果だけでなく、完了時間も比較してください。
''')
    code('''#@title 9. 監視方式の比較（固定参照計画）
from run_project import compare_freshness_guard
comparison = compare_freshness_guard(ROOT, RUN, settings, plans, resume=True, checkpoint=checkpoint)
if not comparison['available']:
    print(comparison['reason'])
else:
    for r in comparison['records']:
        print(r['scenario'], r['mode'], '完了:', r['success'],
              '所要秒:', r['sim_duration_s'], '古い情報下の移動秒:', r['stale_moving_s'],
              '保留中最大変位mm:', 1000*r['max_hold_drift_m'])
''')
    md('## 10. 4場面を録画\n指示一致と計画実行完了を分けて表示します。自分の指示も、セル1で人が登録したUSER_EXPECTED_TARGETSと照合して採点します。期待値でAI出力を差し替えません。AI推論は事前に行い、動作はシミュレーション時間の1倍速で収録します。ソフトウェア描画には時間がかかります。\n\nここまで実行したら、8～10の結果を確認してからセル11を記入してください。エラーなく最後まで実行できたことと、指示達成・比較成功は別です。')
    code('''#@title 10. 4場面の録画
from run_project import record_demonstrations
demonstrations = record_demonstrations(ROOT, RUN, settings, plans, resume=True, checkpoint=checkpoint)
for d in demonstrations['records']:
    print(d['title'], '指示達成:', d['task_success'], '計画実行:', d['success'])
''')
    md('''## 11. 記入② 本人の変更内容・考察・提出物生成
実際に行った変更と測定結果だけを記入してください。空欄や未記入のままなら下書きです。実験設定を変更せずに文章だけを更新でき、再生成ごとに新しい出版フォルダができます。このセルの途中で中断した場合はセル全体を再実行します。動画の合成途中からは再開しません。完成後に提出物をDriveへ保存します。

`OWN_CHANGES`は500字以内、`REFLECTION`は800字以内です。自分が選んだ設計・変更した値・比較した条件と、その理由を記載します。AIに生成してもらったコードを、すべて自分だけで実装したとは書かないでください。考察は、セル8～10の実測値から「変わった点・残った失敗・保留による代償」を記述してください。結果を確認する前に成功した文章を入れないでください。

**ACCOUNTだけの訂正は、セル1で変更・実行してから、このセルと12を再実行します。** 推論・物理実験は再計算せず、提出用のIDを別の表示情報として記録します。実験開始時のACCOUNTを含む設定記録はそのまま残ります。

配布ノートブックを新規アップロードしてDriveから実験を復元した場合、このセルの本人記入文は自動入力されません。編集済みノートブックも保存するか、直前の`publication_inputs.json`の`own_changes`・`reflection`から戻し、未記入へ戻っていないか確認してください。

正常指示の成功と異常指示の拒否を混ぜた成功率は使いません。比較の値を説明するときは、保留後最大変位を停止距離と呼ばず、接触異常判定を完全な無接触保証と呼ばないでください。
''')
    code('''#@title 11. 記入② 本人の変更内容・考察・提出物生成
OWN_CHANGES = "未記入：自分が実施した変更を記入してください。" #@param {type:"string"}
REFLECTION = "未記入：実測値、失敗、効果と代償、限界を記入してください。" #@param {type:"string"}
from inspection_outputs import create_submission
publication = create_submission(ROOT, RUN, settings, OWN_CHANGES, REFLECTION,
                                submission_account=ACCOUNT)
checkpoint('publication_complete')
print(json.dumps(publication['checks'], ensure_ascii=False, indent=2))
if not publication['checks']['ready_for_manual_review']:
    print('必須チェックに未達があります。falseの理由とPDF・動画を確認してから提出判断をしてください。合否判定ではありません。')
print('提出用ZIP:', publication['bundle'])
print('説明PDF:', publication['pdf'])
print('提出動画:', publication['video'])
display(Video(publication['video'], embed=True, width=900))
display(__import__('IPython.display', fromlist=['IFrame']).IFrame(publication['pdf'], width=900, height=600))
''')
    md('''## 12. 成果物をPCへ保存
ZIPにはPDF、60～180秒MP4、GitHub用ソースZIP、チェック結果、参照した試行の記録が入ります。完成済み提出物はDriveのチェックポイントにも含まれますが、PCにも保存して内容を確認してください。

本人が内容を確認後、ソースZIPをpublic GitHubへ、MP4をYouTubeの限定公開以上へ登録し、PDFとURLをOmniCampusへ提出します。自動公開はしません。提出条件は講座の最新案内を確認してください。
''')
    code("#@title 12. 提出用ZIPをPCへ保存\nfrom google.colab import files\nfiles.download(publication['bundle'])\n")
    md('''## 任意：手動バックアップと切断後の再開
自動保存に加え、ZIPをPCにも保存できます。設定、モデルSHA、ソースsnapshot、取得済み計画、各試行と選択情報を含めます。モデル重みは含みません。ZIPからの復元はセル5のRESTORE_ZIPを使います。

ブラウザとの接続だけが切れ、ランタイムが残っていれば処理が継続している場合があります。再接続後に実行状態を確認し、同じ処理を重複起動しないでください。ランタイムが残っている場合はセットアップ・ソース展開を繰り返さず必要なセルから続けます。MuJoCo読込後にセットアップをやり直す場合は、ランタイム再起動が必要です。ランタイム自体が削除された場合は、このノートブックを新しいGPUランタイムで上から実行し直してDriveを接続し、AUTOで復元します。

自動保存はColabの制限を延長する仕組みではありません。切断直前の未完了アップロードは復元対象にせず、最後の検証済みチェックポイントを使用します。そのため直近の未保存エピソードは再実行になる場合があります。実行環境のバージョンが変わった場合やソースが違う場合は、混在を避けるため再開を拒否します。保存時と同じ環境を用意するか、NEWで再実行してください。
''')
    code('''#@title 途中結果をバックアップ
BACKUP_NOW = False #@param {type:"boolean"}
if BACKUP_NOW:
    from inspection_store import create_backup, unique_id
    from google.colab import files
    backup = create_backup(ROOT, RUN, ROOT/'backups'/(unique_id('PAI_backup')+'.zip'))
    files.download(str(backup))
''')
    document={'cells':cells,'nbformat':4,'nbformat_minor':5,
              'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                          'language_info':{'name':'python'},'accelerator':'GPU',
                          'colab':{'name':'PAI_Inspection_Colab_v3.ipynb','provenance':[]}}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(document,ensure_ascii=False,indent=1),encoding='utf-8')
    return output

if __name__=='__main__':
    build_notebook(Path(__file__).parent,Path(__file__).parent/'PAI_Inspection_Colab_v3.ipynb')
