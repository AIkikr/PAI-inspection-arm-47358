"""Run only in a fresh, dedicated Google Colab GPU runtime."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

# Must precede imports: Qwen uses PyTorch only. TensorFlow/JAX can load LLVM
# libraries that conflict with Mesa/OSMesa in the same interpreter.
os.environ['USE_TORCH']='1'
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
os.environ['USE_TORCH_XLA']='0'
os.environ['FORCE_TF_AVAILABLE']='0'

try:
    import google.colab
except ImportError as exc:
    raise RuntimeError('このセットアップはGoogle Colab専用です。手元のPCでは実行しないでください。') from exc
if any(name in sys.modules for name in ('mujoco','OpenGL','tensorflow','jax','jaxlib')):
    raise RuntimeError('描画またはTensorFlow/JAXが読込済みです。ランタイムを再起動して上から実行してください。')
os.environ['MUJOCO_GL']='osmesa'
os.environ['PYOPENGL_PLATFORM']='osmesa'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['DO_NOT_TRACK']='1'
import torch
if not torch.cuda.is_available():
    raise RuntimeError('ランタイム → ランタイムのタイプを変更 → T4などのGPUを選択してください。')
print('GPU:',torch.cuda.get_device_name(0),'torch:',torch.__version__)

def install(command):
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if result.returncode:
        print(result.stdout[-5000:])
        raise RuntimeError('セットアップに失敗しました。表示された末尾のエラーを確認してください。')

install(['apt-get','update','-qq'])
install(['apt-get','install','-y','-qq','libosmesa6','libgl1','libegl1','libgl1-mesa-dri','ffmpeg','fonts-noto-cjk'])
# This text-only project does not use vision/audio extensions. Dedicated runtime only.
install([sys.executable,'-m','pip','uninstall','-y','torchvision','torchaudio','librosa'])
packages=['mujoco==3.3.7','transformers==4.57.1','accelerate==1.11.0',
          'imageio==2.37.0','imageio-ffmpeg==0.6.0','reportlab==4.4.4',
          'Pillow>=10.4,<13','huggingface-hub>=0.34,<1','safetensors>=0.4.3','pypdf>=5,<7']
install([sys.executable,'-m','pip','install','--quiet',*packages])
import importlib.metadata
for module,package in [('PIL','Pillow'),('imageio','imageio'),('transformers','transformers')]:
    if module in sys.modules and hasattr(sys.modules[module],'__version__'):
        if str(sys.modules[module].__version__)!=importlib.metadata.version(package):
            raise RuntimeError('インストール前に読込済みのライブラリがあります。ランタイムを再起動して再実行してください。')
ROOT=Path('/content/pai_inspection_final_v3')
ROOT.mkdir(parents=True,exist_ok=True)
(ROOT/'tests').mkdir(exist_ok=True)
os.chdir(ROOT)
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
print('セットアップ完了:',ROOT)
