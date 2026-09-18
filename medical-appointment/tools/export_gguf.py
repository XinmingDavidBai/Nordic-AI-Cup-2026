"""E9 export: merge a LoRA adapter (downloaded from the HPC run) into the
base model, convert to GGUF, and register it with the local ollama so
tools/pipeline_eval.py / local_evaluator.py can point OLLAMA_MODEL at it.

Run locally (CPU is fine -- this is I/O and format conversion, not training):
    python tools/export_gguf.py --adapter checkpoints/final --name llama3.2-medqa-ft
    python tools/export_gguf.py --adapter checkpoints/fold0 --name llama3.2-medqa-ft-fold0

Run this via a dedicated venv, not your everyday system Python -- do NOT
`pip install` its deps globally. On 2026-09-18 that broke numpy/scipy/
scikit-learn for every other project sharing this machine's system Python
(via transformers' generation utils pulling in sklearn), because llama.cpp's
own requirements file pins its own torch/scipy/scikit-learn/narwhals chain.
One-time setup:
    python3 -m venv tools/.export-venv
    source tools/.export-venv/bin/activate
    pip install torch transformers peft gguf sentencepiece protobuf accelerate
Then run every time via that venv's python, e.g.:
    tools/.export-venv/bin/python3 tools/export_gguf.py --adapter checkpoints/final --name llama3.2-medqa-ft

Clones llama.cpp (shallow, just for its conversion script) into
tools/.llama.cpp/ on first run if not already present. Deliberately does NOT
run llama.cpp's own requirements-convert_hf_to_gguf.txt -- the packages above
already cover what convert_hf_to_gguf.py actually needs for a standard Llama
conversion, and that requirements file pulls in a much larger, conflicting
chain (a pinned CPU torch build, scipy, scikit-learn, narwhals) meant for
broader legacy-model support this project doesn't need.
"""
import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--adapter', required=True, help='path to a LoRA adapter dir, e.g. checkpoints/final')
ap.add_argument('--base-model', default=os.getenv('BASE_MODEL', 'unsloth/Llama-3.2-3B-Instruct'))
ap.add_argument('--name', required=True, help='ollama model name to create, e.g. llama3.2-medqa-ft')
ap.add_argument('--quantize', default='q4_K_M',
                 help='ollama quantization level (matches llama3.2:3b\'s own Q4_K_M), or "none" to skip '
                      'quantization -- ollama create -q needs the F16 source AND the quantized output on '
                      'disk simultaneously (~8.4GB peak for a 3B model), which a nearly-full disk may not '
                      'have even when the F16 alone would fit; "none" avoids that peak at the cost of a '
                      'larger (~6.4GB) and slower-to-run model')
ap.add_argument('--workdir', default=None, help='scratch dir for the merged model + gguf (default: tools/.export/<name>)')
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
workdir = args.workdir or os.path.join(HERE, '.export', args.name)
merged_dir = os.path.join(workdir, 'merged')
gguf_path = os.path.join(workdir, f'{args.name}.f16.gguf')
llama_cpp_dir = os.path.join(HERE, '.llama.cpp')
os.makedirs(workdir, exist_ok=True)

# --- 1. merge LoRA into the base model -------------------------------------
if os.path.isfile(gguf_path):
    print(f'[1/5] {gguf_path} already exists, skipping merge (nothing to convert it from)')
elif not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
    print(f'[1/5] merging {args.adapter} into {args.base_model} ...')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(merged_dir)
    print(f'  saved merged model to {merged_dir}')
else:
    print(f'[1/5] {merged_dir} already exists, skipping merge')

# --- 2. get llama.cpp's conversion script -----------------------------------
# Deliberately NOT running `pip install -r requirements-convert_hf_to_gguf.txt`
# here -- see the module docstring. The venv this script is meant to run in
# already has what convert_hf_to_gguf.py needs (gguf, numpy, sentencepiece,
# protobuf, transformers).
convert_script = os.path.join(llama_cpp_dir, 'convert_hf_to_gguf.py')
if not os.path.isfile(convert_script):
    print('[2/5] cloning llama.cpp (shallow) for its GGUF conversion script ...')
    subprocess.run(['git', 'clone', '--depth', '1',
                     'https://github.com/ggml-org/llama.cpp.git', llama_cpp_dir], check=True)
else:
    print('[2/5] llama.cpp already present, skipping clone')

# --- 3. convert merged HF model to F16 GGUF ---------------------------------
if not os.path.isfile(gguf_path):
    print('[3/5] converting to GGUF (f16) ...')
    subprocess.run([sys.executable, convert_script, merged_dir,
                     '--outfile', gguf_path, '--outtype', 'f16'], check=True)
else:
    print(f'[3/5] {gguf_path} already exists, skipping conversion')

# --- 4. write a Modelfile matching llama3.2:3b's template/params and create ---
print(f'[4/5] creating ollama model "{args.name}" (quantize={args.quantize}) ...')
template = subprocess.run(['ollama', 'show', 'llama3.2:3b', '--template'],
                           capture_output=True, text=True, check=True).stdout
params = subprocess.run(['ollama', 'show', 'llama3.2:3b', '--parameters'],
                         capture_output=True, text=True, check=True).stdout

modelfile_path = os.path.join(workdir, 'Modelfile')
with open(modelfile_path, 'w') as f:
    f.write(f'FROM {gguf_path}\n\n')
    f.write('TEMPLATE """' + template + '"""\n\n')
    for line in params.strip().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            key, val = parts
            f.write(f'PARAMETER {key} {val}\n')

create_cmd = ['ollama', 'create', args.name, '-f', modelfile_path]
if args.quantize != 'none':
    create_cmd += ['-q', args.quantize]
subprocess.run(create_cmd, check=True)

# --- 5. clean up the merged HF model + F16 GGUF (ollama keeps its own
#        quantized copy under ~/.ollama, so these intermediates are dead
#        weight -- each fold's merged+F16 pair is ~12GB, easily enough to
#        fill a disk with only a few GB free, learned the hard way exporting
#        fold0 on 2026-09-18) -----------------------------------------------
import shutil
if os.path.isdir(merged_dir):
    shutil.rmtree(merged_dir)
if os.path.isfile(gguf_path):
    os.remove(gguf_path)
print(f'[5/5] cleaned up {merged_dir} and {gguf_path} (kept: ollama\'s own copy of "{args.name}")')

print(f'\nDone. Point the pipeline at it with:')
print(f'  OLLAMA_MODEL={args.name} python3 tools/pipeline_eval.py results_{args.name}.json')
