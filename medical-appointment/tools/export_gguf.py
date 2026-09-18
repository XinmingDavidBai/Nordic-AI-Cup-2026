"""E9 export: merge a LoRA adapter (downloaded from the HPC run) into the
base model, convert to GGUF, and register it with the local ollama so
tools/pipeline_eval.py / local_evaluator.py can point OLLAMA_MODEL at it.

Run locally (CPU is fine -- this is I/O and format conversion, not training):
    python tools/export_gguf.py --adapter checkpoints/final --name llama3.2-medqa-ft
    python tools/export_gguf.py --adapter checkpoints/fold0 --name llama3.2-medqa-ft-fold0

Needs: pip install torch transformers peft gguf sentencepiece protobuf
Clones llama.cpp (shallow, just for its conversion script) into
tools/.llama.cpp/ on first run if not already present.
"""
import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--adapter', required=True, help='path to a LoRA adapter dir, e.g. checkpoints/final')
ap.add_argument('--base-model', default=os.getenv('BASE_MODEL', 'unsloth/Llama-3.2-3B-Instruct'))
ap.add_argument('--name', required=True, help='ollama model name to create, e.g. llama3.2-medqa-ft')
ap.add_argument('--quantize', default='q4_K_M', help='ollama quantization level (matches llama3.2:3b\'s own Q4_K_M)')
ap.add_argument('--workdir', default=None, help='scratch dir for the merged model + gguf (default: tools/.export/<name>)')
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
workdir = args.workdir or os.path.join(HERE, '.export', args.name)
merged_dir = os.path.join(workdir, 'merged')
gguf_path = os.path.join(workdir, f'{args.name}.f16.gguf')
llama_cpp_dir = os.path.join(HERE, '.llama.cpp')
os.makedirs(workdir, exist_ok=True)

# --- 1. merge LoRA into the base model -------------------------------------
if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
    print(f'[1/4] merging {args.adapter} into {args.base_model} ...')
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
    print(f'[1/4] {merged_dir} already exists, skipping merge')

# --- 2. get llama.cpp's conversion script -----------------------------------
convert_script = os.path.join(llama_cpp_dir, 'convert_hf_to_gguf.py')
if not os.path.isfile(convert_script):
    print('[2/4] cloning llama.cpp (shallow) for its GGUF conversion script ...')
    subprocess.run(['git', 'clone', '--depth', '1',
                     'https://github.com/ggml-org/llama.cpp.git', llama_cpp_dir], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                     os.path.join(llama_cpp_dir, 'requirements', 'requirements-convert_hf_to_gguf.txt')], check=True)
else:
    print('[2/4] llama.cpp already present, skipping clone')

# --- 3. convert merged HF model to F16 GGUF ---------------------------------
if not os.path.isfile(gguf_path):
    print('[3/4] converting to GGUF (f16) ...')
    subprocess.run([sys.executable, convert_script, merged_dir,
                     '--outfile', gguf_path, '--outtype', 'f16'], check=True)
else:
    print(f'[3/4] {gguf_path} already exists, skipping conversion')

# --- 4. write a Modelfile matching llama3.2:3b's template/params and create ---
print(f'[4/4] creating ollama model "{args.name}" (quantize={args.quantize}) ...')
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

subprocess.run(['ollama', 'create', args.name, '-f', modelfile_path, '-q', args.quantize], check=True)
print(f'\nDone. Point the pipeline at it with:')
print(f'  OLLAMA_MODEL={args.name} python3 tools/pipeline_eval.py results_{args.name}.json')
