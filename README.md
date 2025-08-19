# fine-tune-demo
Fine tuning example using Gemma 3 1B

# Requirements
- Linux
- Must have `llama.cpp`

# Commands

1. Convert the HF `safetensors` to `gguf`:

```bash
python convert_hf_to_gguf.py --outfile /path/to/gemma-3-1b-ft.gguf /path/to/gemma-3-1b-finetuned-merged
```

2. Build `llama.cpp`

If you don't have an Nvidia GPU, you can leave `-DGGML_CUDA=ON` out of the command.

```bash
mkdir -p $HOME/bin/llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_INSTALL_PREFIX=$HOME/bin/llama.cpp -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --config Release -j $(nproc)
cmake --install build --config Release
```

3. Then serve:

```bash
$HOME/bin/llama.cpp/bin/llama-server -m /path/to/gemma-3-1b-ft.gguf -ngl 999 --host 0.0.0.0 --port 8000
```

4. Navigate to `http://localhost:8000` to see Web UI and test.