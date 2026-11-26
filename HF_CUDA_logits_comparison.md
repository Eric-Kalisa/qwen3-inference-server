# Qwen3 inference comparison

This directory compares greedy generation from the CUDA Tile kernels and
Hugging Face PyTorch on an A100. The default model is `Qwen/Qwen3-0.6B`; the
same code also supports `Qwen/Qwen3-4B`. Both streams recompute the full sequence
for every token without a KV cache. The current attention kernel supports at
most 128 prompt and generated tokens combined.

Initialize the tokenizer submodule before running:

```powershell
git submodule update --init --recursive
```

Run the following Modal commands in PowerShell from this `inference` directory.
Set UTF-8 output encoding so Modal's status messages display correctly in the
Windows console:

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
```

To compile all CUDA libraries and verify that each shared library loads and
exports the functions used by the runner, without loading a model or running
tensor calculations:

```powershell
..\.venv\Scripts\python.exe -m modal run .\modal_run_qwen3_inference.py --build-only
```

To run the tokenizer, logits, and greedy generation integration tests on an A100:

```powershell
..\.venv\Scripts\python.exe -m modal run .\modal_run_qwen3_inference.py --run-tests
```

To verify that each input token's embedding lands at the same sequence position
in CUDA and Hugging Face before running decoder layers, use:

```powershell
..\.venv\Scripts\python.exe -m modal run .\modal_run_qwen3_inference.py --check-hidden-state-positions --prompt "The capital of France is"
```

This checks the embedding row at every prompt position and fails if any row
differs. It does not run attention, MLP, or decoder-layer calculations.

To compare hidden-state boundaries layer by layer, logging the input shape and
attention/MLP dimensions before each CUDA layer and stopping at the first
out-of-tolerance hidden state, run:

```powershell
..\.venv\Scripts\python.exe -m modal run .\modal_run_qwen3_inference.py --diagnose-layer-hidden-states --prompt "The capital of France is"
```

The diagnostic uses `atol=0.1, rtol=0.02`, matching the inference logit
comparison. It reports the first out-of-tolerance element and does not run any
later CUDA decoder layers after a failing boundary.

The default comparison uses the 0.6B model and generates three tokens:

```powershell
..\.venv\Scripts\python.exe -m modal run .\modal_run_qwen3_inference.py
```

To ask the 4B model a question and generate a short answer from both streams:

```powershell
..\.venv\Scripts\python.exe -m modal run .\modal_run_qwen3_inference.py --model-id Qwen/Qwen3-4B --prompt "<|im_start|>user`nWhat is one practical way to reduce household energy use?`n<|im_end|>`n<|im_start|>assistant`n" --max-new-tokens 64
```

Modal prints each stream's generated answer and token IDs. For steps where both
streams share the same input prefix, it also prints the number of logits outside
`atol=0.1, rtol=0.02` and the 10 largest mismatches. After generated tokens
diverge, later logits use different input prefixes and are not compared.

To use the HTTP interface, start both model routes:

```powershell
..\.venv\Scripts\python.exe -m modal serve .\modal_run_qwen3_inference.py
```

Modal prints a URL for each route: `serve_qwen3` uses 0.6B and `serve_qwen3_4b`
uses 4B. Send the same JSON request to either route's `/generate` endpoint:

```powershell
$body = @{ prompt = "<|im_start|>user`nWhat is one practical way to reduce household energy use?`n<|im_end|>`n<|im_start|>assistant`n"; max_new_tokens = 64 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "<MODEL_ROUTE_URL>/generate" -ContentType "application/json" -Body $body
```

The inference Modal image installs CUDA-enabled PyTorch from its own
`requirements.txt`; the project requirements remain CPU-only for module
case generation.


## Results - Before any KV cache, paged attention, or shared resources

## QWEN3-06.B forward pass with 64 token generation

Prompt: Write a short paragraph explaining how the Moon affects Earth's tides.
Prompt tokens: 14
CUDA: " The paragraph should include the following elements: 1) the Moon's gravity, 2) the Earth's rotation, 3) the tidal forces, 4) the tides, 5) the Moon's position relative to Earth, 6) the effects of the Moon's gravity on Earth's tides"
PyTorch: " The paragraph should include the following elements: 1) the Moon's gravity, 2) the Earth's rotation, 3) the tidal forces, 4) the tides, 5) the Moon's position relative to Earth, 6) the effects of the Moon's gravity on Earth's tides"
step 1: CUDA token=576, PyTorch token=576
  logits: 8322/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.328125
  logit[68892]: CUDA=-3.796875, PyTorch=-4.125, abs_diff=0.328125
  logit[33159]: CUDA=-3.125, PyTorch=-3.4375, abs_diff=0.3125
  logit[17072]: CUDA=2.015625, PyTorch=1.7109375, abs_diff=0.3046875
  logit[27307]: CUDA=0.37109375, PyTorch=0.073730469, abs_diff=0.29736328
  logit[50485]: CUDA=-2.578125, PyTorch=-2.875, abs_diff=0.296875
  logit[6234]: CUDA=-1.5703125, PyTorch=-1.859375, abs_diff=0.2890625
  logit[110116]: CUDA=-1.7265625, PyTorch=-2.015625, abs_diff=0.2890625
  logit[25136]: CUDA=0.31054688, PyTorch=0.026245117, abs_diff=0.28430176
  logit[1768]: CUDA=0.69140625, PyTorch=0.40820312, abs_diff=0.28320312
  logit[23473]: CUDA=-3.84375, PyTorch=-4.125, abs_diff=0.28125
step 2: CUDA token=14311, PyTorch token=14311
  logits: 31031/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.34375
  logit[57102]: CUDA=-1.3515625, PyTorch=-1.6953125, abs_diff=0.34375
  logit[95626]: CUDA=2.515625, PyTorch=2.171875, abs_diff=0.34375
  logit[142993]: CUDA=-3.875, PyTorch=-4.21875, abs_diff=0.34375
  logit[12770]: CUDA=-0.5859375, PyTorch=-0.92578125, abs_diff=0.33984375
  logit[76640]: CUDA=-1.8359375, PyTorch=-2.171875, abs_diff=0.3359375
  logit[131200]: CUDA=0.203125, PyTorch=-0.12597656, abs_diff=0.32910156
  logit[60969]: CUDA=-0.69140625, PyTorch=-1.015625, abs_diff=0.32421875
  logit[22036]: CUDA=2.25, PyTorch=1.9296875, abs_diff=0.3203125
  logit[51657]: CUDA=0.038574219, PyTorch=-0.27539062, abs_diff=0.31396484
  logit[3536]: CUDA=7.8125, PyTorch=7.5, abs_diff=0.3125
step 3: CUDA token=1265, PyTorch token=1265
  logits: 6481/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[7512]: CUDA=10.625, PyTorch=10.3125, abs_diff=0.3125
  logit[98144]: CUDA=-4.65625, PyTorch=-4.96875, abs_diff=0.3125
  logit[1430]: CUDA=6.96875, PyTorch=6.6875, abs_diff=0.28125
  logit[95853]: CUDA=-4.28125, PyTorch=-4.5625, abs_diff=0.28125
  logit[79807]: CUDA=2.953125, PyTorch=2.671875, abs_diff=0.28125
  logit[42404]: CUDA=4.3125, PyTorch=4.03125, abs_diff=0.28125
  logit[4190]: CUDA=8.25, PyTorch=7.96875, abs_diff=0.28125
  logit[48162]: CUDA=2.140625, PyTorch=1.8671875, abs_diff=0.2734375
  logit[77373]: CUDA=-3.515625, PyTorch=-3.78125, abs_diff=0.265625
  logit[63501]: CUDA=2.59375, PyTorch=2.328125, abs_diff=0.265625
step 4: CUDA token=2924, PyTorch token=2924
  logits: 1444/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.234375
  logit[31776]: CUDA=3.765625, PyTorch=4, abs_diff=0.234375
  logit[14996]: CUDA=2, PyTorch=2.234375, abs_diff=0.234375
  logit[50511]: CUDA=0.8828125, PyTorch=1.1015625, abs_diff=0.21875
  logit[12260]: CUDA=3.71875, PyTorch=3.9375, abs_diff=0.21875
  logit[67061]: CUDA=3.28125, PyTorch=3.5, abs_diff=0.21875
  logit[15465]: CUDA=5.28125, PyTorch=5.5, abs_diff=0.21875
  logit[13342]: CUDA=0.3125, PyTorch=0.52734375, abs_diff=0.21484375
  logit[117417]: CUDA=-3.640625, PyTorch=-3.4375, abs_diff=0.203125
  logit[109706]: CUDA=-0.48046875, PyTorch=-0.27734375, abs_diff=0.203125
  logit[56761]: CUDA=0.78125, PyTorch=0.984375, abs_diff=0.203125
step 5: CUDA token=279, PyTorch token=279
  logits: 8790/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.2734375
  logit[77049]: CUDA=-0.125, PyTorch=-0.3984375, abs_diff=0.2734375
  logit[59572]: CUDA=1.3671875, PyTorch=1.109375, abs_diff=0.2578125
  logit[78675]: CUDA=0.59375, PyTorch=0.33789062, abs_diff=0.25585938
  logit[63205]: CUDA=-0.47265625, PyTorch=-0.7265625, abs_diff=0.25390625
  logit[92577]: CUDA=-2.296875, PyTorch=-2.546875, abs_diff=0.25
  logit[67558]: CUDA=-3.65625, PyTorch=-3.90625, abs_diff=0.25
  logit[65296]: CUDA=-7.0625, PyTorch=-7.3125, abs_diff=0.25
  logit[94963]: CUDA=-6.75, PyTorch=-7, abs_diff=0.25
  logit[111472]: CUDA=-6.25, PyTorch=-6.5, abs_diff=0.25
  logit[75308]: CUDA=-5.0625, PyTorch=-5.3125, abs_diff=0.25
step 6: CUDA token=2701, PyTorch token=2701
  logits: 1781/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.21875
  logit[18071]: CUDA=-4.6875, PyTorch=-4.90625, abs_diff=0.21875
  logit[59273]: CUDA=-3.890625, PyTorch=-4.09375, abs_diff=0.203125
  logit[53862]: CUDA=-2.84375, PyTorch=-3.046875, abs_diff=0.203125
  logit[16478]: CUDA=-3.296875, PyTorch=-3.5, abs_diff=0.203125
  logit[73334]: CUDA=-2.5, PyTorch=-2.703125, abs_diff=0.203125
  logit[69359]: CUDA=-2.296875, PyTorch=-2.5, abs_diff=0.203125
  logit[75667]: CUDA=-3.171875, PyTorch=-3.375, abs_diff=0.203125
  logit[81567]: CUDA=-2.71875, PyTorch=-2.921875, abs_diff=0.203125
  logit[72992]: CUDA=-3.203125, PyTorch=-3.40625, abs_diff=0.203125
  logit[60006]: CUDA=-2.203125, PyTorch=-2.40625, abs_diff=0.203125
step 7: CUDA token=5424, PyTorch token=5424
  logits: 3396/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[1362]: CUDA=-2.53125, PyTorch=-2.28125, abs_diff=0.25
  logit[72611]: CUDA=3.234375, PyTorch=3.46875, abs_diff=0.234375
  logit[47118]: CUDA=-4.1875, PyTorch=-3.953125, abs_diff=0.234375
  logit[70966]: CUDA=0.65234375, PyTorch=0.88671875, abs_diff=0.234375
  logit[123887]: CUDA=-1.2421875, PyTorch=-1.015625, abs_diff=0.2265625
  logit[119299]: CUDA=-0.39648438, PyTorch=-0.17089844, abs_diff=0.22558594
  logit[109145]: CUDA=-0.5078125, PyTorch=-0.28710938, abs_diff=0.22070312
  logit[65610]: CUDA=3.28125, PyTorch=3.5, abs_diff=0.21875
  logit[60333]: CUDA=3.25, PyTorch=3.46875, abs_diff=0.21875
  logit[7574]: CUDA=1.0859375, PyTorch=1.3046875, abs_diff=0.21875
step 8: CUDA token=25, PyTorch token=25
  logits: 1183/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.23046875
  logit[57314]: CUDA=-0.86328125, PyTorch=-1.09375, abs_diff=0.23046875
  logit[23918]: CUDA=1.9453125, PyTorch=1.7421875, abs_diff=0.203125
  logit[4647]: CUDA=2.9375, PyTorch=2.734375, abs_diff=0.203125
  logit[69074]: CUDA=-1.5390625, PyTorch=-1.734375, abs_diff=0.1953125
  logit[53392]: CUDA=2.59375, PyTorch=2.40625, abs_diff=0.1875
  logit[5670]: CUDA=2.046875, PyTorch=1.859375, abs_diff=0.1875
  logit[36921]: CUDA=4.125, PyTorch=3.9375, abs_diff=0.1875
  logit[95291]: CUDA=-0.098632812, PyTorch=-0.28515625, abs_diff=0.18652344
  logit[93161]: CUDA=-0.5, PyTorch=-0.68359375, abs_diff=0.18359375
  logit[45764]: CUDA=0.7421875, PyTorch=0.55859375, abs_diff=0.18359375
step 9: CUDA token=220, PyTorch token=220
  logits: 1119/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.203125
  logit[49483]: CUDA=-2.421875, PyTorch=-2.625, abs_diff=0.203125
  logit[38692]: CUDA=-1.453125, PyTorch=-1.65625, abs_diff=0.203125
  logit[67268]: CUDA=1.96875, PyTorch=1.765625, abs_diff=0.203125
  logit[72589]: CUDA=0.34375, PyTorch=0.140625, abs_diff=0.203125
  logit[97823]: CUDA=0.37304688, PyTorch=0.17578125, abs_diff=0.19726562
  logit[74404]: CUDA=-1.8359375, PyTorch=-2.03125, abs_diff=0.1953125
  logit[29441]: CUDA=-4.15625, PyTorch=-4.34375, abs_diff=0.1875
  logit[54981]: CUDA=-1.078125, PyTorch=-1.265625, abs_diff=0.1875
  logit[18997]: CUDA=-2.15625, PyTorch=-1.96875, abs_diff=0.1875
  logit[8981]: CUDA=2.53125, PyTorch=2.34375, abs_diff=0.1875
step 10: CUDA token=16, PyTorch token=16
  logits: 2816/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.2734375
  logit[85954]: CUDA=-2.140625, PyTorch=-1.8671875, abs_diff=0.2734375
  logit[118549]: CUDA=2.765625, PyTorch=3.015625, abs_diff=0.25
  logit[76273]: CUDA=-2.421875, PyTorch=-2.1875, abs_diff=0.234375
  logit[42572]: CUDA=0.71875, PyTorch=0.953125, abs_diff=0.234375
  logit[67558]: CUDA=-1.5, PyTorch=-1.2734375, abs_diff=0.2265625
  logit[68595]: CUDA=1.6875, PyTorch=1.9140625, abs_diff=0.2265625
  logit[35608]: CUDA=0.19628906, PyTorch=0.41992188, abs_diff=0.22363281
  logit[44708]: CUDA=-5, PyTorch=-4.78125, abs_diff=0.21875
  logit[951]: CUDA=4.5, PyTorch=4.71875, abs_diff=0.21875
  logit[7615]: CUDA=4.09375, PyTorch=4.3125, abs_diff=0.21875
step 11: CUDA token=8, PyTorch token=8
  logits: 1541/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.21875
  logit[29193]: CUDA=1.3046875, PyTorch=1.0859375, abs_diff=0.21875
  logit[10994]: CUDA=4.09375, PyTorch=3.875, abs_diff=0.21875
  logit[127503]: CUDA=2.390625, PyTorch=2.171875, abs_diff=0.21875
  logit[67691]: CUDA=-0.50390625, PyTorch=-0.72265625, abs_diff=0.21875
  logit[53795]: CUDA=0.6171875, PyTorch=0.40820312, abs_diff=0.20898438
  logit[50533]: CUDA=-0.58203125, PyTorch=-0.78515625, abs_diff=0.203125
  logit[9734]: CUDA=3.6875, PyTorch=3.484375, abs_diff=0.203125
  logit[1402]: CUDA=3.953125, PyTorch=3.75, abs_diff=0.203125
  logit[15986]: CUDA=2.859375, PyTorch=2.65625, abs_diff=0.203125
  logit[17771]: CUDA=3.03125, PyTorch=2.828125, abs_diff=0.203125
step 12: CUDA token=279, PyTorch token=279
  logits: 1434/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[86045]: CUDA=-0.8671875, PyTorch=-0.625, abs_diff=0.2421875
  logit[57192]: CUDA=3.1875, PyTorch=3.421875, abs_diff=0.234375
  logit[98355]: CUDA=2.0625, PyTorch=2.296875, abs_diff=0.234375
  logit[33266]: CUDA=0.041992188, PyTorch=-0.17871094, abs_diff=0.22070312
  logit[8232]: CUDA=3.40625, PyTorch=3.625, abs_diff=0.21875
  logit[48129]: CUDA=4.46875, PyTorch=4.6875, abs_diff=0.21875
  logit[19578]: CUDA=2.046875, PyTorch=2.265625, abs_diff=0.21875
  logit[104428]: CUDA=-5.15625, PyTorch=-5.375, abs_diff=0.21875
  logit[69805]: CUDA=-1.8203125, PyTorch=-1.6015625, abs_diff=0.21875
  logit[49211]: CUDA=1.6953125, PyTorch=1.9140625, abs_diff=0.21875
step 13: CUDA token=17330, PyTorch token=17330
  logits: 1104/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[60115]: CUDA=-5.78125, PyTorch=-5.53125, abs_diff=0.25
  logit[13722]: CUDA=-2.1875, PyTorch=-1.9375, abs_diff=0.25
  logit[37220]: CUDA=-2.671875, PyTorch=-2.4375, abs_diff=0.234375
  logit[62]: CUDA=5.4375, PyTorch=5.65625, abs_diff=0.21875
  logit[56992]: CUDA=-1.4296875, PyTorch=-1.21875, abs_diff=0.2109375
  logit[9249]: CUDA=0.30664062, PyTorch=0.51171875, abs_diff=0.20507812
  logit[16347]: CUDA=-3.359375, PyTorch=-3.15625, abs_diff=0.203125
  logit[74088]: CUDA=-0.54296875, PyTorch=-0.33984375, abs_diff=0.203125
  logit[58816]: CUDA=-1.75, PyTorch=-1.546875, abs_diff=0.203125
  logit[36233]: CUDA=-2.359375, PyTorch=-2.15625, abs_diff=0.203125
step 14: CUDA token=594, PyTorch token=594
  logits: 4941/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.31445312
  logit[96082]: CUDA=-0.61328125, PyTorch=-0.29882812, abs_diff=0.31445312
  logit[24968]: CUDA=4.84375, PyTorch=4.53125, abs_diff=0.3125
  logit[52892]: CUDA=2.046875, PyTorch=2.34375, abs_diff=0.296875
  logit[18252]: CUDA=5.09375, PyTorch=5.375, abs_diff=0.28125
  logit[54637]: CUDA=-2.671875, PyTorch=-2.953125, abs_diff=0.28125
  logit[52627]: CUDA=4.75, PyTorch=5.03125, abs_diff=0.28125
  logit[30752]: CUDA=1.34375, PyTorch=1.625, abs_diff=0.28125
  logit[59488]: CUDA=6.0625, PyTorch=6.34375, abs_diff=0.28125
  logit[69445]: CUDA=2.703125, PyTorch=2.984375, abs_diff=0.28125
  logit[739]: CUDA=7, PyTorch=7.28125, abs_diff=0.28125
step 15: CUDA token=23249, PyTorch token=23249
  logits: 5145/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.359375
  logit[87496]: CUDA=-3.671875, PyTorch=-3.3125, abs_diff=0.359375
  logit[43561]: CUDA=-5.0625, PyTorch=-4.71875, abs_diff=0.34375
  logit[39788]: CUDA=-8.75, PyTorch=-8.4375, abs_diff=0.3125
  logit[53354]: CUDA=-4.59375, PyTorch=-4.28125, abs_diff=0.3125
  logit[21812]: CUDA=-8.9375, PyTorch=-8.625, abs_diff=0.3125
  logit[58388]: CUDA=-7.59375, PyTorch=-7.28125, abs_diff=0.3125
  logit[68350]: CUDA=-5.40625, PyTorch=-5.09375, abs_diff=0.3125
  logit[54756]: CUDA=-4.625, PyTorch=-4.3125, abs_diff=0.3125
  logit[70312]: CUDA=-10.125, PyTorch=-9.8125, abs_diff=0.3125
  logit[57699]: CUDA=-7.5, PyTorch=-7.1875, abs_diff=0.3125
step 16: CUDA token=11, PyTorch token=11
  logits: 2849/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[10830]: CUDA=-4.34375, PyTorch=-4.03125, abs_diff=0.3125
  logit[40781]: CUDA=-5.96875, PyTorch=-5.6875, abs_diff=0.28125
  logit[15638]: CUDA=-7.3125, PyTorch=-7.0625, abs_diff=0.25
  logit[74624]: CUDA=-6.25, PyTorch=-6, abs_diff=0.25
  logit[39971]: CUDA=-7.53125, PyTorch=-7.28125, abs_diff=0.25
  logit[49077]: CUDA=-4.9375, PyTorch=-4.6875, abs_diff=0.25
  logit[76560]: CUDA=-3.984375, PyTorch=-3.734375, abs_diff=0.25
  logit[103451]: CUDA=-4.59375, PyTorch=-4.34375, abs_diff=0.25
  logit[75326]: CUDA=-5.65625, PyTorch=-5.40625, abs_diff=0.25
  logit[50705]: CUDA=-7.59375, PyTorch=-7.34375, abs_diff=0.25
step 17: CUDA token=220, PyTorch token=220
  logits: 643/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[57833]: CUDA=6.21875, PyTorch=6.46875, abs_diff=0.25
  logit[89935]: CUDA=4.1875, PyTorch=4.40625, abs_diff=0.21875
  logit[76322]: CUDA=-1.1328125, PyTorch=-0.9296875, abs_diff=0.203125
  logit[30560]: CUDA=0.375, PyTorch=0.5625, abs_diff=0.1875
  logit[92266]: CUDA=-0.55859375, PyTorch=-0.37109375, abs_diff=0.1875
  logit[36924]: CUDA=2.546875, PyTorch=2.734375, abs_diff=0.1875
  logit[70487]: CUDA=2.09375, PyTorch=2.28125, abs_diff=0.1875
  logit[144008]: CUDA=2.359375, PyTorch=2.546875, abs_diff=0.1875
  logit[18301]: CUDA=-3, PyTorch=-3.1875, abs_diff=0.1875
  logit[83679]: CUDA=-2.3125, PyTorch=-2.5, abs_diff=0.1875
step 18: CUDA token=17, PyTorch token=17
  logits: 1125/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.21875
  logit[32064]: CUDA=4.21875, PyTorch=4.4375, abs_diff=0.21875
  logit[18662]: CUDA=3.78125, PyTorch=4, abs_diff=0.21875
  logit[6629]: CUDA=4.375, PyTorch=4.59375, abs_diff=0.21875
  logit[52423]: CUDA=-1.1640625, PyTorch=-0.94921875, abs_diff=0.21484375
  logit[138212]: CUDA=-1.4609375, PyTorch=-1.25, abs_diff=0.2109375
  logit[26403]: CUDA=1.90625, PyTorch=2.109375, abs_diff=0.203125
  logit[25248]: CUDA=1.953125, PyTorch=2.15625, abs_diff=0.203125
  logit[110753]: CUDA=-0.33203125, PyTorch=-0.53125, abs_diff=0.19921875
  logit[30618]: CUDA=1.7734375, PyTorch=1.96875, abs_diff=0.1953125
  logit[43601]: CUDA=0.86328125, PyTorch=1.0546875, abs_diff=0.19140625
step 19: CUDA token=8, PyTorch token=8
  logits: 2068/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.234375
  logit[135588]: CUDA=-2.75, PyTorch=-2.984375, abs_diff=0.234375
  logit[45518]: CUDA=-1.8046875, PyTorch=-2.03125, abs_diff=0.2265625
  logit[70863]: CUDA=-2.15625, PyTorch=-2.375, abs_diff=0.21875
  logit[18921]: CUDA=0.70703125, PyTorch=0.4921875, abs_diff=0.21484375
  logit[3959]: CUDA=-0.67578125, PyTorch=-0.88671875, abs_diff=0.2109375
  logit[56792]: CUDA=-1.75, PyTorch=-1.953125, abs_diff=0.203125
  logit[110488]: CUDA=-3.890625, PyTorch=-4.09375, abs_diff=0.203125
  logit[37218]: CUDA=1.1953125, PyTorch=0.9921875, abs_diff=0.203125
  logit[72239]: CUDA=2.703125, PyTorch=2.5, abs_diff=0.203125
  logit[62265]: CUDA=0.96484375, PyTorch=0.765625, abs_diff=0.19921875
step 20: CUDA token=279, PyTorch token=279
  logits: 6482/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.34375
  logit[59499]: CUDA=-3.5, PyTorch=-3.84375, abs_diff=0.34375
  logit[126463]: CUDA=1.578125, PyTorch=1.2421875, abs_diff=0.3359375
  logit[90563]: CUDA=-0.58984375, PyTorch=-0.9140625, abs_diff=0.32421875
  logit[80485]: CUDA=0.24414062, PyTorch=-0.073242188, abs_diff=0.31738281
  logit[49174]: CUDA=-4.0625, PyTorch=-3.75, abs_diff=0.3125
  logit[40614]: CUDA=-8.8125, PyTorch=-8.5, abs_diff=0.3125
  logit[13217]: CUDA=-1.03125, PyTorch=-0.71875, abs_diff=0.3125
  logit[62922]: CUDA=-8.375, PyTorch=-8.0625, abs_diff=0.3125
  logit[14806]: CUDA=5.03125, PyTorch=4.71875, abs_diff=0.3125
  logit[17120]: CUDA=-6.5625, PyTorch=-6.25, abs_diff=0.3125
step 21: CUDA token=9237, PyTorch token=9237
  logits: 4565/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.359375
  logit[72592]: CUDA=-2.890625, PyTorch=-2.53125, abs_diff=0.359375
  logit[137126]: CUDA=-7.28125, PyTorch=-6.9375, abs_diff=0.34375
  logit[3399]: CUDA=0.15917969, PyTorch=0.48828125, abs_diff=0.32910156
  logit[125013]: CUDA=-4.125, PyTorch=-3.796875, abs_diff=0.328125
  logit[19017]: CUDA=-3.375, PyTorch=-3.0625, abs_diff=0.3125
  logit[70497]: CUDA=-8.375, PyTorch=-8.0625, abs_diff=0.3125
  logit[28960]: CUDA=-6.5625, PyTorch=-6.25, abs_diff=0.3125
  logit[116847]: CUDA=-8.625, PyTorch=-8.3125, abs_diff=0.3125
  logit[55718]: CUDA=-8.6875, PyTorch=-8.375, abs_diff=0.3125
  logit[119026]: CUDA=-6.78125, PyTorch=-6.46875, abs_diff=0.3125
step 22: CUDA token=594, PyTorch token=594
  logits: 3723/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.34375
  logit[29338]: CUDA=7.5, PyTorch=7.84375, abs_diff=0.34375
  logit[56760]: CUDA=-10.0625, PyTorch=-9.75, abs_diff=0.3125
  logit[138745]: CUDA=-9.1875, PyTorch=-8.875, abs_diff=0.3125
  logit[104397]: CUDA=-2.15625, PyTorch=-1.8515625, abs_diff=0.3046875
  logit[67632]: CUDA=-2.4375, PyTorch=-2.734375, abs_diff=0.296875
  logit[89551]: CUDA=-3.625, PyTorch=-3.328125, abs_diff=0.296875
  logit[95261]: CUDA=-3.734375, PyTorch=-3.4375, abs_diff=0.296875
  logit[102808]: CUDA=-1.8125, PyTorch=-1.5234375, abs_diff=0.2890625
  logit[72799]: CUDA=-4.40625, PyTorch=-4.125, abs_diff=0.28125
  logit[10475]: CUDA=5.15625, PyTorch=4.875, abs_diff=0.28125
step 23: CUDA token=12695, PyTorch token=12695
  logits: 5900/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.375
  logit[5216]: CUDA=-12.9375, PyTorch=-12.5625, abs_diff=0.375
  logit[14507]: CUDA=6.625, PyTorch=7, abs_diff=0.375
  logit[101844]: CUDA=-6.5625, PyTorch=-6.1875, abs_diff=0.375
  logit[135890]: CUDA=-8.25, PyTorch=-7.90625, abs_diff=0.34375
  logit[18523]: CUDA=-7.9375, PyTorch=-7.59375, abs_diff=0.34375
  logit[71155]: CUDA=-7, PyTorch=-6.65625, abs_diff=0.34375
  logit[35644]: CUDA=-8.3125, PyTorch=-7.96875, abs_diff=0.34375
  logit[98978]: CUDA=-3.78125, PyTorch=-3.453125, abs_diff=0.328125
  logit[80467]: CUDA=-0.859375, PyTorch=-0.54296875, abs_diff=0.31640625
  logit[26443]: CUDA=6, PyTorch=6.3125, abs_diff=0.3125
step 24: CUDA token=11, PyTorch token=11
  logits: 2946/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.296875
  logit[28842]: CUDA=-3.921875, PyTorch=-4.21875, abs_diff=0.296875
  logit[96018]: CUDA=-1.78125, PyTorch=-2.0625, abs_diff=0.28125
  logit[20654]: CUDA=2.84375, PyTorch=2.5625, abs_diff=0.28125
  logit[60894]: CUDA=-7.78125, PyTorch=-7.5, abs_diff=0.28125
  logit[68375]: CUDA=1.9609375, PyTorch=1.6875, abs_diff=0.2734375
  logit[78055]: CUDA=-2.484375, PyTorch=-2.75, abs_diff=0.265625
  logit[73820]: CUDA=3.96875, PyTorch=3.703125, abs_diff=0.265625
  logit[37842]: CUDA=2.671875, PyTorch=2.40625, abs_diff=0.265625
  logit[89945]: CUDA=-0.56640625, PyTorch=-0.82421875, abs_diff=0.2578125
  logit[58519]: CUDA=-0.040039062, PyTorch=-0.29296875, abs_diff=0.25292969
step 25: CUDA token=220, PyTorch token=220
  logits: 2080/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[14383]: CUDA=1.296875, PyTorch=1.046875, abs_diff=0.25
  logit[33151]: CUDA=-0.42773438, PyTorch=-0.66015625, abs_diff=0.23242188
  logit[86355]: CUDA=-0.34375, PyTorch=-0.57421875, abs_diff=0.23046875
  logit[99040]: CUDA=-1.9140625, PyTorch=-2.140625, abs_diff=0.2265625
  logit[110224]: CUDA=-1.03125, PyTorch=-1.2578125, abs_diff=0.2265625
  logit[7393]: CUDA=-0.75, PyTorch=-0.97265625, abs_diff=0.22265625
  logit[86660]: CUDA=-4.375, PyTorch=-4.59375, abs_diff=0.21875
  logit[78023]: CUDA=3.265625, PyTorch=3.046875, abs_diff=0.21875
  logit[41221]: CUDA=4.875, PyTorch=4.65625, abs_diff=0.21875
  logit[100067]: CUDA=-5.03125, PyTorch=-5.25, abs_diff=0.21875
step 26: CUDA token=18, PyTorch token=18
  logits: 14006/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.5
  logit[81693]: CUDA=-9.4375, PyTorch=-9.9375, abs_diff=0.5
  logit[55500]: CUDA=-3.71875, PyTorch=-4.125, abs_diff=0.40625
  logit[8674]: CUDA=6.9375, PyTorch=7.34375, abs_diff=0.40625
  logit[111808]: CUDA=-7.90625, PyTorch=-8.3125, abs_diff=0.40625
  logit[102296]: CUDA=-4, PyTorch=-4.375, abs_diff=0.375
  logit[92386]: CUDA=-5.15625, PyTorch=-5.53125, abs_diff=0.375
  logit[105392]: CUDA=-4.90625, PyTorch=-5.28125, abs_diff=0.375
  logit[62041]: CUDA=-3.78125, PyTorch=-4.15625, abs_diff=0.375
  logit[107900]: CUDA=-4.9375, PyTorch=-5.3125, abs_diff=0.375
  logit[74949]: CUDA=-3.28125, PyTorch=-3.65625, abs_diff=0.375
step 27: CUDA token=8, PyTorch token=8
  logits: 9137/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.359375
  logit[126766]: CUDA=2.34375, PyTorch=2.703125, abs_diff=0.359375
  logit[15774]: CUDA=0.890625, PyTorch=1.1953125, abs_diff=0.3046875
  logit[39873]: CUDA=-0.58984375, PyTorch=-0.31054688, abs_diff=0.27929688
  logit[77105]: CUDA=-0.85546875, PyTorch=-0.58203125, abs_diff=0.2734375
  logit[105997]: CUDA=-3.421875, PyTorch=-3.15625, abs_diff=0.265625
  logit[124457]: CUDA=-1.1875, PyTorch=-0.92578125, abs_diff=0.26171875
  logit[26565]: CUDA=0.17871094, PyTorch=0.4375, abs_diff=0.25878906
  logit[33235]: CUDA=1.359375, PyTorch=1.6171875, abs_diff=0.2578125
  logit[27574]: CUDA=0.49804688, PyTorch=0.75, abs_diff=0.25195312
  logit[13888]: CUDA=-0.703125, PyTorch=-0.45117188, abs_diff=0.25195312
step 28: CUDA token=279, PyTorch token=279
  logits: 2324/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[10425]: CUDA=-8.9375, PyTorch=-8.625, abs_diff=0.3125
  logit[22457]: CUDA=-1.1953125, PyTorch=-0.91796875, abs_diff=0.27734375
  logit[49314]: CUDA=-0.084472656, PyTorch=0.18164062, abs_diff=0.26611328
  logit[104056]: CUDA=-2.140625, PyTorch=-2.40625, abs_diff=0.265625
  logit[104258]: CUDA=-1.5859375, PyTorch=-1.84375, abs_diff=0.2578125
  logit[1577]: CUDA=-4.84375, PyTorch=-4.59375, abs_diff=0.25
  logit[49640]: CUDA=-4.375, PyTorch=-4.125, abs_diff=0.25
  logit[1672]: CUDA=-4.09375, PyTorch=-3.84375, abs_diff=0.25
  logit[53919]: CUDA=-4.5, PyTorch=-4.25, abs_diff=0.25
  logit[28191]: CUDA=-6.28125, PyTorch=-6.03125, abs_diff=0.25
step 29: CUDA token=85459, PyTorch token=85459
  logits: 3292/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[8708]: CUDA=9.125, PyTorch=9.4375, abs_diff=0.3125
  logit[106262]: CUDA=-7.90625, PyTorch=-8.1875, abs_diff=0.28125
  logit[17835]: CUDA=-6, PyTorch=-6.28125, abs_diff=0.28125
  logit[125165]: CUDA=-5.5, PyTorch=-5.78125, abs_diff=0.28125
  logit[44614]: CUDA=0.80078125, PyTorch=1.0625, abs_diff=0.26171875
  logit[70069]: CUDA=1.0078125, PyTorch=1.265625, abs_diff=0.2578125
  logit[59487]: CUDA=-0.4296875, PyTorch=-0.17578125, abs_diff=0.25390625
  logit[18985]: CUDA=-4.6875, PyTorch=-4.9375, abs_diff=0.25
  logit[49916]: CUDA=-2.375, PyTorch=-2.125, abs_diff=0.25
  logit[44150]: CUDA=-6.03125, PyTorch=-5.78125, abs_diff=0.25
step 30: CUDA token=8437, PyTorch token=8437
  logits: 9547/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.453125
  logit[128432]: CUDA=-2.78125, PyTorch=-3.234375, abs_diff=0.453125
  logit[89368]: CUDA=-0.6171875, PyTorch=-1.0625, abs_diff=0.4453125
  logit[89718]: CUDA=-12.1875, PyTorch=-12.625, abs_diff=0.4375
  logit[59207]: CUDA=6, PyTorch=5.5625, abs_diff=0.4375
  logit[113930]: CUDA=-9.8125, PyTorch=-10.25, abs_diff=0.4375
  logit[92545]: CUDA=-4.6875, PyTorch=-4.25, abs_diff=0.4375
  logit[55541]: CUDA=-9.5, PyTorch=-9.9375, abs_diff=0.4375
  logit[15678]: CUDA=-3.8125, PyTorch=-3.390625, abs_diff=0.421875
  logit[141992]: CUDA=-1.40625, PyTorch=-1.8203125, abs_diff=0.4140625
  logit[114668]: CUDA=-1.1796875, PyTorch=-1.59375, abs_diff=0.4140625
step 31: CUDA token=11, PyTorch token=11
  logits: 3010/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[45464]: CUDA=-4.75, PyTorch=-4.5, abs_diff=0.25
  logit[26766]: CUDA=-3.765625, PyTorch=-3.515625, abs_diff=0.25
  logit[20725]: CUDA=-2.53125, PyTorch=-2.28125, abs_diff=0.25
  logit[88456]: CUDA=-3.765625, PyTorch=-3.515625, abs_diff=0.25
  logit[43140]: CUDA=-2.171875, PyTorch=-1.921875, abs_diff=0.25
  logit[5965]: CUDA=-2.578125, PyTorch=-2.328125, abs_diff=0.25
  logit[66446]: CUDA=-2.140625, PyTorch=-1.8984375, abs_diff=0.2421875
  logit[95500]: CUDA=-2.34375, PyTorch=-2.109375, abs_diff=0.234375
  logit[73247]: CUDA=1.875, PyTorch=2.109375, abs_diff=0.234375
  logit[61537]: CUDA=0.78125, PyTorch=1.015625, abs_diff=0.234375
step 32: CUDA token=220, PyTorch token=220
  logits: 14254/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.2890625
  logit[14506]: CUDA=-0.017578125, PyTorch=-0.30664062, abs_diff=0.2890625
  logit[66259]: CUDA=2.1875, PyTorch=1.90625, abs_diff=0.28125
  logit[62365]: CUDA=1.578125, PyTorch=1.3046875, abs_diff=0.2734375
  logit[32803]: CUDA=0.84765625, PyTorch=0.57421875, abs_diff=0.2734375
  logit[78529]: CUDA=-1.7890625, PyTorch=-2.0625, abs_diff=0.2734375
  logit[56194]: CUDA=2.828125, PyTorch=2.5625, abs_diff=0.265625
  logit[66530]: CUDA=3.859375, PyTorch=3.59375, abs_diff=0.265625
  logit[33959]: CUDA=0.31640625, PyTorch=0.051513672, abs_diff=0.26489258
  logit[120585]: CUDA=1.21875, PyTorch=0.95703125, abs_diff=0.26171875
  logit[65943]: CUDA=0.70703125, PyTorch=0.4453125, abs_diff=0.26171875
step 33: CUDA token=19, PyTorch token=19
  logits: 3135/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.28125
  logit[2850]: CUDA=-6.25, PyTorch=-6.53125, abs_diff=0.28125
  logit[32295]: CUDA=0.625, PyTorch=0.36523438, abs_diff=0.25976562
  logit[30986]: CUDA=0.6484375, PyTorch=0.390625, abs_diff=0.2578125
  logit[22705]: CUDA=-4.875, PyTorch=-5.125, abs_diff=0.25
  logit[25693]: CUDA=-1.359375, PyTorch=-1.609375, abs_diff=0.25
  logit[37025]: CUDA=-5.09375, PyTorch=-5.34375, abs_diff=0.25
  logit[41307]: CUDA=-6.375, PyTorch=-6.625, abs_diff=0.25
  logit[18733]: CUDA=-5.15625, PyTorch=-5.40625, abs_diff=0.25
  logit[48313]: CUDA=-6.96875, PyTorch=-7.21875, abs_diff=0.25
  logit[7556]: CUDA=-0.25390625, PyTorch=-0.50390625, abs_diff=0.25
step 34: CUDA token=8, PyTorch token=8
  logits: 2879/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.28125
  logit[84333]: CUDA=-5.1875, PyTorch=-5.46875, abs_diff=0.28125
  logit[75192]: CUDA=-5.90625, PyTorch=-6.1875, abs_diff=0.28125
  logit[65615]: CUDA=-1.46875, PyTorch=-1.734375, abs_diff=0.265625
  logit[49038]: CUDA=-3.765625, PyTorch=-4.03125, abs_diff=0.265625
  logit[17044]: CUDA=-1.4453125, PyTorch=-1.1875, abs_diff=0.2578125
  logit[45499]: CUDA=-6.375, PyTorch=-6.625, abs_diff=0.25
  logit[75453]: CUDA=-3.9375, PyTorch=-4.1875, abs_diff=0.25
  logit[8944]: CUDA=-2.765625, PyTorch=-2.515625, abs_diff=0.25
  logit[6322]: CUDA=-6.59375, PyTorch=-6.84375, abs_diff=0.25
  logit[82776]: CUDA=-0.96484375, PyTorch=-1.2109375, abs_diff=0.24609375
step 35: CUDA token=279, PyTorch token=279
  logits: 3056/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.28125
  logit[95144]: CUDA=-5.625, PyTorch=-5.34375, abs_diff=0.28125
  logit[11388]: CUDA=-2.078125, PyTorch=-1.828125, abs_diff=0.25
  logit[88386]: CUDA=-2.15625, PyTorch=-1.9140625, abs_diff=0.2421875
  logit[33354]: CUDA=1.9921875, PyTorch=2.234375, abs_diff=0.2421875
  logit[1000]: CUDA=-3.78125, PyTorch=-3.546875, abs_diff=0.234375
  logit[76069]: CUDA=-1.203125, PyTorch=-0.96875, abs_diff=0.234375
  logit[99029]: CUDA=-2.34375, PyTorch=-2.109375, abs_diff=0.234375
  logit[54292]: CUDA=-2.640625, PyTorch=-2.40625, abs_diff=0.234375
  logit[117559]: CUDA=-2.484375, PyTorch=-2.25, abs_diff=0.234375
  logit[90348]: CUDA=-2.671875, PyTorch=-2.4375, abs_diff=0.234375
step 36: CUDA token=259, PyTorch token=259
  logits: 3322/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.265625
  logit[92424]: CUDA=-3.625, PyTorch=-3.359375, abs_diff=0.265625
  logit[69184]: CUDA=-7.375, PyTorch=-7.125, abs_diff=0.25
  logit[64422]: CUDA=-6.625, PyTorch=-6.875, abs_diff=0.25
  logit[27306]: CUDA=-7.1875, PyTorch=-6.9375, abs_diff=0.25
  logit[52981]: CUDA=-4.4375, PyTorch=-4.1875, abs_diff=0.25
  logit[90348]: CUDA=-3.765625, PyTorch=-3.515625, abs_diff=0.25
  logit[17473]: CUDA=5.75, PyTorch=6, abs_diff=0.25
  logit[97599]: CUDA=-5.4375, PyTorch=-5.1875, abs_diff=0.25
  logit[81554]: CUDA=-4.875, PyTorch=-4.625, abs_diff=0.25
  logit[57896]: CUDA=-4.96875, PyTorch=-4.71875, abs_diff=0.25
step 37: CUDA token=3341, PyTorch token=3341
  logits: 6832/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.33203125
  logit[53038]: CUDA=-0.7421875, PyTorch=-0.41015625, abs_diff=0.33203125
  logit[148671]: CUDA=-4.0625, PyTorch=-3.734375, abs_diff=0.328125
  logit[83951]: CUDA=-0.90625, PyTorch=-0.58984375, abs_diff=0.31640625
  logit[10960]: CUDA=-2.828125, PyTorch=-2.515625, abs_diff=0.3125
  logit[53830]: CUDA=-7.09375, PyTorch=-6.78125, abs_diff=0.3125
  logit[106967]: CUDA=-10.6875, PyTorch=-10.375, abs_diff=0.3125
  logit[137574]: CUDA=-6.40625, PyTorch=-6.09375, abs_diff=0.3125
  logit[72831]: CUDA=-3.875, PyTorch=-3.5625, abs_diff=0.3125
  logit[42554]: CUDA=-4.6875, PyTorch=-4.375, abs_diff=0.3125
  logit[55866]: CUDA=-0.47460938, PyTorch=-0.17382812, abs_diff=0.30078125
step 38: CUDA token=11, PyTorch token=11
  logits: 18269/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.375
  logit[43019]: CUDA=-11.8125, PyTorch=-11.4375, abs_diff=0.375
  logit[6496]: CUDA=-9.375, PyTorch=-9, abs_diff=0.375
  logit[8018]: CUDA=-10.5625, PyTorch=-10.1875, abs_diff=0.375
  logit[24224]: CUDA=-9, PyTorch=-8.625, abs_diff=0.375
  logit[12597]: CUDA=-7.59375, PyTorch=-7.21875, abs_diff=0.375
  logit[48466]: CUDA=-7.65625, PyTorch=-7.28125, abs_diff=0.375
  logit[96557]: CUDA=-6.375, PyTorch=-6, abs_diff=0.375
  logit[76933]: CUDA=-9.125, PyTorch=-8.75, abs_diff=0.375
  logit[11915]: CUDA=-3.46875, PyTorch=-3.109375, abs_diff=0.359375
  logit[23121]: CUDA=-4.125, PyTorch=-3.765625, abs_diff=0.359375
step 39: CUDA token=220, PyTorch token=220
  logits: 4790/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.5
  logit[862]: CUDA=14.375, PyTorch=13.875, abs_diff=0.5
  logit[104056]: CUDA=2.390625, PyTorch=1.9296875, abs_diff=0.4609375
  logit[49728]: CUDA=7.1875, PyTorch=6.75, abs_diff=0.4375
  logit[10964]: CUDA=8.9375, PyTorch=8.5, abs_diff=0.4375
  logit[34797]: CUDA=6.15625, PyTorch=5.78125, abs_diff=0.375
  logit[99650]: CUDA=0.00039672852, PyTorch=-0.35742188, abs_diff=0.3578186
  logit[83604]: CUDA=4.03125, PyTorch=3.6875, abs_diff=0.34375
  logit[110374]: CUDA=-0.26757812, PyTorch=-0.6015625, abs_diff=0.33398438
  logit[104258]: CUDA=2.0625, PyTorch=1.7421875, abs_diff=0.3203125
  logit[107469]: CUDA=0.055175781, PyTorch=-0.2578125, abs_diff=0.31298828
step 40: CUDA token=20, PyTorch token=20
  logits: 18676/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.390625
  logit[140761]: CUDA=-3.671875, PyTorch=-4.0625, abs_diff=0.390625
  logit[27320]: CUDA=-2.203125, PyTorch=-2.578125, abs_diff=0.375
  logit[83539]: CUDA=-1.6640625, PyTorch=-2.03125, abs_diff=0.3671875
  logit[52645]: CUDA=-1.65625, PyTorch=-2.015625, abs_diff=0.359375
  logit[98984]: CUDA=-5.28125, PyTorch=-5.625, abs_diff=0.34375
  logit[94338]: CUDA=-5.03125, PyTorch=-5.375, abs_diff=0.34375
  logit[85093]: CUDA=-2.484375, PyTorch=-2.828125, abs_diff=0.34375
  logit[7583]: CUDA=-4.75, PyTorch=-5.09375, abs_diff=0.34375
  logit[89748]: CUDA=-0.8359375, PyTorch=-1.171875, abs_diff=0.3359375
  logit[71132]: CUDA=-3.984375, PyTorch=-4.3125, abs_diff=0.328125
step 41: CUDA token=8, PyTorch token=8
  logits: 2058/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[8687]: CUDA=6.46875, PyTorch=6.71875, abs_diff=0.25
  logit[76266]: CUDA=5.125, PyTorch=5.375, abs_diff=0.25
  logit[87272]: CUDA=-0.48242188, PyTorch=-0.26171875, abs_diff=0.22070312
  logit[93420]: CUDA=-3.09375, PyTorch=-2.875, abs_diff=0.21875
  logit[24736]: CUDA=1.75, PyTorch=1.96875, abs_diff=0.21875
  logit[95777]: CUDA=5.09375, PyTorch=5.3125, abs_diff=0.21875
  logit[23882]: CUDA=4.71875, PyTorch=4.9375, abs_diff=0.21875
  logit[60895]: CUDA=3.75, PyTorch=3.96875, abs_diff=0.21875
  logit[60191]: CUDA=2.78125, PyTorch=3, abs_diff=0.21875
  logit[111660]: CUDA=-0.5078125, PyTorch=-0.29296875, abs_diff=0.21484375
step 42: CUDA token=279, PyTorch token=279
  logits: 744/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[70739]: CUDA=-5.125, PyTorch=-5.375, abs_diff=0.25
  logit[122021]: CUDA=-1.625, PyTorch=-1.8359375, abs_diff=0.2109375
  logit[85338]: CUDA=-1.4765625, PyTorch=-1.6796875, abs_diff=0.203125
  logit[59675]: CUDA=-1.9453125, PyTorch=-2.140625, abs_diff=0.1953125
  logit[56595]: CUDA=-1.71875, PyTorch=-1.9140625, abs_diff=0.1953125
  logit[15469]: CUDA=-0.56640625, PyTorch=-0.375, abs_diff=0.19140625
  logit[21913]: CUDA=-3.265625, PyTorch=-3.453125, abs_diff=0.1875
  logit[22457]: CUDA=0.875, PyTorch=1.0625, abs_diff=0.1875
  logit[2679]: CUDA=3.84375, PyTorch=4.03125, abs_diff=0.1875
  logit[40544]: CUDA=-3.796875, PyTorch=-3.984375, abs_diff=0.1875
step 43: CUDA token=17330, PyTorch token=17330
  logits: 6845/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.28125
  logit[90765]: CUDA=-4.40625, PyTorch=-4.125, abs_diff=0.28125
  logit[86016]: CUDA=-2.671875, PyTorch=-2.390625, abs_diff=0.28125
  logit[80887]: CUDA=0.36132812, PyTorch=0.625, abs_diff=0.26367188
  logit[34879]: CUDA=-0.7421875, PyTorch=-0.48046875, abs_diff=0.26171875
  logit[79664]: CUDA=-5.28125, PyTorch=-5.03125, abs_diff=0.25
  logit[49174]: CUDA=-5.71875, PyTorch=-5.46875, abs_diff=0.25
  logit[48643]: CUDA=-4.78125, PyTorch=-4.53125, abs_diff=0.25
  logit[47560]: CUDA=-5.1875, PyTorch=-4.9375, abs_diff=0.25
  logit[46327]: CUDA=-3.3125, PyTorch=-3.0625, abs_diff=0.25
  logit[81501]: CUDA=-5.375, PyTorch=-5.125, abs_diff=0.25
step 44: CUDA token=594, PyTorch token=594
  logits: 4117/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.375
  logit[16062]: CUDA=10.8125, PyTorch=10.4375, abs_diff=0.375
  logit[33159]: CUDA=-5.75, PyTorch=-5.40625, abs_diff=0.34375
  logit[43069]: CUDA=6.0625, PyTorch=5.75, abs_diff=0.3125
  logit[108692]: CUDA=-3.109375, PyTorch=-3.40625, abs_diff=0.296875
  logit[58923]: CUDA=2.453125, PyTorch=2.15625, abs_diff=0.296875
  logit[35020]: CUDA=-0.45507812, PyTorch=-0.75, abs_diff=0.29492188
  logit[40382]: CUDA=7.09375, PyTorch=6.8125, abs_diff=0.28125
  logit[42677]: CUDA=7.1875, PyTorch=6.90625, abs_diff=0.28125
  logit[18545]: CUDA=7.6875, PyTorch=7.40625, abs_diff=0.28125
  logit[16009]: CUDA=7.59375, PyTorch=7.3125, abs_diff=0.28125
step 45: CUDA token=2309, PyTorch token=2309
  logits: 5229/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.4375
  logit[81112]: CUDA=-9.375, PyTorch=-8.9375, abs_diff=0.4375
  logit[46959]: CUDA=-5.9375, PyTorch=-5.5625, abs_diff=0.375
  logit[17368]: CUDA=-5.25, PyTorch=-4.90625, abs_diff=0.34375
  logit[28941]: CUDA=-8.1875, PyTorch=-7.84375, abs_diff=0.34375
  logit[11202]: CUDA=0.42382812, PyTorch=0.75390625, abs_diff=0.33007812
  logit[88071]: CUDA=-2.765625, PyTorch=-2.4375, abs_diff=0.328125
  logit[289]: CUDA=9.125, PyTorch=8.8125, abs_diff=0.3125
  logit[17025]: CUDA=-10.75, PyTorch=-10.4375, abs_diff=0.3125
  logit[19991]: CUDA=-10.5, PyTorch=-10.1875, abs_diff=0.3125
  logit[13396]: CUDA=-10.375, PyTorch=-10.0625, abs_diff=0.3125
step 46: CUDA token=8674, PyTorch token=8674
  logits: 1488/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.296875
  logit[13885]: CUDA=1.90625, PyTorch=2.203125, abs_diff=0.296875
  logit[118000]: CUDA=0.051513672, PyTorch=-0.23046875, abs_diff=0.28198242
  logit[92672]: CUDA=-3.125, PyTorch=-2.84375, abs_diff=0.28125
  logit[87690]: CUDA=-4.125, PyTorch=-4.375, abs_diff=0.25
  logit[4650]: CUDA=4.84375, PyTorch=5.09375, abs_diff=0.25
  logit[17257]: CUDA=-5.6875, PyTorch=-5.4375, abs_diff=0.25
  logit[59057]: CUDA=3.796875, PyTorch=4.03125, abs_diff=0.234375
  logit[58441]: CUDA=3.421875, PyTorch=3.65625, abs_diff=0.234375
  logit[9362]: CUDA=5.6875, PyTorch=5.90625, abs_diff=0.21875
  logit[13403]: CUDA=5.625, PyTorch=5.84375, abs_diff=0.21875
step 47: CUDA token=311, PyTorch token=311
  logits: 1466/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[4342]: CUDA=-4.28125, PyTorch=-4.03125, abs_diff=0.25
  logit[8769]: CUDA=0.796875, PyTorch=1.03125, abs_diff=0.234375
  logit[97369]: CUDA=-1.7578125, PyTorch=-1.53125, abs_diff=0.2265625
  logit[28941]: CUDA=-1.359375, PyTorch=-1.1328125, abs_diff=0.2265625
  logit[39322]: CUDA=-5.40625, PyTorch=-5.1875, abs_diff=0.21875
  logit[7875]: CUDA=-5.75, PyTorch=-5.53125, abs_diff=0.21875
  logit[84733]: CUDA=-4.71875, PyTorch=-4.5, abs_diff=0.21875
  logit[117543]: CUDA=-2.15625, PyTorch=-2.375, abs_diff=0.21875
  logit[38782]: CUDA=-5.28125, PyTorch=-5.0625, abs_diff=0.21875
  logit[21673]: CUDA=-6.0625, PyTorch=-5.84375, abs_diff=0.21875
step 48: CUDA token=9237, PyTorch token=9237
  logits: 3582/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.296875
  logit[92063]: CUDA=2.46875, PyTorch=2.765625, abs_diff=0.296875
  logit[20579]: CUDA=-1.7578125, PyTorch=-1.46875, abs_diff=0.2890625
  logit[90506]: CUDA=-2.40625, PyTorch=-2.6875, abs_diff=0.28125
  logit[15043]: CUDA=0.45703125, PyTorch=0.73828125, abs_diff=0.28125
  logit[43361]: CUDA=2.625, PyTorch=2.90625, abs_diff=0.28125
  logit[50356]: CUDA=-6.125, PyTorch=-6.40625, abs_diff=0.28125
  logit[46265]: CUDA=-5.96875, PyTorch=-5.6875, abs_diff=0.28125
  logit[65129]: CUDA=1.703125, PyTorch=1.9765625, abs_diff=0.2734375
  logit[86875]: CUDA=-1.8671875, PyTorch=-1.609375, abs_diff=0.2578125
  logit[32530]: CUDA=-1.3515625, PyTorch=-1.09375, abs_diff=0.2578125
step 49: CUDA token=11, PyTorch token=11
  logits: 2622/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.2890625
  logit[67459]: CUDA=-2.1875, PyTorch=-1.8984375, abs_diff=0.2890625
  logit[60774]: CUDA=0.53125, PyTorch=0.796875, abs_diff=0.265625
  logit[131679]: CUDA=-2.375, PyTorch=-2.109375, abs_diff=0.265625
  logit[74099]: CUDA=3.15625, PyTorch=3.421875, abs_diff=0.265625
  logit[111323]: CUDA=-1.828125, PyTorch=-1.5703125, abs_diff=0.2578125
  logit[73419]: CUDA=-0.016235352, PyTorch=0.23925781, abs_diff=0.25549316
  logit[16495]: CUDA=-1.1875, PyTorch=-0.93359375, abs_diff=0.25390625
  logit[78147]: CUDA=-2.796875, PyTorch=-2.546875, abs_diff=0.25
  logit[97369]: CUDA=-6.5, PyTorch=-6.25, abs_diff=0.25
  logit[61174]: CUDA=3.046875, PyTorch=3.296875, abs_diff=0.25
step 50: CUDA token=220, PyTorch token=220
  logits: 16207/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[73421]: CUDA=4.46875, PyTorch=4.15625, abs_diff=0.3125
  logit[49598]: CUDA=6.875, PyTorch=6.59375, abs_diff=0.28125
  logit[24284]: CUDA=6.46875, PyTorch=6.1875, abs_diff=0.28125
  logit[21391]: CUDA=7.625, PyTorch=7.34375, abs_diff=0.28125
  logit[31334]: CUDA=5.90625, PyTorch=5.625, abs_diff=0.28125
  logit[37103]: CUDA=4.78125, PyTorch=4.5, abs_diff=0.28125
  logit[59752]: CUDA=4.75, PyTorch=4.46875, abs_diff=0.28125
  logit[12104]: CUDA=2.234375, PyTorch=1.953125, abs_diff=0.28125
  logit[65471]: CUDA=5.25, PyTorch=4.96875, abs_diff=0.28125
  logit[41489]: CUDA=6.84375, PyTorch=6.5625, abs_diff=0.28125
step 51: CUDA token=21, PyTorch token=21
  logits: 942/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.21875
  logit[28239]: CUDA=-3.96875, PyTorch=-4.1875, abs_diff=0.21875
  logit[44592]: CUDA=4.0625, PyTorch=3.859375, abs_diff=0.203125
  logit[44146]: CUDA=3.34375, PyTorch=3.140625, abs_diff=0.203125
  logit[88218]: CUDA=-0.25390625, PyTorch=-0.45507812, abs_diff=0.20117188
  logit[12533]: CUDA=-0.48242188, PyTorch=-0.68359375, abs_diff=0.20117188
  logit[67395]: CUDA=-0.42578125, PyTorch=-0.625, abs_diff=0.19921875
  logit[105293]: CUDA=-0.78515625, PyTorch=-0.9765625, abs_diff=0.19140625
  logit[38640]: CUDA=-4.03125, PyTorch=-4.21875, abs_diff=0.1875
  logit[41505]: CUDA=-3.140625, PyTorch=-3.328125, abs_diff=0.1875
  logit[88682]: CUDA=-2.21875, PyTorch=-2.40625, abs_diff=0.1875
step 52: CUDA token=8, PyTorch token=8
  logits: 2009/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[76055]: CUDA=1.984375, PyTorch=1.734375, abs_diff=0.25
  logit[57708]: CUDA=3.9375, PyTorch=3.6875, abs_diff=0.25
  logit[19705]: CUDA=1.09375, PyTorch=0.84765625, abs_diff=0.24609375
  logit[82423]: CUDA=0.92578125, PyTorch=0.69921875, abs_diff=0.2265625
  logit[38801]: CUDA=0.11425781, PyTorch=-0.10693359, abs_diff=0.22119141
  logit[59587]: CUDA=4.15625, PyTorch=3.9375, abs_diff=0.21875
  logit[41690]: CUDA=2.53125, PyTorch=2.3125, abs_diff=0.21875
  logit[3943]: CUDA=4.71875, PyTorch=4.5, abs_diff=0.21875
  logit[79284]: CUDA=4.65625, PyTorch=4.4375, abs_diff=0.21875
  logit[58035]: CUDA=3.65625, PyTorch=3.4375, abs_diff=0.21875
step 53: CUDA token=279, PyTorch token=279
  logits: 2536/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.28125
  logit[37669]: CUDA=3.96875, PyTorch=4.25, abs_diff=0.28125
  logit[80336]: CUDA=1.7109375, PyTorch=1.9765625, abs_diff=0.265625
  logit[15311]: CUDA=3.796875, PyTorch=4.0625, abs_diff=0.265625
  logit[23261]: CUDA=2.734375, PyTorch=2.984375, abs_diff=0.25
  logit[27385]: CUDA=3.25, PyTorch=3.5, abs_diff=0.25
  logit[27597]: CUDA=6.21875, PyTorch=6.46875, abs_diff=0.25
  logit[36566]: CUDA=4.96875, PyTorch=5.21875, abs_diff=0.25
  logit[22870]: CUDA=5.90625, PyTorch=6.15625, abs_diff=0.25
  logit[20844]: CUDA=6.59375, PyTorch=6.84375, abs_diff=0.25
  logit[19818]: CUDA=6.21875, PyTorch=6.46875, abs_diff=0.25
step 54: CUDA token=6239, PyTorch token=6239
  logits: 2368/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[50994]: CUDA=-6.09375, PyTorch=-6.34375, abs_diff=0.25
  logit[88465]: CUDA=-2.125, PyTorch=-2.359375, abs_diff=0.234375
  logit[27719]: CUDA=-2.34375, PyTorch=-2.578125, abs_diff=0.234375
  logit[61354]: CUDA=0.828125, PyTorch=0.59375, abs_diff=0.234375
  logit[30961]: CUDA=-3.609375, PyTorch=-3.84375, abs_diff=0.234375
  logit[85302]: CUDA=3.25, PyTorch=3.015625, abs_diff=0.234375
  logit[76400]: CUDA=0.9375, PyTorch=0.703125, abs_diff=0.234375
  logit[1213]: CUDA=1.0078125, PyTorch=0.77734375, abs_diff=0.23046875
  logit[83687]: CUDA=-0.39648438, PyTorch=-0.625, abs_diff=0.22851562
  logit[74792]: CUDA=-1.9609375, PyTorch=-2.1875, abs_diff=0.2265625
step 55: CUDA token=315, PyTorch token=315
  logits: 5345/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[23066]: CUDA=-1.0703125, PyTorch=-1.3203125, abs_diff=0.25
  logit[18677]: CUDA=1.046875, PyTorch=0.80859375, abs_diff=0.23828125
  logit[4874]: CUDA=-2.34375, PyTorch=-2.578125, abs_diff=0.234375
  logit[31448]: CUDA=-3.296875, PyTorch=-3.53125, abs_diff=0.234375
  logit[26310]: CUDA=1.828125, PyTorch=1.59375, abs_diff=0.234375
  logit[7065]: CUDA=-1.7109375, PyTorch=-1.9453125, abs_diff=0.234375
  logit[12842]: CUDA=-3.890625, PyTorch=-4.125, abs_diff=0.234375
  logit[30613]: CUDA=0.7890625, PyTorch=0.56640625, abs_diff=0.22265625
  logit[14628]: CUDA=-5.15625, PyTorch=-5.375, abs_diff=0.21875
  logit[18540]: CUDA=-1.671875, PyTorch=-1.890625, abs_diff=0.21875
step 56: CUDA token=279, PyTorch token=279
  logits: 1881/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.28125
  logit[32114]: CUDA=-6.875, PyTorch=-7.15625, abs_diff=0.28125
  logit[123563]: CUDA=-3.578125, PyTorch=-3.84375, abs_diff=0.265625
  logit[50938]: CUDA=-4.15625, PyTorch=-3.890625, abs_diff=0.265625
  logit[14818]: CUDA=-3.53125, PyTorch=-3.28125, abs_diff=0.25
  logit[24727]: CUDA=-6.4375, PyTorch=-6.1875, abs_diff=0.25
  logit[27439]: CUDA=-2.34375, PyTorch=-2.09375, abs_diff=0.25
  logit[19659]: CUDA=-4.40625, PyTorch=-4.65625, abs_diff=0.25
  logit[1561]: CUDA=-3.875, PyTorch=-4.125, abs_diff=0.25
  logit[11639]: CUDA=5.125, PyTorch=4.875, abs_diff=0.25
  logit[32511]: CUDA=-5.46875, PyTorch=-5.21875, abs_diff=0.25
step 57: CUDA token=17330, PyTorch token=17330
  logits: 2589/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[93420]: CUDA=-9, PyTorch=-9.3125, abs_diff=0.3125
  logit[13774]: CUDA=-9.5625, PyTorch=-9.875, abs_diff=0.3125
  logit[63390]: CUDA=-1.4140625, PyTorch=-1.6953125, abs_diff=0.28125
  logit[60649]: CUDA=-7.78125, PyTorch=-8.0625, abs_diff=0.28125
  logit[36166]: CUDA=-7.78125, PyTorch=-8.0625, abs_diff=0.28125
  logit[67116]: CUDA=-7.6875, PyTorch=-7.96875, abs_diff=0.28125
  logit[35589]: CUDA=-3.21875, PyTorch=-3.5, abs_diff=0.28125
  logit[28089]: CUDA=-2.671875, PyTorch=-2.9375, abs_diff=0.265625
  logit[7924]: CUDA=-5.34375, PyTorch=-5.59375, abs_diff=0.25
  logit[15744]: CUDA=-5.0625, PyTorch=-5.3125, abs_diff=0.25
step 58: CUDA token=594, PyTorch token=594
  logits: 13378/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.42773438
  logit[24046]: CUDA=-0.6171875, PyTorch=-0.18945312, abs_diff=0.42773438
  logit[22603]: CUDA=1.1484375, PyTorch=1.515625, abs_diff=0.3671875
  logit[46254]: CUDA=-0.25195312, PyTorch=0.099609375, abs_diff=0.3515625
  logit[12165]: CUDA=3.390625, PyTorch=3.734375, abs_diff=0.34375
  logit[17959]: CUDA=-2.15625, PyTorch=-1.8125, abs_diff=0.34375
  logit[64752]: CUDA=-1.0234375, PyTorch=-0.6796875, abs_diff=0.34375
  logit[46606]: CUDA=-4.53125, PyTorch=-4.1875, abs_diff=0.34375
  logit[23768]: CUDA=0.42773438, PyTorch=0.765625, abs_diff=0.33789062
  logit[18567]: CUDA=-2.546875, PyTorch=-2.21875, abs_diff=0.328125
  logit[44625]: CUDA=-3.375, PyTorch=-3.046875, abs_diff=0.328125
step 59: CUDA token=23249, PyTorch token=23249
  logits: 845/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[1894]: CUDA=6.71875, PyTorch=6.96875, abs_diff=0.25
  logit[34473]: CUDA=-7.0625, PyTorch=-7.3125, abs_diff=0.25
  logit[72886]: CUDA=-6.46875, PyTorch=-6.71875, abs_diff=0.25
  logit[230]: CUDA=-6.65625, PyTorch=-6.90625, abs_diff=0.25
  logit[23006]: CUDA=-4.125, PyTorch=-4.375, abs_diff=0.25
  logit[111167]: CUDA=-7, PyTorch=-6.75, abs_diff=0.25
  logit[144031]: CUDA=-6.78125, PyTorch=-7.03125, abs_diff=0.25
  logit[78444]: CUDA=-6.09375, PyTorch=-6.34375, abs_diff=0.25
  logit[35798]: CUDA=0.033935547, PyTorch=0.27734375, abs_diff=0.2434082
  logit[118344]: CUDA=-2.234375, PyTorch=-1.9921875, abs_diff=0.2421875
step 60: CUDA token=389, PyTorch token=389
  logits: 6888/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[70315]: CUDA=-3.578125, PyTorch=-3.890625, abs_diff=0.3125
  logit[18540]: CUDA=-4.46875, PyTorch=-4.78125, abs_diff=0.3125
  logit[69752]: CUDA=-9.25, PyTorch=-9.5625, abs_diff=0.3125
  logit[53837]: CUDA=-8.875, PyTorch=-9.1875, abs_diff=0.3125
  logit[10685]: CUDA=-2.84375, PyTorch=-3.140625, abs_diff=0.296875
  logit[24210]: CUDA=-3.734375, PyTorch=-4.03125, abs_diff=0.296875
  logit[64907]: CUDA=1.375, PyTorch=1.078125, abs_diff=0.296875
  logit[9041]: CUDA=-4.4375, PyTorch=-4.71875, abs_diff=0.28125
  logit[865]: CUDA=-4.875, PyTorch=-5.15625, abs_diff=0.28125
  logit[18328]: CUDA=-4.65625, PyTorch=-4.9375, abs_diff=0.28125
step 61: CUDA token=9237, PyTorch token=9237
  logits: 879/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.21875
  logit[92792]: CUDA=-1.359375, PyTorch=-1.140625, abs_diff=0.21875
  logit[39180]: CUDA=-4.5625, PyTorch=-4.34375, abs_diff=0.21875
  logit[98165]: CUDA=-3.171875, PyTorch=-2.96875, abs_diff=0.203125
  logit[62780]: CUDA=2.15625, PyTorch=2.359375, abs_diff=0.203125
  logit[1841]: CUDA=2.734375, PyTorch=2.9375, abs_diff=0.203125
  logit[99742]: CUDA=-2.984375, PyTorch=-3.1875, abs_diff=0.203125
  logit[78037]: CUDA=-2.4375, PyTorch=-2.640625, abs_diff=0.203125
  logit[23725]: CUDA=3.828125, PyTorch=4.03125, abs_diff=0.203125
  logit[81134]: CUDA=-1.8984375, PyTorch=-1.703125, abs_diff=0.1953125
  logit[48358]: CUDA=-1.6171875, PyTorch=-1.421875, abs_diff=0.1953125
step 62: CUDA token=594, PyTorch token=594
  logits: 1987/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[1262]: CUDA=3.796875, PyTorch=3.484375, abs_diff=0.3125
  logit[398]: CUDA=8.5, PyTorch=8.1875, abs_diff=0.3125
  logit[325]: CUDA=7.9375, PyTorch=7.65625, abs_diff=0.28125
  logit[1541]: CUDA=-1.2890625, PyTorch=-1.5546875, abs_diff=0.265625
  logit[88479]: CUDA=-3.46875, PyTorch=-3.734375, abs_diff=0.265625
  logit[10285]: CUDA=4, PyTorch=3.734375, abs_diff=0.265625
  logit[278]: CUDA=6.3125, PyTorch=6.0625, abs_diff=0.25
  logit[385]: CUDA=5.125, PyTorch=4.875, abs_diff=0.25
  logit[1127]: CUDA=2.484375, PyTorch=2.234375, abs_diff=0.25
  logit[9896]: CUDA=3.75, PyTorch=3.5, abs_diff=0.25
step 63: CUDA token=259, PyTorch token=259
  logits: 1427/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[1326]: CUDA=-0.13671875, PyTorch=-0.42382812, abs_diff=0.28710938
  logit[14702]: CUDA=-5.53125, PyTorch=-5.8125, abs_diff=0.28125
  logit[74792]: CUDA=-1.53125, PyTorch=-1.796875, abs_diff=0.265625
  logit[15752]: CUDA=-1.8046875, PyTorch=-2.0625, abs_diff=0.2578125
  logit[18459]: CUDA=-1.7578125, PyTorch=-2.015625, abs_diff=0.2578125
  logit[17869]: CUDA=-5.9375, PyTorch=-5.6875, abs_diff=0.25
  logit[20421]: CUDA=-6.78125, PyTorch=-7.03125, abs_diff=0.25
  logit[2398]: CUDA=-4.5, PyTorch=-4.75, abs_diff=0.25
  logit[27486]: CUDA=-5.96875, PyTorch=-6.21875, abs_diff=0.25
  logit[12367]: CUDA=-6.3125, PyTorch=-6.5625, abs_diff=0.25
step 64: CUDA token=3341, PyTorch token=3341
  logits: 14857/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.5
  logit[86805]: CUDA=-10, PyTorch=-9.5, abs_diff=0.5
  logit[261]: CUDA=-9.0625, PyTorch=-8.5625, abs_diff=0.5
  logit[100321]: CUDA=-7.65625, PyTorch=-8.125, abs_diff=0.46875
  logit[108739]: CUDA=-7.40625, PyTorch=-7.875, abs_diff=0.46875
  logit[109096]: CUDA=-5.5625, PyTorch=-6.03125, abs_diff=0.46875
  logit[85868]: CUDA=-7.96875, PyTorch=-8.4375, abs_diff=0.46875
  logit[99915]: CUDA=-1.8828125, PyTorch=-2.34375, abs_diff=0.4609375
  logit[89775]: CUDA=-3.890625, PyTorch=-4.34375, abs_diff=0.453125
  logit[535]: CUDA=0.8828125, PyTorch=1.3359375, abs_diff=0.453125
  logit[52833]: CUDA=-2.234375, PyTorch=-2.6875, abs_diff=0.453125
Stopping app - local entrypoint completed.

## QWEN3-4B Auto-regressive prompted generation with 64 Tokens

Prompt: <|im_start|>user
What is one practical way to reduce household energy use?
<|im_end|>
<|im_start|>assistant

Prompt tokens: 19
CUDA: '<think>\nOkay, the user is asking for one practical way to reduce household energy use. Let me think about the most effective and common methods. First, I should consider the most impactful actions that are easy to implement.\n\nEnergy conservation in the home often starts with lighting. Switching to LED bulbs is a common recommendation because'
PyTorch: '<think>\nOkay, the user is asking for one practical way to reduce household energy use. Let me think about the most effective and common methods. First, I should consider the most impactful actions that are easy to implement.\n\nEnergy use in households is often tied to heating, cooling, lighting, and appliances. So maybe something'
step 1: CUDA token=151667, PyTorch token=151667
  logits: 3268/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.40625
  logit[93544]: CUDA=5.34375, PyTorch=5.75, abs_diff=0.40625
  logit[3968]: CUDA=-7.65625, PyTorch=-7.25, abs_diff=0.40625
  logit[44878]: CUDA=11.25, PyTorch=11.625, abs_diff=0.375
  logit[110839]: CUDA=6.59375, PyTorch=6.96875, abs_diff=0.375
  logit[9616]: CUDA=7.46875, PyTorch=7.84375, abs_diff=0.375
  logit[59923]: CUDA=5.125, PyTorch=5.5, abs_diff=0.375
  logit[65648]: CUDA=6.15625, PyTorch=6.53125, abs_diff=0.375
  logit[112441]: CUDA=11.3125, PyTorch=11.6875, abs_diff=0.375
  logit[50118]: CUDA=9.125, PyTorch=9.5, abs_diff=0.375
  logit[20393]: CUDA=8.125, PyTorch=8.5, abs_diff=0.375
step 2: CUDA token=198, PyTorch token=198
  logits: 48031/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.8046875
  logit[69308]: CUDA=1.7734375, PyTorch=0.96875, abs_diff=0.8046875
  logit[49806]: CUDA=8.0625, PyTorch=7.375, abs_diff=0.6875
  logit[85723]: CUDA=3.421875, PyTorch=2.734375, abs_diff=0.6875
  logit[53270]: CUDA=0.71875, PyTorch=0.053466797, abs_diff=0.6652832
  logit[64425]: CUDA=-1.1328125, PyTorch=-1.7734375, abs_diff=0.640625
  logit[121512]: CUDA=2.234375, PyTorch=1.59375, abs_diff=0.640625
  logit[52719]: CUDA=2.59375, PyTorch=1.9609375, abs_diff=0.6328125
  logit[80917]: CUDA=-3.875, PyTorch=-4.5, abs_diff=0.625
  logit[30955]: CUDA=5.1875, PyTorch=5.8125, abs_diff=0.625
  logit[37154]: CUDA=-2.34375, PyTorch=-2.96875, abs_diff=0.625
step 3: CUDA token=32313, PyTorch token=32313
  logits: 31132/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.59375
  logit[98908]: CUDA=-3.734375, PyTorch=-3.140625, abs_diff=0.59375
  logit[35444]: CUDA=-2.296875, PyTorch=-1.71875, abs_diff=0.578125
  logit[114320]: CUDA=-4.4375, PyTorch=-3.875, abs_diff=0.5625
  logit[57738]: CUDA=-3.875, PyTorch=-3.34375, abs_diff=0.53125
  logit[2125]: CUDA=-3.609375, PyTorch=-3.078125, abs_diff=0.53125
  logit[82130]: CUDA=-6.125, PyTorch=-5.59375, abs_diff=0.53125
  logit[45796]: CUDA=-13.75, PyTorch=-13.25, abs_diff=0.5
  logit[67643]: CUDA=-9.25, PyTorch=-8.75, abs_diff=0.5
  logit[53924]: CUDA=2.75, PyTorch=3.25, abs_diff=0.5
  logit[4443]: CUDA=8.375, PyTorch=8.875, abs_diff=0.5
step 4: CUDA token=11, PyTorch token=11
  logits: 18555/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.5625
  logit[1846]: CUDA=-3.6875, PyTorch=-4.25, abs_diff=0.5625
  logit[1588]: CUDA=13.0625, PyTorch=12.5625, abs_diff=0.5
  logit[1493]: CUDA=3.15625, PyTorch=2.671875, abs_diff=0.484375
  logit[582]: CUDA=3.796875, PyTorch=3.34375, abs_diff=0.453125
  logit[279]: CUDA=12.5, PyTorch=12.0625, abs_diff=0.4375
  logit[53619]: CUDA=-8.625, PyTorch=-9.0625, abs_diff=0.4375
  logit[99212]: CUDA=-7.34375, PyTorch=-7.78125, abs_diff=0.4375
  logit[498]: CUDA=5.15625, PyTorch=4.71875, abs_diff=0.4375
  logit[105505]: CUDA=-3.96875, PyTorch=-4.40625, abs_diff=0.4375
  logit[105351]: CUDA=-7.40625, PyTorch=-7.84375, abs_diff=0.4375
step 5: CUDA token=279, PyTorch token=279
  logits: 1729/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[13951]: CUDA=7.1875, PyTorch=7.4375, abs_diff=0.25
  logit[103087]: CUDA=-2.40625, PyTorch=-2.15625, abs_diff=0.25
  logit[9621]: CUDA=1.4765625, PyTorch=1.2421875, abs_diff=0.234375
  logit[98654]: CUDA=1.0859375, PyTorch=0.85546875, abs_diff=0.23046875
  logit[38687]: CUDA=0.71875, PyTorch=0.4921875, abs_diff=0.2265625
  logit[28855]: CUDA=-0.20996094, PyTorch=-0.43359375, abs_diff=0.22363281
  logit[60565]: CUDA=5.96875, PyTorch=5.75, abs_diff=0.21875
  logit[46211]: CUDA=2.609375, PyTorch=2.828125, abs_diff=0.21875
  logit[26865]: CUDA=3.765625, PyTorch=3.546875, abs_diff=0.21875
  logit[84169]: CUDA=3.046875, PyTorch=2.828125, abs_diff=0.21875
step 6: CUDA token=1196, PyTorch token=1196
  logits: 1021/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[1743]: CUDA=5.875, PyTorch=6.125, abs_diff=0.25
  logit[798]: CUDA=2.9375, PyTorch=3.15625, abs_diff=0.21875
  logit[69364]: CUDA=3.203125, PyTorch=3.40625, abs_diff=0.203125
  logit[101117]: CUDA=2.203125, PyTorch=2.40625, abs_diff=0.203125
  logit[15546]: CUDA=-2.328125, PyTorch=-2.125, abs_diff=0.203125
  logit[100727]: CUDA=1.6953125, PyTorch=1.8984375, abs_diff=0.203125
  logit[34003]: CUDA=2.9375, PyTorch=3.140625, abs_diff=0.203125
  logit[47444]: CUDA=2.125, PyTorch=2.328125, abs_diff=0.203125
  logit[9160]: CUDA=1.828125, PyTorch=2.03125, abs_diff=0.203125
  logit[58436]: CUDA=-0.11328125, PyTorch=0.08203125, abs_diff=0.1953125
step 7: CUDA token=374, PyTorch token=374
  logits: 109/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.20703125
  logit[558]: CUDA=-0.86328125, PyTorch=-0.65625, abs_diff=0.20703125
  logit[114411]: CUDA=-2.53125, PyTorch=-2.34375, abs_diff=0.1875
  logit[57701]: CUDA=-0.052734375, PyTorch=0.10546875, abs_diff=0.15820312
  logit[1216]: CUDA=0.31445312, PyTorch=0.47265625, abs_diff=0.15820312
  logit[15432]: CUDA=-2.671875, PyTorch=-2.515625, abs_diff=0.15625
  logit[46631]: CUDA=1.90625, PyTorch=2.0625, abs_diff=0.15625
  logit[68717]: CUDA=-0.15429688, PyTorch=-0.0023193359, abs_diff=0.15197754
  logit[107474]: CUDA=-1.828125, PyTorch=-1.6796875, abs_diff=0.1484375
  logit[8263]: CUDA=-1.9765625, PyTorch=-2.125, abs_diff=0.1484375
  logit[133226]: CUDA=-0.53515625, PyTorch=-0.390625, abs_diff=0.14453125
step 8: CUDA token=10161, PyTorch token=10161
  logits: 219/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[54054]: CUDA=5.46875, PyTorch=5.6875, abs_diff=0.21875
  logit[53295]: CUDA=5.5, PyTorch=5.71875, abs_diff=0.21875
  logit[1681]: CUDA=0.05078125, PyTorch=-0.15527344, abs_diff=0.20605469
  logit[10215]: CUDA=1.3359375, PyTorch=1.53125, abs_diff=0.1953125
  logit[101117]: CUDA=2.09375, PyTorch=2.28125, abs_diff=0.1875
  logit[3081]: CUDA=-2.28125, PyTorch=-2.46875, abs_diff=0.1875
  logit[14898]: CUDA=-4.15625, PyTorch=-4.34375, abs_diff=0.1875
  logit[112826]: CUDA=-2.90625, PyTorch=-2.71875, abs_diff=0.1875
  logit[113900]: CUDA=1.3046875, PyTorch=1.484375, abs_diff=0.1796875
  logit[7271]: CUDA=-2.828125, PyTorch=-3, abs_diff=0.171875
step 9: CUDA token=369, PyTorch token=369
  logits: 2657/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.34375
  logit[99650]: CUDA=-5.9375, PyTorch=-6.28125, abs_diff=0.34375
  logit[107469]: CUDA=-5.40625, PyTorch=-5.71875, abs_diff=0.3125
  logit[136283]: CUDA=-5.53125, PyTorch=-5.84375, abs_diff=0.3125
  logit[807]: CUDA=6.6875, PyTorch=6.375, abs_diff=0.3125
  logit[53791]: CUDA=-6.21875, PyTorch=-6.5, abs_diff=0.28125
  logit[102091]: CUDA=-5.875, PyTorch=-6.15625, abs_diff=0.28125
  logit[6865]: CUDA=-4.84375, PyTorch=-5.125, abs_diff=0.28125
  logit[62493]: CUDA=-5.21875, PyTorch=-5.5, abs_diff=0.28125
  logit[20069]: CUDA=-6.78125, PyTorch=-7.0625, abs_diff=0.28125
  logit[75269]: CUDA=0.73828125, PyTorch=0.47265625, abs_diff=0.265625
step 10: CUDA token=825, PyTorch token=825
  logits: 14707/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.375
  logit[10295]: CUDA=11.1875, PyTorch=10.8125, abs_diff=0.375
  logit[7373]: CUDA=10.625, PyTorch=10.25, abs_diff=0.375
  logit[11050]: CUDA=7.6875, PyTorch=7.34375, abs_diff=0.34375
  logit[26279]: CUDA=6.59375, PyTorch=6.28125, abs_diff=0.3125
  logit[4285]: CUDA=10.5, PyTorch=10.1875, abs_diff=0.3125
  logit[30814]: CUDA=1.7421875, PyTorch=1.4453125, abs_diff=0.296875
  logit[81861]: CUDA=1.375, PyTorch=1.0859375, abs_diff=0.2890625
  logit[3742]: CUDA=5.0625, PyTorch=4.78125, abs_diff=0.28125
  logit[1739]: CUDA=4.53125, PyTorch=4.25, abs_diff=0.28125
  logit[8282]: CUDA=5.09375, PyTorch=4.8125, abs_diff=0.28125
step 11: CUDA token=14976, PyTorch token=14976
  logits: 48423/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.5625
  logit[32796]: CUDA=-11.125, PyTorch=-11.6875, abs_diff=0.5625
  logit[137905]: CUDA=-13.1875, PyTorch=-13.75, abs_diff=0.5625
  logit[38165]: CUDA=-12.125, PyTorch=-12.6875, abs_diff=0.5625
  logit[32473]: CUDA=-9.8125, PyTorch=-10.3125, abs_diff=0.5
  logit[32320]: CUDA=-16, PyTorch=-16.5, abs_diff=0.5
  logit[42647]: CUDA=-13, PyTorch=-13.5, abs_diff=0.5
  logit[42894]: CUDA=-16.25, PyTorch=-16.75, abs_diff=0.5
  logit[25964]: CUDA=-11.6875, PyTorch=-12.1875, abs_diff=0.5
  logit[44318]: CUDA=-14.4375, PyTorch=-14.9375, abs_diff=0.5
  logit[21898]: CUDA=-10.1875, PyTorch=-10.6875, abs_diff=0.5
step 12: CUDA token=1616, PyTorch token=1616
  logits: 22228/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.5
  logit[41030]: CUDA=-16.375, PyTorch=-16.875, abs_diff=0.5
  logit[87470]: CUDA=-19.5, PyTorch=-20, abs_diff=0.5
  logit[109020]: CUDA=-14.875, PyTorch=-15.375, abs_diff=0.5
  logit[111586]: CUDA=-16.375, PyTorch=-16.875, abs_diff=0.5
  logit[76175]: CUDA=-10, PyTorch=-10.5, abs_diff=0.5
  logit[117990]: CUDA=-10, PyTorch=-10.5, abs_diff=0.5
  logit[62907]: CUDA=-17, PyTorch=-17.5, abs_diff=0.5
  logit[16618]: CUDA=-7.90625, PyTorch=-8.375, abs_diff=0.46875
  logit[43942]: CUDA=-10.4375, PyTorch=-10.875, abs_diff=0.4375
  logit[36630]: CUDA=-14.5625, PyTorch=-15, abs_diff=0.4375
step 13: CUDA token=311, PyTorch token=311
  logits: 3672/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[10744]: CUDA=8.125, PyTorch=7.84375, abs_diff=0.28125
  logit[113901]: CUDA=1.890625, PyTorch=1.6171875, abs_diff=0.2734375
  logit[2841]: CUDA=7.6875, PyTorch=7.4375, abs_diff=0.25
  logit[13154]: CUDA=6.28125, PyTorch=6.03125, abs_diff=0.25
  logit[3753]: CUDA=7.375, PyTorch=7.125, abs_diff=0.25
  logit[12534]: CUDA=4.09375, PyTorch=3.84375, abs_diff=0.25
  logit[14744]: CUDA=3.125, PyTorch=2.875, abs_diff=0.25
  logit[28960]: CUDA=2.5, PyTorch=2.25, abs_diff=0.25
  logit[29894]: CUDA=4.5625, PyTorch=4.3125, abs_diff=0.25
  logit[5101]: CUDA=4.1875, PyTorch=3.9375, abs_diff=0.25
step 14: CUDA token=7949, PyTorch token=7949
  logits: 4609/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.421875
  logit[72576]: CUDA=-2.359375, PyTorch=-2.78125, abs_diff=0.421875
  logit[77966]: CUDA=-4.28125, PyTorch=-4.6875, abs_diff=0.40625
  logit[21564]: CUDA=-2.140625, PyTorch=-2.515625, abs_diff=0.375
  logit[17606]: CUDA=-7.9375, PyTorch=-7.5625, abs_diff=0.375
  logit[12575]: CUDA=2.671875, PyTorch=2.296875, abs_diff=0.375
  logit[9411]: CUDA=-8.375, PyTorch=-8, abs_diff=0.375
  logit[67884]: CUDA=-9.1875, PyTorch=-9.5625, abs_diff=0.375
  logit[3440]: CUDA=-11.875, PyTorch=-11.5, abs_diff=0.375
  logit[54073]: CUDA=-8.125, PyTorch=-8.5, abs_diff=0.375
  logit[19286]: CUDA=-11.125, PyTorch=-11.5, abs_diff=0.375
step 15: CUDA token=13732, PyTorch token=13732
  logits: 2535/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.4375
  logit[44105]: CUDA=-13.3125, PyTorch=-13.75, abs_diff=0.4375
  logit[120277]: CUDA=-8.0625, PyTorch=-8.4375, abs_diff=0.375
  logit[68076]: CUDA=-3.96875, PyTorch=-4.34375, abs_diff=0.375
  logit[126793]: CUDA=-9.25, PyTorch=-9.625, abs_diff=0.375
  logit[78448]: CUDA=-13.0625, PyTorch=-13.4375, abs_diff=0.375
  logit[76829]: CUDA=-12.625, PyTorch=-13, abs_diff=0.375
  logit[119059]: CUDA=-11.4375, PyTorch=-11.8125, abs_diff=0.375
  logit[128000]: CUDA=-1.3671875, PyTorch=-1.015625, abs_diff=0.3515625
  logit[38138]: CUDA=-0.3984375, PyTorch=-0.75, abs_diff=0.3515625
  logit[45740]: CUDA=1.6875, PyTorch=2.03125, abs_diff=0.34375
step 16: CUDA token=4802, PyTorch token=4802
  logits: 3214/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.375
  logit[5233]: CUDA=-8.9375, PyTorch=-9.3125, abs_diff=0.375
  logit[91394]: CUDA=-3.125, PyTorch=-3.5, abs_diff=0.375
  logit[95402]: CUDA=-6.5, PyTorch=-6.84375, abs_diff=0.34375
  logit[61561]: CUDA=-5.125, PyTorch=-5.46875, abs_diff=0.34375
  logit[76805]: CUDA=-7.1875, PyTorch=-7.53125, abs_diff=0.34375
  logit[116563]: CUDA=-2.453125, PyTorch=-2.78125, abs_diff=0.328125
  logit[56111]: CUDA=4.03125, PyTorch=3.703125, abs_diff=0.328125
  logit[9313]: CUDA=-9.625, PyTorch=-9.9375, abs_diff=0.3125
  logit[19956]: CUDA=-7.875, PyTorch=-8.1875, abs_diff=0.3125
  logit[7157]: CUDA=-4.28125, PyTorch=-4.59375, abs_diff=0.3125
step 17: CUDA token=990, PyTorch token=990
  logits: 2263/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.40625
  logit[107494]: CUDA=-3.96875, PyTorch=-3.5625, abs_diff=0.40625
  logit[1679]: CUDA=-8.25, PyTorch=-8.625, abs_diff=0.375
  logit[3847]: CUDA=11.375, PyTorch=11.75, abs_diff=0.375
  logit[90682]: CUDA=-4.4375, PyTorch=-4.8125, abs_diff=0.375
  logit[51568]: CUDA=-8.6875, PyTorch=-9.0625, abs_diff=0.375
  logit[110782]: CUDA=-5.34375, PyTorch=-4.96875, abs_diff=0.375
  logit[88080]: CUDA=-9.0625, PyTorch=-9.4375, abs_diff=0.375
  logit[1196]: CUDA=12.4375, PyTorch=12.8125, abs_diff=0.375
  logit[20002]: CUDA=-0.71875, PyTorch=-0.36914062, abs_diff=0.34960938
  logit[93981]: CUDA=-6.5, PyTorch=-6.84375, abs_diff=0.34375
step 18: CUDA token=13, PyTorch token=13
  logits: 518/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.5
  logit[17701]: CUDA=-5.59375, PyTorch=-5.9375, abs_diff=0.34375
  logit[26850]: CUDA=-5.21875, PyTorch=-5.53125, abs_diff=0.3125
  logit[55430]: CUDA=-7.5, PyTorch=-7.8125, abs_diff=0.3125
  logit[57073]: CUDA=-8.375, PyTorch=-8.6875, abs_diff=0.3125
  logit[86779]: CUDA=-9.75, PyTorch=-10.0625, abs_diff=0.3125
  logit[58418]: CUDA=-8.25, PyTorch=-8.5625, abs_diff=0.3125
  logit[94081]: CUDA=-8.75, PyTorch=-9.0625, abs_diff=0.3125
  logit[68562]: CUDA=-8.3125, PyTorch=-8.625, abs_diff=0.3125
  logit[51418]: CUDA=-8.375, PyTorch=-8.6875, abs_diff=0.3125
  logit[73530]: CUDA=-8.125, PyTorch=-8.4375, abs_diff=0.3125
step 19: CUDA token=6771, PyTorch token=6771
  logits: 516/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[113761]: CUDA=0.9375, PyTorch=1.1328125, abs_diff=0.1953125
  logit[30089]: CUDA=-2.015625, PyTorch=-1.828125, abs_diff=0.1875
  logit[103776]: CUDA=2.59375, PyTorch=2.78125, abs_diff=0.1875
  logit[64170]: CUDA=-1.2734375, PyTorch=-1.09375, abs_diff=0.1796875
  logit[119352]: CUDA=-0.40234375, PyTorch=-0.22558594, abs_diff=0.17675781
  logit[99985]: CUDA=-2.421875, PyTorch=-2.25, abs_diff=0.171875
  logit[7733]: CUDA=-3.34375, PyTorch=-3.171875, abs_diff=0.171875
  logit[92307]: CUDA=1.953125, PyTorch=2.125, abs_diff=0.171875
  logit[19224]: CUDA=-3.28125, PyTorch=-3.109375, abs_diff=0.171875
  logit[51901]: CUDA=-0.7578125, PyTorch=-0.5859375, abs_diff=0.171875
step 20: CUDA token=752, PyTorch token=752
  logits: 4869/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.28125
  logit[91911]: CUDA=-4.09375, PyTorch=-3.8125, abs_diff=0.28125
  logit[117234]: CUDA=-5.875, PyTorch=-5.59375, abs_diff=0.28125
  logit[106808]: CUDA=-3.640625, PyTorch=-3.375, abs_diff=0.265625
  logit[86840]: CUDA=-2.859375, PyTorch=-2.59375, abs_diff=0.265625
  logit[79562]: CUDA=0.030273438, PyTorch=0.28710938, abs_diff=0.25683594
  logit[37041]: CUDA=-4.53125, PyTorch=-4.28125, abs_diff=0.25
  logit[75557]: CUDA=3.75, PyTorch=4, abs_diff=0.25
  logit[73861]: CUDA=-3.78125, PyTorch=-3.53125, abs_diff=0.25
  logit[76287]: CUDA=-1.4609375, PyTorch=-1.2109375, abs_diff=0.25
  logit[42177]: CUDA=-7.1875, PyTorch=-6.9375, abs_diff=0.25
step 21: CUDA token=1744, PyTorch token=1744
  logits: 4338/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[71325]: CUDA=-10.3125, PyTorch=-10, abs_diff=0.3125
  logit[26213]: CUDA=-2.953125, PyTorch=-2.65625, abs_diff=0.296875
  logit[108177]: CUDA=-6.875, PyTorch=-6.59375, abs_diff=0.28125
  logit[51216]: CUDA=-8.125, PyTorch=-7.84375, abs_diff=0.28125
  logit[15634]: CUDA=2.921875, PyTorch=3.1875, abs_diff=0.265625
  logit[68337]: CUDA=-6.25, PyTorch=-6, abs_diff=0.25
  logit[69395]: CUDA=-7.1875, PyTorch=-6.9375, abs_diff=0.25
  logit[44953]: CUDA=-5.9375, PyTorch=-5.6875, abs_diff=0.25
  logit[18512]: CUDA=5.5625, PyTorch=5.3125, abs_diff=0.25
  logit[12051]: CUDA=-2.375, PyTorch=-2.125, abs_diff=0.25
step 22: CUDA token=911, PyTorch token=911
  logits: 11131/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.375
  logit[49899]: CUDA=8.0625, PyTorch=7.78125, abs_diff=0.28125
  logit[3055]: CUDA=7.5, PyTorch=7.21875, abs_diff=0.28125
  logit[20699]: CUDA=6.6875, PyTorch=6.40625, abs_diff=0.28125
  logit[15750]: CUDA=8.0625, PyTorch=7.78125, abs_diff=0.28125
  logit[4815]: CUDA=5.96875, PyTorch=5.6875, abs_diff=0.28125
  logit[78459]: CUDA=2.09375, PyTorch=1.828125, abs_diff=0.265625
  logit[23638]: CUDA=3.78125, PyTorch=3.515625, abs_diff=0.265625
  logit[3381]: CUDA=4.09375, PyTorch=3.828125, abs_diff=0.265625
  logit[109148]: CUDA=-1.3046875, PyTorch=-1.5703125, abs_diff=0.265625
  logit[8865]: CUDA=3.15625, PyTorch=2.890625, abs_diff=0.265625
step 23: CUDA token=279, PyTorch token=279
  logits: 2550/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[111272]: CUDA=-4.96875, PyTorch=-5.21875, abs_diff=0.25
  logit[118372]: CUDA=-6.21875, PyTorch=-6.46875, abs_diff=0.25
  logit[108329]: CUDA=-5.9375, PyTorch=-6.1875, abs_diff=0.25
  logit[118770]: CUDA=-5.34375, PyTorch=-5.59375, abs_diff=0.25
  logit[103583]: CUDA=-6.84375, PyTorch=-7.09375, abs_diff=0.25
  logit[119353]: CUDA=1.9296875, PyTorch=1.6796875, abs_diff=0.25
  logit[48311]: CUDA=-1.8125, PyTorch=-2.046875, abs_diff=0.234375
  logit[41055]: CUDA=-3.34375, PyTorch=-3.578125, abs_diff=0.234375
  logit[114154]: CUDA=-3.46875, PyTorch=-3.703125, abs_diff=0.234375
  logit[64082]: CUDA=-3.921875, PyTorch=-4.15625, abs_diff=0.234375
step 24: CUDA token=1429, PyTorch token=1429
  logits: 2021/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.359375
  logit[65636]: CUDA=-2.03125, PyTorch=-1.671875, abs_diff=0.359375
  logit[102556]: CUDA=-2.4375, PyTorch=-2.1875, abs_diff=0.25
  logit[71494]: CUDA=-5.125, PyTorch=-4.875, abs_diff=0.25
  logit[55080]: CUDA=-3.5625, PyTorch=-3.3125, abs_diff=0.25
  logit[69647]: CUDA=-5.25, PyTorch=-5, abs_diff=0.25
  logit[108081]: CUDA=-6.5, PyTorch=-6.25, abs_diff=0.25
  logit[93883]: CUDA=-5.6875, PyTorch=-5.4375, abs_diff=0.25
  logit[23565]: CUDA=0.20703125, PyTorch=0.45703125, abs_diff=0.25
  logit[88962]: CUDA=-1.515625, PyTorch=-1.2734375, abs_diff=0.2421875
  logit[89382]: CUDA=-2.3125, PyTorch=-2.078125, abs_diff=0.234375
step 25: CUDA token=7373, PyTorch token=7373
  logits: 6434/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.5
  logit[14976]: CUDA=16.875, PyTorch=16.375, abs_diff=0.5
  logit[65636]: CUDA=4.1875, PyTorch=3.78125, abs_diff=0.40625
  logit[118603]: CUDA=-5.5, PyTorch=-5.875, abs_diff=0.375
  logit[87588]: CUDA=-9.6875, PyTorch=-10.0625, abs_diff=0.375
  logit[134801]: CUDA=-3.84375, PyTorch=-4.21875, abs_diff=0.375
  logit[110796]: CUDA=-7.625, PyTorch=-8, abs_diff=0.375
  logit[17891]: CUDA=-6.46875, PyTorch=-6.84375, abs_diff=0.375
  logit[94134]: CUDA=-4.6875, PyTorch=-5.03125, abs_diff=0.34375
  logit[108433]: CUDA=-5.53125, PyTorch=-5.875, abs_diff=0.34375
  logit[108081]: CUDA=-3.109375, PyTorch=-3.453125, abs_diff=0.34375
step 26: CUDA token=323, PyTorch token=323
  logits: 3639/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.625
  logit[3175]: CUDA=19.875, PyTorch=20.5, abs_diff=0.625
  logit[10953]: CUDA=19, PyTorch=19.5, abs_diff=0.5
  logit[7354]: CUDA=18.75, PyTorch=19.25, abs_diff=0.5
  logit[17389]: CUDA=11.6875, PyTorch=12.125, abs_diff=0.4375
  logit[15338]: CUDA=7.78125, PyTorch=8.1875, abs_diff=0.40625
  logit[13529]: CUDA=8.3125, PyTorch=8.6875, abs_diff=0.375
  logit[512]: CUDA=8.8125, PyTorch=9.1875, abs_diff=0.375
  logit[10888]: CUDA=8.1875, PyTorch=8.5625, abs_diff=0.375
  logit[20210]: CUDA=-10.3125, PyTorch=-10.6875, abs_diff=0.375
  logit[11327]: CUDA=9.6875, PyTorch=10.0625, abs_diff=0.375
step 27: CUDA token=4185, PyTorch token=4185
  logits: 1178/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.28125
  logit[101432]: CUDA=-6, PyTorch=-6.28125, abs_diff=0.28125
  logit[46923]: CUDA=-1.203125, PyTorch=-1.46875, abs_diff=0.265625
  logit[77895]: CUDA=-2.78125, PyTorch=-3.03125, abs_diff=0.25
  logit[109066]: CUDA=-2.890625, PyTorch=-3.140625, abs_diff=0.25
  logit[110796]: CUDA=-4.625, PyTorch=-4.875, abs_diff=0.25
  logit[33763]: CUDA=6.6875, PyTorch=6.4375, abs_diff=0.25
  logit[98237]: CUDA=-3.015625, PyTorch=-2.78125, abs_diff=0.234375
  logit[50468]: CUDA=3.15625, PyTorch=2.921875, abs_diff=0.234375
  logit[17891]: CUDA=-3.609375, PyTorch=-3.84375, abs_diff=0.234375
  logit[6655]: CUDA=-1.234375, PyTorch=-1.4609375, abs_diff=0.2265625
step 28: CUDA token=5413, PyTorch token=5413
  logits: 2871/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.5
  logit[7373]: CUDA=9.625, PyTorch=9.1875, abs_diff=0.4375
  logit[52321]: CUDA=9.1875, PyTorch=8.8125, abs_diff=0.375
  logit[46923]: CUDA=2.578125, PyTorch=2.203125, abs_diff=0.375
  logit[67539]: CUDA=2.078125, PyTorch=1.7109375, abs_diff=0.3671875
  logit[104775]: CUDA=0.0070800781, PyTorch=-0.35546875, abs_diff=0.36254883
  logit[77895]: CUDA=0.31445312, PyTorch=-0.040527344, abs_diff=0.35498047
  logit[88086]: CUDA=-3.140625, PyTorch=-3.46875, abs_diff=0.328125
  logit[54188]: CUDA=6.03125, PyTorch=5.71875, abs_diff=0.3125
  logit[39684]: CUDA=-5.125, PyTorch=-5.4375, abs_diff=0.3125
  logit[44094]: CUDA=5.59375, PyTorch=5.3125, abs_diff=0.28125
step 29: CUDA token=13, PyTorch token=13
  logits: 224/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[52656]: CUDA=-1.328125, PyTorch=-1.5234375, abs_diff=0.1953125
  logit[23990]: CUDA=-0.47460938, PyTorch=-0.2890625, abs_diff=0.18554688
  logit[102756]: CUDA=0.19140625, PyTorch=0.016845703, abs_diff=0.17456055
  logit[88086]: CUDA=-0.66015625, PyTorch=-0.828125, abs_diff=0.16796875
  logit[94231]: CUDA=0.5078125, PyTorch=0.67578125, abs_diff=0.16796875
  logit[99390]: CUDA=-0.8046875, PyTorch=-0.63671875, abs_diff=0.16796875
  logit[114522]: CUDA=-2.5625, PyTorch=-2.71875, abs_diff=0.15625
  logit[55714]: CUDA=2.53125, PyTorch=2.6875, abs_diff=0.15625
  logit[64962]: CUDA=2.296875, PyTorch=2.140625, abs_diff=0.15625
  logit[104775]: CUDA=-1.0859375, PyTorch=-1.2421875, abs_diff=0.15625
step 30: CUDA token=5512, PyTorch token=5512
  logits: 1156/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[36107]: CUDA=8.375, PyTorch=8.0625, abs_diff=0.3125
  logit[47279]: CUDA=2.390625, PyTorch=2.125, abs_diff=0.265625
  logit[65222]: CUDA=0.6640625, PyTorch=0.41210938, abs_diff=0.25195312
  logit[38934]: CUDA=0.90625, PyTorch=0.65625, abs_diff=0.25
  logit[25422]: CUDA=-3.875, PyTorch=-4.125, abs_diff=0.25
  logit[12824]: CUDA=-5.09375, PyTorch=-4.84375, abs_diff=0.25
  logit[66260]: CUDA=-3.40625, PyTorch=-3.15625, abs_diff=0.25
  logit[103026]: CUDA=-1.2734375, PyTorch=-1.5234375, abs_diff=0.25
  logit[22605]: CUDA=2.328125, PyTorch=2.078125, abs_diff=0.25
  logit[13481]: CUDA=6.03125, PyTorch=5.78125, abs_diff=0.25
step 31: CUDA token=11, PyTorch token=11
  logits: 6729/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[4965]: CUDA=3.125, PyTorch=2.8125, abs_diff=0.3125
  logit[9669]: CUDA=1.7109375, PyTorch=1.4140625, abs_diff=0.296875
  logit[49831]: CUDA=-0.80859375, PyTorch=-1.1015625, abs_diff=0.29296875
  logit[62790]: CUDA=6.5, PyTorch=6.21875, abs_diff=0.28125
  logit[33017]: CUDA=2.953125, PyTorch=2.671875, abs_diff=0.28125
  logit[38140]: CUDA=-1.4453125, PyTorch=-1.71875, abs_diff=0.2734375
  logit[42226]: CUDA=2.875, PyTorch=2.609375, abs_diff=0.265625
  logit[8948]: CUDA=1.7734375, PyTorch=1.5078125, abs_diff=0.265625
  logit[85760]: CUDA=-1.203125, PyTorch=-1.46875, abs_diff=0.265625
  logit[59675]: CUDA=1.09375, PyTorch=0.83203125, abs_diff=0.26171875
step 32: CUDA token=358, PyTorch token=358
  logits: 2437/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[18857]: CUDA=-2.8125, PyTorch=-2.5, abs_diff=0.3125
  logit[54431]: CUDA=-6.21875, PyTorch=-5.9375, abs_diff=0.28125
  logit[39438]: CUDA=-4.65625, PyTorch=-4.375, abs_diff=0.28125
  logit[28951]: CUDA=-7.84375, PyTorch=-7.5625, abs_diff=0.28125
  logit[32538]: CUDA=5.75, PyTorch=6.03125, abs_diff=0.28125
  logit[33781]: CUDA=-1.296875, PyTorch=-1.0234375, abs_diff=0.2734375
  logit[93679]: CUDA=0.953125, PyTorch=1.21875, abs_diff=0.265625
  logit[19042]: CUDA=-3.125, PyTorch=-2.859375, abs_diff=0.265625
  logit[9540]: CUDA=-0.24316406, PyTorch=0.022460938, abs_diff=0.265625
  logit[89279]: CUDA=-1.9296875, PyTorch=-1.671875, abs_diff=0.2578125
step 33: CUDA token=1265, PyTorch token=1265
  logits: 1574/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[53407]: CUDA=0.8046875, PyTorch=0.58203125, abs_diff=0.22265625
  logit[100282]: CUDA=-5.40625, PyTorch=-5.625, abs_diff=0.21875
  logit[43532]: CUDA=-3.484375, PyTorch=-3.703125, abs_diff=0.21875
  logit[74200]: CUDA=-4.84375, PyTorch=-5.0625, abs_diff=0.21875
  logit[70824]: CUDA=-2.171875, PyTorch=-2.390625, abs_diff=0.21875
  logit[108450]: CUDA=-1, PyTorch=-1.21875, abs_diff=0.21875
  logit[1609]: CUDA=0.99609375, PyTorch=0.78125, abs_diff=0.21484375
  logit[84111]: CUDA=-0.78125, PyTorch=-0.9921875, abs_diff=0.2109375
  logit[107580]: CUDA=-0.7890625, PyTorch=-1, abs_diff=0.2109375
  logit[28850]: CUDA=0.74609375, PyTorch=0.53515625, abs_diff=0.2109375
step 34: CUDA token=2908, PyTorch token=2908
  logits: 1275/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.28125
  logit[13096]: CUDA=5.71875, PyTorch=6, abs_diff=0.28125
  logit[19434]: CUDA=0.36132812, PyTorch=0.61328125, abs_diff=0.25195312
  logit[27052]: CUDA=4.0625, PyTorch=4.3125, abs_diff=0.25
  logit[15020]: CUDA=-4.71875, PyTorch=-4.96875, abs_diff=0.25
  logit[110940]: CUDA=-6.0625, PyTorch=-6.3125, abs_diff=0.25
  logit[14586]: CUDA=1.9609375, PyTorch=2.203125, abs_diff=0.2421875
  logit[36953]: CUDA=-0.5234375, PyTorch=-0.28320312, abs_diff=0.24023438
  logit[13598]: CUDA=0.796875, PyTorch=1.03125, abs_diff=0.234375
  logit[53636]: CUDA=-2.640625, PyTorch=-2.875, abs_diff=0.234375
  logit[12771]: CUDA=3.484375, PyTorch=3.71875, abs_diff=0.234375
step 35: CUDA token=279, PyTorch token=279
  logits: 2419/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.34375
  logit[6730]: CUDA=2.4375, PyTorch=2.09375, abs_diff=0.34375
  logit[18801]: CUDA=4.96875, PyTorch=4.625, abs_diff=0.34375
  logit[17262]: CUDA=-0.75390625, PyTorch=-1.09375, abs_diff=0.33984375
  logit[39907]: CUDA=1.7578125, PyTorch=1.421875, abs_diff=0.3359375
  logit[38074]: CUDA=0.61328125, PyTorch=0.28515625, abs_diff=0.328125
  logit[55229]: CUDA=1.7578125, PyTorch=1.4453125, abs_diff=0.3125
  logit[104339]: CUDA=0.6953125, PyTorch=0.40039062, abs_diff=0.29492188
  logit[81454]: CUDA=-0.421875, PyTorch=-0.71484375, abs_diff=0.29296875
  logit[20849]: CUDA=4.3125, PyTorch=4.59375, abs_diff=0.28125
  logit[9032]: CUDA=-4.40625, PyTorch=-4.6875, abs_diff=0.28125
step 36: CUDA token=1429, PyTorch token=1429
  logits: 2552/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.375
  logit[22032]: CUDA=9.125, PyTorch=9.4375, abs_diff=0.3125
  logit[2064]: CUDA=1.4765625, PyTorch=1.7578125, abs_diff=0.28125
  logit[12200]: CUDA=2.34375, PyTorch=2.609375, abs_diff=0.265625
  logit[9435]: CUDA=5.875, PyTorch=6.125, abs_diff=0.25
  logit[22686]: CUDA=7.0625, PyTorch=7.3125, abs_diff=0.25
  logit[53055]: CUDA=4.59375, PyTorch=4.84375, abs_diff=0.25
  logit[6645]: CUDA=4.25, PyTorch=4.5, abs_diff=0.25
  logit[71289]: CUDA=4.9375, PyTorch=5.1875, abs_diff=0.25
  logit[3468]: CUDA=5.65625, PyTorch=5.90625, abs_diff=0.25
  logit[5570]: CUDA=5.15625, PyTorch=5.40625, abs_diff=0.25
step 37: CUDA token=97890, PyTorch token=97890
  logits: 2227/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.5
  logit[14976]: CUDA=12, PyTorch=11.5, abs_diff=0.5
  logit[65636]: CUDA=0.52734375, PyTorch=0.14453125, abs_diff=0.3828125
  logit[135085]: CUDA=-0.8046875, PyTorch=-1.171875, abs_diff=0.3671875
  logit[100419]: CUDA=-8.375, PyTorch=-8.6875, abs_diff=0.3125
  logit[125403]: CUDA=-9.3125, PyTorch=-9, abs_diff=0.3125
  logit[31267]: CUDA=6.125, PyTorch=5.8125, abs_diff=0.3125
  logit[59779]: CUDA=-9.5625, PyTorch=-9.25, abs_diff=0.3125
  logit[87230]: CUDA=-9.0625, PyTorch=-8.75, abs_diff=0.3125
  logit[26650]: CUDA=4.96875, PyTorch=4.6875, abs_diff=0.28125
  logit[134408]: CUDA=-4.6875, PyTorch=-4.96875, abs_diff=0.28125
step 38: CUDA token=6168, PyTorch token=6168
  logits: 5143/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.375
  logit[52454]: CUDA=-6.65625, PyTorch=-7.03125, abs_diff=0.375
  logit[121537]: CUDA=-11.5, PyTorch=-11.875, abs_diff=0.375
  logit[93597]: CUDA=-4.46875, PyTorch=-4.8125, abs_diff=0.34375
  logit[41359]: CUDA=-6.21875, PyTorch=-6.5625, abs_diff=0.34375
  logit[31291]: CUDA=-5.09375, PyTorch=-5.40625, abs_diff=0.3125
  logit[6991]: CUDA=-8.6875, PyTorch=-9, abs_diff=0.3125
  logit[22300]: CUDA=-8.125, PyTorch=-8.4375, abs_diff=0.3125
  logit[7216]: CUDA=-8.125, PyTorch=-8.4375, abs_diff=0.3125
  logit[4011]: CUDA=-8.5, PyTorch=-8.8125, abs_diff=0.3125
  logit[55793]: CUDA=-9.875, PyTorch=-10.1875, abs_diff=0.3125
step 39: CUDA token=429, PyTorch token=429
  logits: 12666/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.2890625
  logit[79537]: CUDA=1.09375, PyTorch=1.3828125, abs_diff=0.2890625
  logit[24905]: CUDA=5.0625, PyTorch=5.34375, abs_diff=0.28125
  logit[36445]: CUDA=6.28125, PyTorch=6.5625, abs_diff=0.28125
  logit[114748]: CUDA=-2.796875, PyTorch=-2.515625, abs_diff=0.28125
  logit[72955]: CUDA=5.09375, PyTorch=5.375, abs_diff=0.28125
  logit[77809]: CUDA=2.90625, PyTorch=3.1875, abs_diff=0.28125
  logit[60428]: CUDA=6.8125, PyTorch=7.09375, abs_diff=0.28125
  logit[97051]: CUDA=-1.1640625, PyTorch=-0.91015625, abs_diff=0.25390625
  logit[18512]: CUDA=5.15625, PyTorch=5.40625, abs_diff=0.25
  logit[39243]: CUDA=4.6875, PyTorch=4.9375, abs_diff=0.25
step 40: CUDA token=525, PyTorch token=525
  logits: 875/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.25
  logit[107363]: CUDA=-7.5625, PyTorch=-7.3125, abs_diff=0.25
  logit[107653]: CUDA=-6.75, PyTorch=-6.5, abs_diff=0.25
  logit[110697]: CUDA=-2.28125, PyTorch=-2.046875, abs_diff=0.234375
  logit[89500]: CUDA=-1.7265625, PyTorch=-1.4921875, abs_diff=0.234375
  logit[92643]: CUDA=-3.5625, PyTorch=-3.328125, abs_diff=0.234375
  logit[27402]: CUDA=0.34570312, PyTorch=0.5703125, abs_diff=0.22460938
  logit[106628]: CUDA=-5.5625, PyTorch=-5.34375, abs_diff=0.21875
  logit[80779]: CUDA=3.015625, PyTorch=2.8125, abs_diff=0.203125
  logit[106269]: CUDA=2.328125, PyTorch=2.53125, abs_diff=0.203125
  logit[75265]: CUDA=-2.09375, PyTorch=-2.296875, abs_diff=0.203125
step 41: CUDA token=4135, PyTorch token=4135
  logits: 4160/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.5625
  logit[14976]: CUDA=14.9375, PyTorch=14.375, abs_diff=0.5625
  logit[65636]: CUDA=2.578125, PyTorch=2.03125, abs_diff=0.546875
  logit[100419]: CUDA=-6.15625, PyTorch=-6.65625, abs_diff=0.5
  logit[6588]: CUDA=0.14941406, PyTorch=-0.2890625, abs_diff=0.43847656
  logit[31267]: CUDA=10.75, PyTorch=10.3125, abs_diff=0.4375
  logit[134408]: CUDA=-5.78125, PyTorch=-6.21875, abs_diff=0.4375
  logit[118577]: CUDA=-8.625, PyTorch=-9.0625, abs_diff=0.4375
  logit[135085]: CUDA=-2.6875, PyTorch=-3.109375, abs_diff=0.421875
  logit[63167]: CUDA=-4.875, PyTorch=-5.28125, abs_diff=0.40625
  logit[17891]: CUDA=-6.25, PyTorch=-6.65625, abs_diff=0.40625
step 42: CUDA token=311, PyTorch token=311
  logits: 1958/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[23809]: CUDA=-10.25, PyTorch=-10.5625, abs_diff=0.3125  logit[32336]: CUDA=-6.78125, PyTorch=-7.0625, abs_diff=0.28125
  logit[2736]: CUDA=1.7109375, PyTorch=1.4453125, abs_diff=0.265625
  logit[51435]: CUDA=-0.55078125, PyTorch=-0.8125, abs_diff=0.26171875  logit[25001]: CUDA=-1.7109375, PyTorch=-1.9609375, abs_diff=0.25  logit[24999]: CUDA=-6.40625, PyTorch=-6.65625, abs_diff=0.25
  logit[22729]: CUDA=-5.34375, PyTorch=-5.59375, abs_diff=0.25
  logit[17638]: CUDA=-6.40625, PyTorch=-6.65625, abs_diff=0.25  logit[32460]: CUDA=-4.90625, PyTorch=-5.15625, abs_diff=0.25

step 43: CUDA token=4211, PyTorch token=4211  logits: 1096/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.296875  logit[126175]: CUDA=-4.21875, PyTorch=-3.921875, abs_diff=0.296875  logit[95972]: CUDA=-0.053222656, PyTorch=-0.3203125, abs_diff=0.26708984
  logit[28580]: CUDA=-4.3125, PyTorch=-4.0625, abs_diff=0.25  logit[68419]: CUDA=-5.1875, PyTorch=-4.9375, abs_diff=0.25  logit[2690]: CUDA=-7.0625, PyTorch=-6.8125, abs_diff=0.25  logit[71142]: CUDA=-5.25, PyTorch=-5, abs_diff=0.25
  logit[35886]: CUDA=3.6875, PyTorch=3.9375, abs_diff=0.25  logit[24369]: CUDA=-1.125, PyTorch=-0.88671875, abs_diff=0.23828125  logit[39613]: CUDA=-2.703125, PyTorch=-2.46875, abs_diff=0.234375
  logit[29146]: CUDA=-4.09375, PyTorch=-3.859375, abs_diff=0.234375
  logits: 3840/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125


  logit[5411]: CUDA=3.796875, PyTorch=4.0625, abs_diff=0.265625  logit[1655]: CUDA=6.40625, PyTorch=6.15625, abs_diff=0.25  logit[23782]: CUDA=7.1875, PyTorch=7.4375, abs_diff=0.25
  logit[69270]: CUDA=-3.21875, PyTorch=-2.96875, abs_diff=0.25  logit[65579]: CUDA=-5.53125, PyTorch=-5.28125, abs_diff=0.25
  logit[72229]: CUDA=-4.09375, PyTorch=-3.859375, abs_diff=0.234375
  logit[31395]: CUDA=1.6953125, PyTorch=1.9296875, abs_diff=0.234375  logit[25992]: CUDA=2.59375, PyTorch=2.828125, abs_diff=0.234375
step 45: CUDA token=32675, PyTorch token=32675  logits: 7258/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.375
  logit[93188]: CUDA=-5.5625, PyTorch=-5.21875, abs_diff=0.34375
  logit[74674]: CUDA=-8.1875, PyTorch=-7.875, abs_diff=0.3125  logit[5102]: CUDA=-8.3125, PyTorch=-8, abs_diff=0.3125
  logit[20534]: CUDA=-8.625, PyTorch=-8.3125, abs_diff=0.3125
  logit[75012]: CUDA=-4.96875, PyTorch=-4.65625, abs_diff=0.3125  logit[76470]: CUDA=-4.9375, PyTorch=-4.625, abs_diff=0.3125
  logit[23923]: CUDA=-10.5, PyTorch=-10.1875, abs_diff=0.3125
  logit[90007]: CUDA=-10.625, PyTorch=-10.3125, abs_diff=0.3125  logit[89746]: CUDA=-4.84375, PyTorch=-4.53125, abs_diff=0.3125
step 46: CUDA token=28611, PyTorch token=990  logits: 1871/151936 outside atol=0.1 rtol=0.02; max_abs_diff=0.3125
  logit[47150]: CUDA=-5.65625, PyTorch=-5.96875, abs_diff=0.3125
  logit[5435]: CUDA=3.0625, PyTorch=3.359375, abs_diff=0.296875  logit[25298]: CUDA=-3.78125, PyTorch=-4.0625, abs_diff=0.28125
  logit[55353]: CUDA=-6.46875, PyTorch=-6.75, abs_diff=0.28125  logit[35617]: CUDA=-4.3125, PyTorch=-4.59375, abs_diff=0.28125
  logit[62939]: CUDA=-4.3125, PyTorch=-4.59375, abs_diff=0.28125
  logit[73193]: CUDA=-7.1875, PyTorch=-7.46875, abs_diff=0.28125  logit[80104]: CUDA=-7.96875, PyTorch=-8.25, abs_diff=0.28125
  logit[67368]: CUDA=-4.71875, PyTorch=-5, abs_diff=0.28125  logit[53742]: CUDA=-5.1875, PyTorch=-5.46875, abs_diff=0.28125
step 47: CUDA token=304, PyTorch token=304  logit comparison skipped: token prefixes have diverged
step 48: CUDA token=279, PyTorch token=28839  logit comparison skipped: token prefixes have diverged
step 49: CUDA token=2114, PyTorch token=374  logit comparison skipped: token prefixes have diverged
step 50: CUDA token=3545, PyTorch token=3545
  logit comparison skipped: token prefixes have divergedstep 51: CUDA token=8471, PyTorch token=17339

step 52: CUDA token=448, PyTorch token=311
step 53: CUDA token=17716, PyTorch token=23579
  logit comparison skipped: token prefixes have divergedstep 54: CUDA token=13, PyTorch token=11  logit comparison skipped: token prefixes have diverged
step 55: CUDA token=15586, PyTorch token=26917  logit comparison skipped: token prefixes have diverged
step 56: CUDA token=287, PyTorch token=11
  logit comparison skipped: token prefixes have divergedstep 57: CUDA token=311, PyTorch token=17716
  logit comparison skipped: token prefixes have diverged




  logit comparison skipped: token prefixes have divergedstep 61: CUDA token=264, PyTorch token=13
step 62: CUDA token=4185, PyTorch token=2055  logit comparison skipped: token prefixes have divergedstep 63: CUDA token=27682, PyTorch token=7196  logit comparison skipped: token prefixes have divergedstep 64: CUDA token=1576, PyTorch token=2494

## 11/25 Results with QWEN3-4b