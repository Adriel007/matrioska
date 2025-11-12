#CONFIGS:

## Base Model
* **Model:** `mistralai/Mistral-7B-Instruct-v0.3`
* **Size:** 7 Billion (7B) parameters
* **Type:** Causal Language Model (CausalLM)

## Quantization Configuration (4-bit)
* **Library:** `BitsAndBytes`
* **Load:** `load_in_4bit = True`
* **Quant Type:** `bnb_4bit_quant_type = "nf4"`
* **Double Quant:** `bnb_4bit_use_double_quant = True`
* **Compute Type:** `bnb_4bit_compute_dtype = torch.float16`

## Generation Parameters (Inference)
* **Max New Tokens:** `max_tokens = 4000`
* **Sampling:** `do_sample = True`
* **Temperature:** `temperature = 0.3`
* **Top-P (Nucleus Sampling):** `top_p = 0.85`
