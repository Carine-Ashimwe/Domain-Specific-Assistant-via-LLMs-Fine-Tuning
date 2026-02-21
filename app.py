"""
MCH AI Assistant — Gradio Web Interface
========================================
Domain  : Maternal and Child Health (MCH)
Model   : TinyLlama-1.1B + LoRA fine-tuned adapter (Carinea/MCH_LLM_Fine_Tuning)
Features: Streaming responses, FAQ cache, severity labelling, domain filtering
"""

import os
import json
import shutil
import tempfile
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from huggingface_hub import hf_hub_download
from threading import Thread

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_REPO_ID      = os.environ.get("MODEL_REPO_ID", "Carinea/MCH_LLM_Fine_Tuning")
BASE_MODEL_ID      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device             = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS     = 120      # Lower = faster on CPU
TEMPERATURE        = 0.2      # Low temperature = focused, deterministic answers
TOP_P              = 0.9
REPETITION_PENALTY = 1.15

SYSTEM_PROMPT = (
    "You are a Maternal and Child Health (MCH) assistant. "
    "Answer briefly and clearly in 3-5 sentences. "
    "Only answer about pregnancy, labour, postpartum, newborn care, vaccinations, or child health. "
    "If unsure, advise seeing a health professional. Do not invent facts."
)

# ── aLoRA keys that break standard PEFT LoraConfig ────────────────────────────
ALORA_KEYS = {
    "alora_invocation_tokens", "alora_invocation_token",
    "alora_rank_pattern", "alora_alpha_pattern",
}

# ── Domain keyword filter (fast string matching before model inference) ────────
MCH_KEYWORDS = [
    "pregnan", "antenatal", "prenatal", "labour", "labor", "delivery",
    "postpartum", "newborn", "neonate", "infant", "child", "baby",
    "breastfeed", "lactation", "immuniz", "vaccin", "preeclampsia",
    "hemorrhage", "bleeding", "danger", "fever", "diarrhea", "diarrhoea",
    "malaria", "nutrition", "family planning", "contraception", "miscarriage",
    "cesarean", "c-section", "midwife", "mch", "maternal", "birth",
    "jaundice", "dehydration", "obstetric", "neonatal", "placenta",
    "postnatal", "amniotic", "fundus",
]

# ── FAQ Cache: instant answers for common queries (no model inference) ─────────
FAQ_CACHE = {
    "danger signs pregnancy": (
        "⚠️ **Danger signs in pregnancy:** vaginal bleeding; severe abdominal pain; "
        "severe headache or blurred vision; swelling of face/hands; fever; convulsions; "
        "reduced fetal movements; leaking fluid before labour. Seek urgent care immediately."
    ),
    "danger signs labour": (
        "🔴 **Danger signs in labour:** heavy bleeding; prolonged labour; severe headache "
        "or convulsions; fever or foul-smelling discharge; fetal distress; prolapsed cord. "
        "Seek urgent care immediately."
    ),
    "danger signs newborn": (
        "🔴 **Newborn danger signs:** poor feeding; fast/difficult breathing; fever or low "
        "temperature; convulsions; lethargy; early jaundice; umbilical redness/pus; severe "
        "diarrhoea. Seek urgent care immediately."
    ),
    "bcg vaccine": (
        "BCG is given **at birth** or at first contact (up to 12 months if unvaccinated). "
        "A small scar develops at the injection site within 2-6 weeks — this is normal."
    ),
    "vaccination schedule": (
        "WHO infant schedule: **BCG** at birth; **OPV + DPT-HepB-Hib + PCV** at 6, 10, and "
        "14 weeks; **Measles-Rubella** at 9 months and again at 15-18 months. Check your "
        "national schedule for any local additions."
    ),
    "breastfeed": (
        "Newborns should be breastfed **8-12 times per 24 hours** (every 2-3 hours), on demand. "
        "Allow the baby to empty one breast fully before offering the other."
    ),
    "how often breastfeed": (
        "Breastfeed **on demand**, at least 8-12 times in 24 hours. Watch for early hunger cues "
        "like rooting and hand-to-mouth movements — do not wait for crying."
    ),
    "postpartum depression": (
        "Signs: persistent sadness, loss of interest, sleep problems, appetite changes, guilt, "
        "hopelessness lasting more than 2 weeks. Assess at every postnatal visit and refer to "
        "mental health services if present."
    ),
    "antenatal care": (
        "WHO recommends **at least 8 ANC contacts**: monthly until 28 weeks, fortnightly to "
        "36 weeks, then weekly. More frequent visits are needed if complications arise."
    ),
    "weight gain pregnancy": (
        "Recommended total weight gain: **9-14 kg** (~300 extra kcal/day). Varies by "
        "pre-pregnancy BMI. Gain of more than 2 kg/month after 20 weeks should be investigated."
    ),
    "postnatal care schedule": (
        "Postnatal contacts: **24 hours, 3 days, 1-2 weeks, and 6 weeks** after delivery. "
        "Assess for bleeding, infection, breastfeeding, newborn danger signs, and maternal mental "
        "health at each visit."
    ),
    "stages of labour": (
        "Labour has 3 stages: (1) Cervical dilation 0-10 cm; (2) Pushing — 10 cm to baby "
        "delivery; (3) Placental stage — delivery of the placenta, normally within 30 minutes."
    ),
    "preeclampsia": (
        "Preeclampsia signs: BP above 140/90 mmHg, protein in urine, severe headache, visual "
        "changes, upper right abdominal pain, facial/hand swelling. Check BP and urine protein. "
        "Refer immediately if suspected."
    ),
    "normal bleeding after delivery": (
        "Lochia (normal postpartum bleeding): heavy/bright red for 3 days, darker and moderate "
        "days 4-10, then scanty for up to 6 weeks. Soaking more than 1 pad/hour is excessive "
        "— refer urgently."
    ),
    "dehydration child": (
        "Mild-moderate dehydration: give ORS 50-100 mL/kg over 3-4 hours. Danger signs: sunken "
        "eyes, very dry mouth, skin pinch returns slowly, unable to drink, high fever. "
        "These require urgent referral."
    ),
    "family planning after delivery": (
        "Progestin-only pills or copper IUD can start at 6 weeks postpartum (safe while "
        "breastfeeding). Combined hormonal methods after 6 months. Condoms can be used "
        "immediately. Discuss preference with the mother."
    ),
}

# ── UI description with embedded user instructions ────────────────────────────
UI_DESCRIPTION = """
**How to use this assistant:**
1. Type your MCH question in the box below and press **Enter** or click **Submit**
2. Use the example questions (click any) to get started quickly
3. Response labels: 🟢 Standard info | 🟡 Monitor closely | 🔴 Seek urgent care

**Topics covered:** Pregnancy · Labour · Postpartum Care · Newborn Care · Breastfeeding · Vaccinations · Child Illness · Family Planning

---
⚠️ *This assistant provides WHO guideline-based information only. Always consult a qualified health professional for clinical decisions.*
"""

# ── Adapter loading: patch adapter_config.json to remove aLoRA keys ───────────
def load_adapter_safely(base_model, repo_id: str) -> PeftModel:
    """
    Download adapter_config.json from HuggingFace, strip unsupported aLoRA keys,
    copy weights to a temp directory, and load PeftModel from there.
    This avoids calling PeftConfig.from_pretrained(repo_id) which crashes on aLoRA configs.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        # 1. Download and sanitise adapter config
        config_path = hf_hub_download(repo_id=repo_id, filename="adapter_config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)
        cleaned = {k: v for k, v in cfg.items() if k not in ALORA_KEYS}
        cleaned["peft_type"] = "LORA"
        with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
            json.dump(cleaned, f, indent=2)

        # 2. Download adapter weights (safetensors preferred, bin as fallback)
        weight_copied = False
        for fname in ["adapter_model.safetensors", "adapter_model.bin"]:
            try:
                w_path = hf_hub_download(repo_id=repo_id, filename=fname)
                shutil.copy(w_path, os.path.join(tmp_dir, fname))
                weight_copied = True
                print(f"  Adapter weights: {fname}")
                break
            except Exception:
                continue
        if not weight_copied:
            raise FileNotFoundError("No adapter weights (safetensors or bin) found in repo.")

        # 3. Load PeftModel from the sanitised temp directory
        return PeftModel.from_pretrained(base_model, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Model loading ──────────────────────────────────────────────────────────────
print(f"Loading model on {device}...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True,
)

try:
    model = load_adapter_safely(model, MODEL_REPO_ID)
    print("LoRA adapter loaded.")
    try:
        # Merge adapter weights into base model — removes per-layer LoRA overhead
        model = model.merge_and_unload()
        print("Adapter merged for faster inference.")
    except Exception as me:
        print(f"Merge skipped ({me}).")
except Exception as e:
    print(f"Adapter load failed: {e} — running base model only.")

if device == "cuda":
    model = model.to(device)
    torch.cuda.empty_cache()

# torch.compile reduces CPU inference overhead (PyTorch 2.x)
if device == "cpu":
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile.")
    except Exception:
        pass

model.eval()
print("Model ready.")


# ── Helper functions ───────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    """Lowercase, strip, and collapse whitespace."""
    return " ".join((text or "").lower().strip().split())

def _is_greeting(text: str) -> bool:
    """Return True if the message is a simple greeting."""
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
    t = _normalize(text)
    return t in greetings or any(t.startswith(g + " ") for g in greetings)

def _is_gratitude(text: str) -> bool:
    """Return True if the message expresses thanks."""
    thanks = {"thanks", "thank you", "thx", "thank you so much", "much appreciated"}
    t = _normalize(text)
    return t in thanks or any(t.startswith(tk) for tk in thanks)

def _is_domain_related(text: str) -> bool:
    """Return True if the message contains at least one MCH keyword."""
    t = _normalize(text)
    return any(k in t for k in MCH_KEYWORDS)

def _faq_lookup(text: str):
    """Return a cached FAQ answer if all words in a key appear in the query."""
    t = _normalize(text)
    for key, answer in FAQ_CACHE.items():
        if all(word in t for word in key.split()):
            return answer
    return None

def _extract_severity(text: str) -> int:
    """Classify response urgency: 0 = standard, 1 = monitor, 2 = urgent."""
    t = text.upper()[:300]
    if any(w in t for w in ["URGENT", "URGENTLY", "IMMEDIATELY", "DANGER", "EMERGENCY", "REFER IMMEDIATELY"]):
        return 2
    if any(w in t for w in ["REFER", "MONITOR", "SERIOUS", "CONTACT"]):
        return 1
    return 0

def _clean_response(raw: str) -> str:
    """Strip prompt artifacts and truncate to last complete sentence."""
    for marker in ["<|assistant|>", "ASSISTANT:", "[/INST]"]:
        if marker in raw:
            raw = raw.split(marker)[-1]
    for stop in ["<|user|>", "USER:", "[INST]"]:
        if stop in raw:
            raw = raw.split(stop)[0]
    raw = raw.strip()
    if len(raw) > 350 and "." in raw:
        raw = raw[:400].rsplit(".", 1)[0] + "."
    return raw.strip() or "Unable to generate a response."


# ── Streaming generation ───────────────────────────────────────────────────────
def _generate_streaming(message: str):
    """
    Generate a response token-by-token using TextIteratorStreamer.
    Yields partial text so Gradio renders words as they arrive (real-time effect).
    Generation runs in a background thread to avoid blocking the streamer iterator.
    """
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{message.strip()}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    collected = ""
    for new_text in streamer:
        if any(stop in new_text for stop in ["<|user|>", "[INST]", "USER:"]):
            break
        collected += new_text
        yield collected   # Gradio updates the UI on each yield

    thread.join()

    # Final cleaned response with severity label
    final = _clean_response(collected)
    sev = _extract_severity(final)
    labels = [
        "🟢 *Standard response*",
        "🟡 *Important — monitor closely*",
        "🔴 *URGENT — refer/seek care immediately*",
    ]
    yield f"{final}\n\n{labels[sev]}\n*Always consult a qualified health professional.*"


# ── Main chat handler ──────────────────────────────────────────────────────────
def chat_response(message: str, history):
    """
    Main Gradio chat function. Handles greetings, gratitude, domain filtering,
    FAQ cache lookup, and model inference (with streaming).
    """
    if not message or not message.strip():
        yield "Please type a question about maternal or child health."
        return

    # Instant responses — no model needed
    if _is_greeting(message):
        yield (
            "👋 Hello! I'm your **Maternal & Child Health (MCH) Assistant**.\n\n"
            "I can answer questions about:\n"
            "- 🤰 Pregnancy & antenatal care\n"
            "- 🏥 Labour & delivery\n"
            "- 👶 Newborn care & danger signs\n"
            "- 🍼 Breastfeeding\n"
            "- 💉 Vaccinations\n"
            "- 👨‍👩‍👧 Child illness & family planning\n\n"
            "What would you like to know?"
        )
        return

    if _is_gratitude(message):
        yield "You're welcome! Feel free to ask any more MCH questions. 😊"
        return

    if not _is_domain_related(message):
        yield (
            "ℹ️ I specialise in **maternal and child health** only.\n\n"
            "Please ask about pregnancy, labour, postpartum care, newborn care, "
            "breastfeeding, vaccinations, family planning, or common child illnesses.\n\n"
            "💡 *Try one of the example questions below to get started.*"
        )
        return

    # FAQ cache — instant guideline-based answers
    faq = _faq_lookup(message)
    if faq:
        yield faq + "\n\n*Guideline-based answer | Always consult a qualified health professional.*"
        return

    # Model inference with streaming
    yield from _generate_streaming(message)


# ── Gradio UI ──────────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=chat_response,
    title="🏥 Maternal & Child Health (MCH) AI Assistant",
    description=UI_DESCRIPTION,
    examples=[
        "What are danger signs in pregnancy?",
        "When should the BCG vaccine be given?",
        "What are danger signs in a newborn?",
        "How often should a newborn be breastfed?",
        "What are signs of postpartum depression?",
        "What is the infant vaccination schedule?",
        "How do I manage fever in a 1-year-old?",
        "What are danger signs in labour?",
        "What is normal bleeding after delivery?",
        "When can family planning start after delivery?",
    ],
    theme=gr.themes.Soft(),
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
