import json
import os
import re
import time
from pathlib import Path

import gradio as gr
import pandas as pd

ARTIFACTS = Path(__file__).parent / "artifacts"
REMOTE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_json(name, fallback=None):
    path = ARTIFACTS / name
    if not path.exists():
        return fallback
    with open(path) as handle:
        return json.load(handle)


def load_table(name):
    path = ARTIFACTS / name
    if not path.exists():
        return None
    return pd.read_csv(path)


EVIDENCE = load_json("evidence_records.json", {})
CONFIG = load_json("detector_config.json", {})
ZONE_MAP = load_json("zone_map.json", {})
DETECTOR_TABLE = load_table("detector_comparison.csv")
VERIFICATION_TABLE = load_table("verification_summary.csv")

ZONES = CONFIG.get("zones", ["Bedroom", "Bathroom", "Kitchen", "Dining",
                             "Living Room", "Office", "Entrance", "Other"])
BASELINE_DAYS = CONFIG.get("baseline_window_days", 28)

SPECULATIVE_TERMS = [
    "fall", "fell", "fallen", "stroke", "heart", "illness", " ill ", "sick", "injur",
    "unconscious", "emergency", "hospital", "medication", "medicine", "pain", "dementia",
    "confus", "dizzy", "faint", "collapse", "died", "dead", "danger", "depress", "lonely",
    "suffer", "risk of",
]
APPROVED_CLOSINGS = {
    "please check on the resident.",
    "please contact the resident.",
    "a check is recommended.",
    "this is worth checking.",
}
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{2}\b")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
SPECULATION_PATTERN = re.compile(
    r"\b(might|may|could|would|should|probably|possibly|perhaps|likely|maybe|seem|seems|"
    r"appear|appears|suggest|suggests|indicate|indicates|presumably|potentially|suspect|"
    r"assume|hopefully|due to|because of|caused by|imply|implies)\b"
)
ZONE_WORDS = [name.lower() for name in ZONES if name != "Other"]


def evidence_blob(evidence):
    return json.dumps(evidence, default=str)


def grounding_terms(evidence):
    text = " ".join(str(value) for value in evidence.values() if value not in (None, [], ""))
    return set(re.findall(r"[a-z]{5,}", text.lower()))


def allowed_dates(evidence):
    return set(DATE_PATTERN.findall(evidence_blob(evidence)))


def allowed_times(evidence):
    return set(TIME_PATTERN.findall(DATE_PATTERN.sub(" ", evidence_blob(evidence))))


def allowed_numbers(evidence):
    text = TIME_PATTERN.sub(" ", DATE_PATTERN.sub(" ", evidence_blob(evidence)))
    return {round(float(value)) for value in NUMBER_PATTERN.findall(text)}


def has_grounding(sentence, evidence):
    lowered = sentence.lower()
    if any(word in lowered for word in grounding_terms(evidence)):
        return True
    return any(str(number) in sentence for number in allowed_numbers(evidence))


def split_sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def check_sentence(sentence, evidence):
    lowered = " " + sentence.lower().strip() + " "
    if sentence.lower().strip() in APPROVED_CLOSINGS:
        return True, "approved closing"
    for term in SPECULATIVE_TERMS:
        if term in lowered:
            return False, f"unsupported cause or judgement: {term.strip()}"
    hedge = SPECULATION_PATTERN.search(lowered)
    if hedge:
        return False, f"speculation beyond the evidence: {hedge.group(1)}"
    permitted_dates = allowed_dates(evidence)
    for value in DATE_PATTERN.findall(sentence):
        if value not in permitted_dates:
            return False, f"date not present in evidence: {value}"
    undated = DATE_PATTERN.sub(" ", sentence)
    permitted_times = allowed_times(evidence)
    for value in TIME_PATTERN.findall(undated):
        if value not in permitted_times:
            return False, f"time not present in evidence: {value}"
    permitted_numbers = allowed_numbers(evidence)
    for value in NUMBER_PATTERN.findall(TIME_PATTERN.sub(" ", undated)):
        if round(float(value)) not in permitted_numbers:
            return False, f"number not present in evidence: {value}"
    blob = evidence_blob(evidence).lower()
    for zone in ZONE_WORDS:
        if zone in lowered and zone not in blob:
            return False, f"room not present in evidence: {zone}"
    if not has_grounding(sentence, evidence):
        return False, "no fact from the evidence appears in this sentence"
    return True, "supported by evidence"


def verify(message, evidence):
    sentences = split_sentences(message)
    kept, dropped = [], []
    for sentence in sentences:
        supported, reason = check_sentence(sentence, evidence)
        (kept if supported else dropped).append({"sentence": sentence, "reason": reason})
    total = len(sentences) or 1
    return {
        "verified_message": " ".join(item["sentence"] for item in kept),
        "removed": dropped,
        "faithfulness": round(len(kept) / total, 3),
        "sentences": len(sentences),
    }


PRODUCTION_PROMPT = (
    "You write short alerts for the care team of an elderly person living alone. "
    "Use only the facts in the evidence given to you. "
    "Never state or suggest a cause such as a fall, an illness or an injury. "
    "Never invent numbers or times. "
    "Write plainly for a family member. Do not mention statistics, sigma, standard deviations, "
    "confidence or percentages. "
    "Write at most three short sentences. "
    "End with exactly this sentence: Please check on the resident."
)
PERMISSIVE_PROMPT = (
    "You are a helpful assistant for an elderly care team. "
    "Look at the sensor evidence below and explain to the caregiver what happened, "
    "what it probably means, and what they should do. Be warm and reassuring."
)
PROMPTS = {"Production prompt": PRODUCTION_PROMPT, "Permissive prompt": PERMISSIVE_PROMPT}
HIDDEN_FIELDS = {"deviation_sigma", "resident", "known_context"}


def build_user_prompt(evidence):
    lines = [
        f"{key}: {value}" for key, value in evidence.items()
        if key not in HIDDEN_FIELDS and value not in (None, [], "")
    ]
    return "Evidence\n" + "\n".join(lines) + "\n\nWrite the alert."


def read_token():
    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def remote_message(evidence, system_prompt, attempts=2):
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=REMOTE_MODEL, token=read_token(), timeout=60)
    failure = None
    for attempt in range(attempts):
        try:
            reply = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user_prompt(evidence)},
                ],
                max_tokens=180,
                temperature=0.3,
            )
            return reply.choices[0].message.content.strip()
        except Exception as error:
            failure = error
            time.sleep(2 * (attempt + 1))
    raise failure


def local_message(evidence, loose=False):
    parts = [
        f"The {evidence['measure']} was {evidence['observed']} on {evidence['date']}.",
        f"Over the last {evidence.get('baseline_window_days', BASELINE_DAYS)} days "
        f"the usual value is {evidence['usual']}.",
    ]
    if evidence.get("quiet_rooms"):
        parts.append("There was almost no activity in the "
                     + " and the ".join(room.lower() for room in evidence["quiet_rooms"]) + ".")
    if loose:
        parts.append("The resident may have had a fall during the night.")
        parts.append("This pattern often points to an illness starting.")
        parts.append("Movement was around 45 percent below the usual level.")
    parts.append("Please check on the resident.")
    return " ".join(parts)


def generate(day_label, condition):
    key = day_label.split("  ")[0].strip()
    evidence = EVIDENCE.get(key)
    if evidence is None:
        return {}, "", "", "", pd.DataFrame(columns=["removed sentence", "reason"])

    loose = condition == "Permissive prompt"
    try:
        message = remote_message(evidence, PROMPTS[condition])
        source = f"hosted model, {REMOTE_MODEL}"
    except Exception as error:
        message = local_message(evidence, loose)
        source = f"local fallback, hosted model unavailable ({type(error).__name__})"

    outcome = verify(message, evidence)
    removed = pd.DataFrame(
        [[item["sentence"], item["reason"]] for item in outcome["removed"]],
        columns=["removed sentence", "reason"],
    )
    status = (
        f"Faithfulness {outcome['faithfulness']}  |  "
        f"{outcome['sentences'] - len(outcome['removed'])} of {outcome['sentences']} "
        f"sentences kept  |  source: {source}"
    )
    return evidence, message, outcome["verified_message"], status, removed


def alert_choices():
    labels = []
    for key, record in EVIDENCE.items():
        labels.append(f"{key}  {record.get('severity', '')}  {record.get('measure', '')}")
    return sorted(labels)


INTRO = """
# Mimamori AI

Routine monitoring for an elderly person living alone, built on ambient motion sensors.
No camera and no microphone.

The detector finds a day that breaks the resident's own routine and writes a structured
evidence record. A language model turns that record into a readable message. Every sentence
is then checked back against the evidence, and anything the sensors did not show is removed
before the alert is delivered.

Pick a real alert below and compare the two prompts. The production prompt keeps the model
inside the evidence. The permissive prompt is the kind somebody writes when nobody is
enforcing discipline, and it is where the verification layer earns its place.
"""

RESULTS_NOTE = """
### How the detector was tested

Four kinds of anomaly were injected into held out days and each detector was scored on how well
it separated altered days from untouched ones. Recall is measured with the false alarm rate held
at five percent. AUC needs no threshold, where 0.5 is chance and 1.0 is perfect.

### How the verification layer was tested

The same alerts were sent through both prompts. Faithfulness is the share of sentences that
survive checking. After verification it is 1.0 by construction in both cases.
"""

with gr.Blocks(title="Mimamori AI") as demo:
    gr.Markdown(INTRO)

    with gr.Tab("Run an alert"):
        with gr.Row():
            day_input = gr.Dropdown(
                choices=alert_choices(),
                value=(alert_choices() or [None])[0],
                label="Alert day",
                scale=3,
            )
            condition_input = gr.Radio(
                choices=list(PROMPTS.keys()),
                value="Production prompt",
                label="Prompt",
                scale=2,
            )
        run_button = gr.Button("Generate and verify", variant="primary")
        status_output = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=2):
                evidence_output = gr.JSON(label="Evidence record given to the model")
            with gr.Column(scale=3):
                raw_output = gr.Textbox(label="What the model wrote", lines=9)
                verified_output = gr.Textbox(label="What is actually sent", lines=6)
        removed_output = gr.Dataframe(
            headers=["removed sentence", "reason"],
            label="Removed before sending",
            wrap=True,
        )
        run_button.click(
            generate,
            inputs=[day_input, condition_input],
            outputs=[evidence_output, raw_output, verified_output, status_output, removed_output],
        )

    with gr.Tab("Measured results"):
        gr.Markdown(RESULTS_NOTE)
        if DETECTOR_TABLE is not None:
            gr.Dataframe(value=DETECTOR_TABLE, label="Detector comparison", interactive=False)
        if VERIFICATION_TABLE is not None:
            gr.Dataframe(value=VERIFICATION_TABLE, label="Verification by prompt", interactive=False)

    with gr.Tab("Rooms and settings"):
        gr.Markdown(
            "Rooms are not configured by hand. The system counts which labelled activity each "
            "motion sensor sees most often and assigns the room from that."
        )
        if ZONE_MAP:
            rooms = pd.DataFrame(
                sorted(
                    ({"room": zone, "sensors": ", ".join(sorted(s for s, z in ZONE_MAP.items() if z == zone))}
                     for zone in sorted(set(ZONE_MAP.values()))),
                    key=lambda row: row["room"],
                )
            )
            gr.Dataframe(value=rooms, interactive=False, wrap=True)
        gr.JSON(value=CONFIG, label="Detector configuration")

if __name__ == "__main__":
    demo.launch()
