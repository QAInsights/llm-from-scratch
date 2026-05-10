"""Gradio app for Thirukkural GPT - Valluvar or AI."""
import random
import re

import gradio as gr
import torch

from model import GPT, GPTConfig


def load_model():
    """Load the trained model and tokenizer."""
    # Allow GPTConfig for safe loading
    from model import GPTConfig
    torch.serialization.add_safe_globals([GPTConfig])
    checkpoint = torch.load("checkpoint_final.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, stoi, itos


def generate(model, prompt, stoi, itos, max_new_tokens=200, temperature=0.8, device="cpu"):
    """Generate text from prompt."""
    model = model.to(device)

    # Encode prompt
    prompt_tokens = [stoi.get(c, stoi.get(" ", 0)) for c in prompt]
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx[:, -model.config.block_size :]

            # Get predictions
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Sample
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat((idx, idx_next), dim=1)

    # Decode
    tokens = idx[0].tolist()
    result = "".join([itos.get(t, "") for t in tokens])
    return result


def is_real_kural(text, original_text):
    """Check if generated text exists in original kurals.
    
    A kural is considered "real" if:
    1. The Tamil couplet (2 lines) exists in original
    2. The English translation matches
    """
    lines = text.strip().split("\n")
    
    # Get Tamil lines (contain Tamil Unicode)
    tamil_lines = [l.strip() for l in lines if re.search(r"[\u0B80-\u0BFF]", l)]
    # Get English lines (no Tamil, just text)
    english_lines = [l.strip() for l in lines if l.strip() and not re.search(r"[\u0B80-\u0BFF]", l)]
    
    if len(tamil_lines) < 2:
        return False
    
    # Check if Tamil couplet exists in original
    first_tamil = tamil_lines[0]
    second_tamil = tamil_lines[1] if len(tamil_lines) > 1 else ""
    
    # A true kural needs both Tamil lines to exist consecutively
    tamil_couplet = first_tamil + "\n" + second_tamil
    if tamil_couplet not in original_text:
        return False
    
    # Also check that English lines roughly match (at least one should exist)
    if english_lines:
        first_english = english_lines[0]
        # Check if this English translation exists near the Tamil
        return first_english in original_text
    
    return True


# Load model and data
print("Loading model...")
model, stoi, itos = load_model()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

# Load original text for verification
with open("thirukkural_clean.txt", "r", encoding="utf-8") as f:
    ORIGINAL_TEXT = f.read()


def generate_kural(prompt, temperature, max_tokens):
    """Generate and format kural with proper structure."""
    # Generate with higher token count to ensure complete kural
    output_raw = generate(model, prompt, stoi, itos, int(max_tokens) + 100, temperature)
    
    # Extract first complete kural from generated text
    lines = output_raw.strip().split("\n")
    
    # Find the first proper kural (skip headers, get 2 Tamil + 2 English lines)
    tamil_lines = []
    english_lines = []
    
    for line in lines:
        line = line.strip()
        if not line or " - " in line:
            continue
        # Skip short Tamil headers (1-2 words)
        if re.search(r"[\u0B80-\u0BFF]", line) and len(line.split()) <= 2 and not re.search(r"[a-zA-Z]", line):
            continue
            
        if re.search(r"[\u0B80-\u0BFF]", line):
            if len(tamil_lines) < 2:
                tamil_lines.append(line)
        elif line and len(english_lines) < 2:
            english_lines.append(line)
    
    # Build formatted output
    formatted_lines = []
    if tamil_lines:
        formatted_lines.extend(tamil_lines[:2])
    if english_lines:
        formatted_lines.extend(english_lines[:2])
    
    output = "\n".join(formatted_lines) if formatted_lines else format_kural(output_raw)

    # Check if real or AI
    is_real = is_real_kural(output_raw, ORIGINAL_TEXT)
    source = "📖 Original Thirukkural" if is_real else "🤖 AI Generated"

    return output, source


def format_kural(text):
    """Format kural text with proper structure (2 Tamil + 2 English lines)."""
    lines = text.strip().split("\n")
    
    # Skip headers: lines with " - " OR short single Tamil words (chapter names)
    def is_header(line):
        # Headers have " - " or are short Tamil-only phrases (1-3 words)
        if " - " in line:
            return True
        # Check if it's a short Tamil phrase (likely a chapter title)
        if re.search(r"[\u0B80-\u0BFF]", line) and len(line.split()) <= 3:
            # And no English words
            if not re.search(r"[a-zA-Z]", line):
                return True
        return False
    
    content_lines = [l.strip() for l in lines if l.strip() and not is_header(l)]
    
    # Classify lines
    tamil_lines = [l for l in content_lines if re.search(r"[\u0B80-\u0BFF]", l)]
    english_lines = [l for l in content_lines if l and not re.search(r"[\u0B80-\u0BFF]", l)]
    
    # Build proper 4-line kural
    formatted = []
    
    # Tamil couplet (2 lines)
    if len(tamil_lines) >= 2:
        formatted.extend(tamil_lines[:2])
    elif len(tamil_lines) == 1:
        formatted.append(tamil_lines[0])
        formatted.append("")  # Placeholder
    
    # English translation (2 lines)
    if len(english_lines) >= 2:
        formatted.extend(english_lines[:2])
    elif len(english_lines) == 1:
        formatted.append(english_lines[0])
        formatted.append("")
    
    return "\n".join(formatted)


def valluvar_or_ai_quiz():
    """Generate a quiz: one real, one AI."""
    # Get random real kural - find a proper 4-line kural
    lines = ORIGINAL_TEXT.strip().split("\n")
    
    # Find a random valid kural (2 Tamil + 2 English lines)
    attempts = 0
    real_kural = ""
    while attempts < 100:
        idx = random.randint(0, len(lines) - 4)
        chunk = lines[idx:idx+4]
        tamil_count = sum(1 for l in chunk if re.search(r"[\u0B80-\u0BFF]", l))
        english_count = sum(1 for l in chunk if l.strip() and not re.search(r"[\u0B80-\u0BFF]", l))
        if tamil_count == 2 and english_count == 2:
            real_kural = "\n".join(chunk).strip()
            break
        attempts += 1
    
    # Fallback if no proper kural found
    if not real_kural:
        real_kural = "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு\n'A' leads letters; the Ancient Lord\nLeads and lords the entire world"

    # Generate AI kural with random prompt
    prompts = ["கடவுள் வாழ்த்து", "நட்பு", "அறன்", "வான் சிறப்பு", "அரசியல்"]
    prompt = random.choice(prompts)
    ai_kural_raw = generate(model, prompt, stoi, itos, 150, 0.8)
    ai_kural = format_kural(ai_kural_raw)

    # Format real kural too
    real_kural = format_kural(real_kural)

    # Shuffle
    kurals = [("A", real_kural, True), ("B", ai_kural, False)]
    random.shuffle(kurals)

    return (
        f"## Option A\n```\n{kurals[0][1]}\n```\n\n---\n\n## Option B\n```\n{kurals[1][1]}\n```",
        kurals[0][2],
        kurals[1][2],
        "A" if kurals[0][2] else "B",
    )


# Gradio Interface
with gr.Blocks(title="Valluvar or AI?") as demo:
    gr.Markdown("# 🕉️ Valluvar or AI?")
    gr.Markdown(
        "An AI that writes new Thirukkurals in the style of Thiruvalluvar. "
        "Enter a Tamil theme to generate bilingual wisdom."
    )

    with gr.Tab("✨ Generate Kural"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Theme (Tamil)",
                    placeholder="e.g., கடவுள் வாழ்த்து, நட்பு, அரசியல்",
                    value="கடவுள் வாழ்த்து",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature (Creativity)",
                )
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=400,
                    value=200,
                    step=50,
                    label="Max Tokens",
                )
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                output = gr.Textbox(
                    label="Generated Kural",
                    lines=10,
                )
                source = gr.Textbox(label="Source")

        generate_btn.click(
            fn=generate_kural,
            inputs=[prompt, temperature, max_tokens],
            outputs=[output, source],
        )

        # Quick theme buttons
        gr.Markdown("### Quick Themes")
        with gr.Row():
            themes = [
                "கடவுள் வாழ்த்து",
                "வான் சிறப்பு",
                "நட்பு",
                "அரசியல்",
                "அறன் வலியுறுத்தல்",
            ]
            for theme in themes:
                btn = gr.Button(theme, size="sm")
                btn.click(lambda t=theme: t, outputs=prompt)

    with gr.Tab("🎯 Valluvar or AI? Quiz"):
        gr.Markdown("Can you tell which is the original Thirukkural and which is AI-generated?")

        quiz_output = gr.Markdown()
        with gr.Row():
            guess_a = gr.Button("Option A is Real", variant="secondary")
            guess_b = gr.Button("Option B is Real", variant="secondary")
        quiz_result = gr.Markdown()
        new_quiz_btn = gr.Button("New Quiz", variant="primary")

        # Store answers
        a_is_real = gr.State()
        b_is_real = gr.State()
        correct_answer = gr.State()

        def check_answer(guess, a_real, b_real, correct):
            if guess == correct:
                return "✅ Correct! You identified the original Thirukkural."
            return "❌ Wrong! The original Thirukkural was: " + correct

        new_quiz_btn.click(
            fn=valluvar_or_ai_quiz,
            outputs=[quiz_output, a_is_real, b_is_real, correct_answer],
        )

        guess_a.click(
            fn=lambda a, b, c: check_answer("A", a, b, c),
            inputs=[a_is_real, b_is_real, correct_answer],
            outputs=quiz_result,
        )

        guess_b.click(
            fn=lambda a, b, c: check_answer("B", a, b, c),
            inputs=[a_is_real, b_is_real, correct_answer],
            outputs=quiz_result,
        )

    with gr.Tab("📊 About"):
        gr.Markdown(
            f"""
            ## Model Details

            - **Architecture:** GPT ({model.config.n_layer}L/{model.config.n_head}H/{model.config.n_embd}D)
            - **Parameters:** {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M
            - **Vocabulary:** {len(stoi)} characters (Tamil + English)
            - **Training Data:** Thirukkural (1330 kurals with English translations)
            - **Tokenization:** Character-level

            ## Training

            - Steps: 10,000
            - Device: Apple MPS (Mac Mini)
            - Time: ~5 hours
            - Final Loss: ~1.5

            ## Capabilities

            - ✅ Generate authentic Tamil couplets (2 lines × 4 words)
            - ✅ Produce coherent English translations
            - ✅ Handle traditional themes (virtue, politics, love)
            - ❌ Modern topics (science, technology) - not in training data

            ## Examples of AI vs Original

            The model sometimes generates exact memorized kurals from the 1330,
            and sometimes creates entirely new ones in Thiruvalluvar's style.

            Built with ❤️ using PyTorch and Gradio.
            """
        )


if __name__ == "__main__":
    demo.launch()
