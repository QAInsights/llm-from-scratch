---
title: Valluvar or AI?
emoji: 🕉️
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 4.x
app_file: app.py
pinned: false
license: mit
---

# Valluvar or AI? 🕉️

An AI that writes new Thirukkurals in the style of Thiruvalluvar.

## Features

- **Generate Kural**: Enter a Tamil theme and get a bilingual couplet
- **Valluvar or AI Quiz**: Can you tell which is original and which is AI-generated?
- **Temperature Control**: Adjust creativity from coherent (0.5) to wild (2.0)

## Model

- **Architecture**: GPT (8L/8H/512D, 25.4M params)
- **Training Data**: Thirukkural (1330 kurals + English translations)
- **Tokenization**: Character-level

## Examples

**Traditional themes work great:**
- `கடவுள் வாழ்த்து` (Praise of God) ✅
- `அரசியல்` (Politics/Governance) ✅
- `நட்பு` (Friendship) ✅

**Modern topics don't work:**
- `விஞ்ஞானம்` (Science) ❌
- `கணிதம்` (Mathematics) ❌

The model learned Thiruvalluvar's form and traditional themes, but not modern concepts.

## How to Use

1. Enter a Tamil word or theme
2. Adjust temperature (0.8 recommended)
3. Click Generate
4. See if the model memorized or created something new!
