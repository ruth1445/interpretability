# Towards Monosemanticity: Extracting Monosemantic Features from Neural Networks

## Problem

The 'black box' nature of AI simply means that we can feed them inputs and observe outputs, but we don't necessarily understand how they compute.

The deeper problem: **polysemanticity**. When researchers examine individual neurons in neural networks, they find that single neurons respond to completely unrelated concepts. In vision models, a neuron fires for both cat faces AND car fronts. In language models, a neuron responds to academic citations, English dialogue, HTTP requests, AND Korean text simultaneously.

This makes neurons useless as units of analysis. You can't reason about how a network works if you don't understand what individual components are computing.

## Solution

**Sparse Autoencoders (SAEs) for Feature Extraction**

I replicated research on mechanistic interpretability by implementing sparse autoencoders to decompose neural network activations into **monosemantic features**—individual dimensions that each respond to a single, interpretable concept.

### My Approach

1. **Extract Model Activations** — Fed a physics corpus through GPT-2 (Layer 10) and captured internal activation vectors
2. **Train a Sparse Autoencoder** — Built an 8× expansion factor SAE (512 → 4,096 features) trained on 180,233 token activations
3. **Identify and Interpret Features** — Extracted top tokens for each learned feature and labeled those showing coherent concepts

### Key Finding

When you apply SAEs correctly, something remarkable happens: **polysemantic neurons become monosemantic features**. One feature = one concept. Suddenly the network becomes legible.

## Technical Details

**Dataset:** Physics corpus (68,164 words, 87,003 tokens)
- Life 3.0
- The Feynman Lectures on Physics - Volume I
- Quantum Mechanics Lectures - University of Oxford
- Black Hole Information Paradox - MIT Physics
- General Relativity, Black Holes, and Cosmology - University of Colorado

**Model & Scale:**
- Pre-trained GPT-2, analyzing layer 10 activations
- SAE trained on 180,233 token activations
- Expansion factor: 8× (512 to 4,096 features)
- Analysis focused on the strongest, most interpretable features

**Methods:**
- Sparse autoencoder architecture with L1 regularization for sparsity
- Feature activation analysis: examining which inputs cause each feature to fire
- Token attribution: identifying the most representative tokens for each feature
- Manual interpretation of learned feature concepts

## Results

Successfully extracted monosemantic features with clear, interpretable meanings:

- **Feature 8:** Transformation/change verbs ("transformed," "converted," "turned," "into," "collapse")
- **Conditional modality:** Modal verbs ("will," "should," "can," "may," "might")
- **Epistemic verbs:** Verbs of knowing and understanding ("sense," "know," "catch up," "spirit")

Each feature showed:
- Clear activation distributions (sparse, interpretable patterns)
- Coherent semantic meaning across multiple contexts
- Strong token attribution (high cosine similarity scores 0.6-1.0)

## What I Learned

This project taught me three things:

1. **Understanding requires decomposition.** Black boxes only stay black if you don't find the right way to break them down.

2. **Structure inference from incomplete data is possible.** You don't need 8 billion activations like the original paper. With thoughtful design and careful feature analysis, you can extract signal from a smaller, focused dataset.

3. **Engineering means understanding deeply.** As a data scientist, I don't just build models that work. I understand why they work, and I can reason about where they'll fail. That's the difference between coding and engineering.

## Why This Matters

The bigger picture: **models that understand the world through a text-driven lens must be understood through that same lens.** 

Despite being multimodal, a model's main mode of learning is text-based. As models grow more capable, monosemanticity allows us to see their inner workings and guide the concepts they learn. This is foundational for AI safety—if we can understand how large language models actually compute, we can predict their failure modes, catch misalignment, and build systems we can trust.

This work directly complements how I approach data science: every model I build, I want to understand. Not just "it works," but "why does it work, and what could go wrong?"
