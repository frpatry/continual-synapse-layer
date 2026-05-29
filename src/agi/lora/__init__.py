"""LoRA distillation training for the AGI architecture (Phase 2e).

Trains a small ``rank=8`` LoRA adapter on Qwen2.5-1.5B-Instruct so
the *raw* model — without the metacog scaffolding loop — mimics
the metacog-protected pipeline's responses. The strategy is
**distillation via supervised fine-tuning**, not RL:

- **Teacher** = Qwen + full metacog scaffolding
  (memory retrieve → pre-evaluate → if admit_ignorance: use
  template; else: generate with facts in context; post-evaluate
  and override if hallucinated)
- **Student** = Qwen + LoRA adapter, generates autoregressively
  from the same ``(query, facts)`` input the teacher saw

The student learns the *behaviour* (when to defer, when to
quote facts, how to format) without needing the full metacog
loop at inference time. LoRA modifications stay bounded
(≤ 3M params on rank-8 q_proj + v_proj only).

The actual GPU training happens in
``colab/phase_2e_lora_training.ipynb``. The local code in this
package is responsible for:

- Defining the teacher pipeline (:mod:`.distillation`).
- Generating diverse training queries + running the teacher to
  produce ``(prompt, target)`` pairs (:mod:`.training_data`).
- The LoRA trainer abstraction (:mod:`.lora_trainer`) — peft is
  a lazy import inside the training entry points so unit tests
  don't need it locally.
"""
