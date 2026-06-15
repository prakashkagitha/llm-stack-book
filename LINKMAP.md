# LINKMAP — Cross-Reference Map for *The LLM Stack*

When you reference another chapter, link to it like: `[Title](../<part-dir>/<file>.html)`.
Every chapter HTML file lives one directory deep, so the prefix is always `../`.
Below is the full table of contents with the exact link target for every chapter.

## Front Matter
- **Preface: How to Read This Book & The Map of the Stack** — `[Preface: How to Read This Book & The Map of the Stack](../00-frontmatter/00-preface.html)`

## Part I — Mathematical & Systems Foundations
- 1.1 **Linear Algebra for Deep Learning** — `[Linear Algebra for Deep Learning](../01-foundations/01-linear-algebra.html)`
- 1.2 **Probability, Statistics & Information Theory** — `[Probability, Statistics & Information Theory](../01-foundations/02-probability-information.html)`
- 1.3 **Calculus, Optimization & Convexity** — `[Calculus, Optimization & Convexity](../01-foundations/03-calculus-optimization.html)`
- 1.4 **Numerical Computing, Floating Point & Precision** — `[Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html)`
- 1.5 **Machine Learning Fundamentals** — `[Machine Learning Fundamentals](../01-foundations/05-ml-fundamentals.html)`
- 1.6 **Neural Networks From Scratch: MLPs & Backprop** — `[Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html)`
- 1.7 **Automatic Differentiation & PyTorch Internals** — `[Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html)`
- 1.8 **GPU Architecture & The Memory Hierarchy** — `[GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html)`
- 1.9 **Parallel Computing & Collective Communication** — `[Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html)`
- 1.10 **The Accelerator Landscape: TPUs, Trainium, AMD/ROCm & Gaudi** — `[The Accelerator Landscape: TPUs, Trainium, AMD/ROCm & Gaudi](../01-foundations/10-accelerator-landscape.html)`

## Part II — The Transformer Architecture
- 2.1 **Tokenization: BPE, WordPiece, Unigram & Byte-Level** — `[Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html)`
- 2.2 **Embeddings & The Input Pipeline** — `[Embeddings & The Input Pipeline](../02-transformer/02-embeddings-input.html)`
- 2.3 **The Attention Mechanism From Scratch** — `[The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html)`
- 2.4 **Multi-Head Attention, MQA, GQA & MLA** — `[Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html)`
- 2.5 **Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi** — `[Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html)`
- 2.6 **The Transformer Block: Norms, Residuals, MLPs & Activations** — `[The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html)`
- 2.7 **Building a GPT From Scratch (nanoGPT-style)** — `[Building a GPT From Scratch (nanoGPT-style)](../02-transformer/07-build-gpt-from-scratch.html)`
- 2.8 **Architecture Variants: Encoder-Decoder, Decoder-Only & Prefix-LM** — `[Architecture Variants: Encoder-Decoder, Decoder-Only & Prefix-LM](../02-transformer/08-architecture-variants.html)`
- 2.9 **Mixture-of-Experts (MoE) Architectures** — `[Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html)`
- 2.10 **Modern Architecture Improvements & Design Choices** — `[Modern Architecture Improvements & Design Choices](../02-transformer/10-modern-arch-improvements.html)`
- 2.11 **Beyond Attention: SSMs, Mamba, RWKV & Linear Attention** — `[Beyond Attention: SSMs, Mamba, RWKV & Linear Attention](../02-transformer/11-ssm-and-alternatives.html)`
- 2.12 **Diffusion & Non-Autoregressive Language Models** — `[Diffusion & Non-Autoregressive Language Models](../02-transformer/12-diffusion-nonAR-lms.html)`

## Part III — Pretraining at Scale
- 3.1 **Pretraining Data: Sources, Crawling & The Data Pipeline** — `[Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html)`
- 3.2 **Data Cleaning, Deduplication & Quality Filtering** — `[Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html)`
- 3.3 **The Pretraining Objective & Loss** — `[The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html)`
- 3.4 **Scaling Laws: Kaplan, Chinchilla & Beyond** — `[Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)`
- 3.5 **Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP** — `[Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html)`
- 3.6 **Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism** — `[Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html)`
- 3.7 **Megatron-LM, DeepSpeed & Parallelism in Practice** — `[Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html)`
- 3.8 **Mixed Precision, bf16 & FP8 Training** — `[Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)`
- 3.9 **Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo** — `[Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html)`
- 3.10 **Learning Rate Schedules, Warmup, Batch Size & Hyperparameters** — `[Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html)`
- 3.11 **Training Stability, Loss Spikes & Debugging Large Runs** — `[Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html)`
- 3.12 **Checkpointing, Fault Tolerance & Long-Running Jobs** — `[Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html)`
- 3.13 **Long-Context Pretraining & Context Extension** — `[Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html)`
- 3.14 **Data Mixing, Domain Weighting & Curriculum** — `[Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html)`
- 3.15 **Synthetic Data for Pre- and Post-Training** — `[Synthetic Data for Pre- and Post-Training](../03-pretraining/15-synthetic-data.html)`
- 3.16 **Continual & Domain-Adaptive Pretraining** — `[Continual & Domain-Adaptive Pretraining](../03-pretraining/16-continual-pretraining.html)`

## Part IV — Kernels, Efficiency & Quantization
- 4.1 **The Roofline Model & Performance Engineering** — `[The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)`
- 4.2 **FlashAttention I: IO-Awareness & The Online Softmax** — `[FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html)`
- 4.3 **FlashAttention 2 & 3: Work Partitioning, Warp Specialization & FP8** — `[FlashAttention 2 & 3: Work Partitioning, Warp Specialization & FP8](../04-kernels-efficiency/03-flash-attention-2-3.html)`
- 4.4 **Writing GPU Kernels with Triton** — `[Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html)`
- 4.5 **CUDA Programming Essentials for ML Engineers** — `[CUDA Programming Essentials for ML Engineers](../04-kernels-efficiency/05-cuda-essentials.html)`
- 4.6 **PagedAttention & KV-Cache Memory Management** — `[PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html)`
- 4.7 **Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)** — `[Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html)`
- 4.8 **Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT** — `[Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html)`
- 4.9 **Kernel Fusion, torch.compile, CUDA Graphs & Compilers** — `[Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html)`
- 4.10 **Memory-Efficient Training: Checkpointing, Offloading & LoRA Math** — `[Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html)`

## Part V — Post-Training & Alignment
- 5.1 **Supervised Fine-Tuning & Instruction Tuning** — `[Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html)`
- 5.2 **Chat Templates, Data Formatting & Sequence Packing** — `[Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html)`
- 5.3 **PEFT I: LoRA, QLoRA, DoRA & The Adapter Family** — `[PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html)`
- 5.4 **PEFT II: Prompt/Prefix Tuning, IA3, Model Merging & Soups** — `[PEFT II: Prompt/Prefix Tuning, IA3, Model Merging & Soups](../05-posttraining-alignment/04-peft-prompt-merging.html)`
- 5.5 **The RLHF Pipeline & Reward Modeling** — `[The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html)`
- 5.6 **Policy Gradients & PPO for Language Models** — `[Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html)`
- 5.7 **Direct Preference Optimization & Its Variants** — `[Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html)`
- 5.8 **GRPO, RLOO & Critic-Free RL** — `[GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html)`
- 5.9 **RL with Verifiable Rewards (RLVR) & The Reasoning Recipe** — `[RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html)`
- 5.10 **Reasoning, Chain-of-Thought & Test-Time Compute** — `[Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html)`
- 5.11 **Constitutional AI, RLAIF & Self-Improvement** — `[Constitutional AI, RLAIF & Self-Improvement](../05-posttraining-alignment/11-constitutional-rlaif.html)`
- 5.12 **Distillation, Model Compression & Knowledge Transfer** — `[Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html)`
- 5.13 **Reward Hacking, Over-Optimization & Alignment Failures** — `[Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html)`

## Part VI — RL Infrastructure (Deep Dive)
- 6.1 **The Anatomy of an RL-for-LLM System** — `[The Anatomy of an RL-for-LLM System](../06-rl-infra/01-anatomy-rl-system.html)`
- 6.2 **The Generation–Training Loop & Rollout Engines** — `[The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html)`
- 6.3 **TRL: HuggingFace's RL Library** — `[TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html)`
- 6.4 **veRL: HybridFlow & The Single-Controller Architecture** — `[veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html)`
- 6.5 **OpenRLHF, NeMo-Aligner & Ray-Based Systems** — `[OpenRLHF, NeMo-Aligner & Ray-Based Systems](../06-rl-infra/05-openrlhf-nemo-ray.html)`
- 6.6 **Prime-RL, Async RL & Decentralized Training** — `[Prime-RL, Async RL & Decentralized Training](../06-rl-infra/06-prime-rl-async.html)`
- 6.7 **Colocated vs Disaggregated RL & Weight Synchronization** — `[Colocated vs Disaggregated RL & Weight Synchronization](../06-rl-infra/07-colocated-vs-disaggregated.html)`
- 6.8 **Reward Engineering, Verifiers & Sandboxes** — `[Reward Engineering, Verifiers & Sandboxes](../06-rl-infra/08-reward-verifiers-sandboxes.html)`
- 6.9 **Advantage Estimation, KL Control & Stability Tricks** — `[Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html)`
- 6.10 **Agentic & Multi-Turn RL** — `[Agentic & Multi-Turn RL](../06-rl-infra/10-agentic-multiturn-rl.html)`
- 6.11 **Scaling RL: Throughput, Load Balancing & The Latest Tricks** — `[Scaling RL: Throughput, Load Balancing & The Latest Tricks](../06-rl-infra/11-scaling-rl-tricks.html)`
- 6.12 **RL Data, Curriculum & Replay Management** — `[RL Data, Curriculum & Replay Management](../06-rl-infra/12-rl-data-curriculum-replay.html)`

## Part VII — Inference & Serving
- 7.1 **The Anatomy of LLM Inference: Prefill, Decode & The KV Cache** — `[The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html)`
- 7.2 **Continuous Batching & Request Scheduling** — `[Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)`
- 7.3 **vLLM: Architecture, PagedAttention & Internals** — `[vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html)`
- 7.4 **SGLang: RadixAttention & Structured Programs** — `[SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html)`
- 7.5 **TensorRT-LLM, TGI & Other Serving Stacks** — `[TensorRT-LLM, TGI & Other Serving Stacks](../07-inference-serving/05-trtllm-tgi-stacks.html)`
- 7.6 **Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead** — `[Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)`
- 7.7 **Prefix Caching & KV-Cache Reuse** — `[Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html)`
- 7.8 **Disaggregated Prefill/Decode & Chunked Prefill** — `[Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html)`
- 7.9 **Sampling Strategies & Decoding Algorithms** — `[Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html)`
- 7.10 **Structured & Constrained Generation** — `[Structured & Constrained Generation](../07-inference-serving/10-structured-generation.html)`
- 7.11 **Multi-GPU & Multi-Node Inference** — `[Multi-GPU & Multi-Node Inference](../07-inference-serving/11-multi-gpu-inference.html)`
- 7.12 **Inference Economics: Latency, Throughput & Cost** — `[Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html)`
- 7.13 **Serving Mixture-of-Experts: Expert Parallelism & All-to-All Inference** — `[Serving Mixture-of-Experts: Expert Parallelism & All-to-All Inference](../07-inference-serving/13-serving-moe.html)`
- 7.14 **Multi-Tenant LoRA & Adapter Serving at Scale** — `[Multi-Tenant LoRA & Adapter Serving at Scale](../07-inference-serving/14-multi-tenant-lora-serving.html)`

## Part VIII — Agents & Harness Engineering
- 8.1 **Tool Use & Function Calling** — `[Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html)`
- 8.2 **The Agentic Loop: ReAct, Plan-Execute & Reflection** — `[The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html)`
- 8.3 **Harness Engineering: Building a Coding Agent** — `[Harness Engineering: Building a Coding Agent](../08-agents-harness/03-harness-coding-agent.html)`
- 8.4 **Context Engineering & Management** — `[Context Engineering & Management](../08-agents-harness/04-context-engineering.html)`
- 8.5 **Memory Systems for Agents** — `[Memory Systems for Agents](../08-agents-harness/05-agent-memory.html)`
- 8.6 **The Model Context Protocol (MCP)** — `[The Model Context Protocol (MCP)](../08-agents-harness/06-mcp.html)`
- 8.7 **Multi-Agent Systems & Orchestration** — `[Multi-Agent Systems & Orchestration](../08-agents-harness/07-multi-agent-systems.html)`
- 8.8 **Agent Evaluation & Benchmarks** — `[Agent Evaluation & Benchmarks](../08-agents-harness/08-agent-evaluation.html)`
- 8.9 **Prompt Engineering as Engineering** — `[Prompt Engineering as Engineering](../08-agents-harness/09-prompt-engineering.html)`

## Part IX — Retrieval & RAG
- 9.1 **Embeddings & Representation Learning** — `[Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html)`
- 9.2 **Vector Databases & Approximate Nearest Neighbor Search** — `[Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html)`
- 9.3 **Retrieval-Augmented Generation Architectures** — `[Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html)`
- 9.4 **Chunking, Reranking & Hybrid Search** — `[Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html)`
- 9.5 **Advanced RAG: GraphRAG, Agentic RAG & Long-Context vs RAG** — `[Advanced RAG: GraphRAG, Agentic RAG & Long-Context vs RAG](../09-rag-retrieval/05-advanced-rag.html)`
- 9.6 **Multimodal & Visual-Document Retrieval: ColPali & Late Interaction** — `[Multimodal & Visual-Document Retrieval: ColPali & Late Interaction](../09-rag-retrieval/06-multimodal-visual-retrieval.html)`

## Part X — Multimodal & Generative Frontiers
- 10.1 **Vision Transformers & Image Encoders** — `[Vision Transformers & Image Encoders](../10-multimodal-and-arch/01-vision-transformers.html)`
- 10.2 **Vision-Language Models** — `[Vision-Language Models](../10-multimodal-and-arch/02-vision-language-models.html)`
- 10.3 **Audio, Speech & Multimodal Fusion** — `[Audio, Speech & Multimodal Fusion](../10-multimodal-and-arch/03-audio-speech-multimodal.html)`
- 10.4 **Diffusion Models & Generative Modeling (Breadth)** — `[Diffusion Models & Generative Modeling (Breadth)](../10-multimodal-and-arch/04-diffusion-generative.html)`
- 10.5 **Unified & Any-to-Any Models** — `[Unified & Any-to-Any Models](../10-multimodal-and-arch/05-unified-any-to-any.html)`

## Part XI — Evaluation
- 11.1 **The Evaluation Problem & Benchmark Landscape** — `[The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html)`
- 11.2 **LLM-as-a-Judge & Automated Evaluation** — `[LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html)`
- 11.3 **Building Eval Harnesses** — `[Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html)`
- 11.4 **Reasoning, Coding & Agentic Evals** — `[Reasoning, Coding & Agentic Evals](../11-evaluation/04-reasoning-coding-agentic-evals.html)`
- 11.5 **Red-Teaming, Safety & Robustness Evaluation** — `[Red-Teaming, Safety & Robustness Evaluation](../11-evaluation/05-redteaming-safety-eval.html)`
- 11.6 **Statistical Rigor in Evaluation: Confidence Intervals & Significance** — `[Statistical Rigor in Evaluation: Confidence Intervals & Significance](../11-evaluation/06-statistical-rigor-eval.html)`

## Part XII — Production, Systems & MLOps
- 12.1 **Designing an LLM Serving System** — `[Designing an LLM Serving System](../12-production-mlops/01-serving-system-design.html)`
- 12.2 **Observability, Logging & LLMOps** — `[Observability, Logging & LLMOps](../12-production-mlops/02-observability-llmops.html)`
- 12.3 **Caching, Routing & Cost Control in Production** — `[Caching, Routing & Cost Control in Production](../12-production-mlops/03-caching-routing-cost.html)`
- 12.4 **Safety, Guardrails & Content Moderation** — `[Safety, Guardrails & Content Moderation](../12-production-mlops/04-safety-guardrails.html)`
- 12.5 **Data Flywheels & Continuous Improvement** — `[Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html)`
- 12.6 **Security: Prompt Injection, Jailbreaks & Defenses** — `[Security: Prompt Injection, Jailbreaks & Defenses](../12-production-mlops/06-security-prompt-injection.html)`
- 12.7 **Online Evaluation: A/B Testing, Canaries & Guardrail Metrics** — `[Online Evaluation: A/B Testing, Canaries & Guardrail Metrics](../12-production-mlops/07-online-eval-ab-testing.html)`
- 12.8 **Reliability Engineering for LLM Systems: SLOs & Incident Response** — `[Reliability Engineering for LLM Systems: SLOs & Incident Response](../12-production-mlops/08-reliability-engineering.html)`

## Part XIII — Interpretability, Safety & Governance
- 13.1 **Mechanistic Interpretability & Model Internals** — `[Mechanistic Interpretability & Model Internals](../13-interp-safety-gov/01-mechanistic-interpretability.html)`
- 13.2 **Knowledge Editing & Machine Unlearning** — `[Knowledge Editing & Machine Unlearning](../13-interp-safety-gov/02-knowledge-editing-unlearning.html)`
- 13.3 **Privacy, Memorization & Differential Privacy for LLMs** — `[Privacy, Memorization & Differential Privacy for LLMs](../13-interp-safety-gov/03-privacy-memorization-dp.html)`
- 13.4 **Watermarking, Provenance & AI-Content Detection** — `[Watermarking, Provenance & AI-Content Detection](../13-interp-safety-gov/04-watermarking-provenance.html)`
- 13.5 **AI Safety: Scalable Oversight, Dangerous-Capability Evals & Frontier Safety** — `[AI Safety: Scalable Oversight, Dangerous-Capability Evals & Frontier Safety](../13-interp-safety-gov/05-ai-safety-oversight.html)`
- 13.6 **AI Governance, Compliance & Regulation** — `[AI Governance, Compliance & Regulation](../13-interp-safety-gov/06-governance-compliance.html)`

## Appendix
- **Glossary of Terms** — `[Glossary of Terms](../99-appendix/01-glossary.html)`
- **The Math Reference Sheet** — `[The Math Reference Sheet](../99-appendix/02-math-reference.html)`
- **Key Papers: An Annotated Reading List** — `[Key Papers: An Annotated Reading List](../99-appendix/03-papers-reading-list.html)`
- **Tooling & Environment Setup Cheatsheet** — `[Tooling & Environment Setup Cheatsheet](../99-appendix/04-tooling-setup.html)`
- **From-Scratch Code Index** — `[From-Scratch Code Index](../99-appendix/05-from-scratch-index.html)`

> The interview companion lives at `../interview/<file>.html` (separate site).
