
===== BOOK: The LLM Stack =====

## Front Matter  [00-frontmatter]
  Preface: How to Read This Book & The Map of the Stack

## Part I — Mathematical & Systems Foundations  [01-foundations]
  1.1 Linear Algebra for Deep Learning
  1.2 Probability, Statistics & Information Theory
  1.3 Calculus, Optimization & Convexity
  1.4 Numerical Computing, Floating Point & Precision
  1.5 Machine Learning Fundamentals
  1.6 Neural Networks From Scratch: MLPs & Backprop
  1.7 Automatic Differentiation & PyTorch Internals
  1.8 GPU Architecture & The Memory Hierarchy
  1.9 Parallel Computing & Collective Communication

## Part II — The Transformer Architecture  [02-transformer]
  2.1 Tokenization: BPE, WordPiece, Unigram & Byte-Level
  2.2 Embeddings & The Input Pipeline
  2.3 The Attention Mechanism From Scratch
  2.4 Multi-Head Attention, MQA, GQA & MLA
  2.5 Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi
  2.6 The Transformer Block: Norms, Residuals, MLPs & Activations
  2.7 Building a GPT From Scratch (nanoGPT-style)
  2.8 Architecture Variants: Encoder-Decoder, Decoder-Only & Prefix-LM
  2.9 Mixture-of-Experts (MoE) Architectures
  2.10 Modern Architecture Improvements & Design Choices
  2.11 Beyond Attention: SSMs, Mamba, RWKV & Linear Attention

## Part III — Pretraining at Scale  [03-pretraining]
  3.1 Pretraining Data: Sources, Crawling & The Data Pipeline
  3.2 Data Cleaning, Deduplication & Quality Filtering
  3.3 The Pretraining Objective & Loss
  3.4 Scaling Laws: Kaplan, Chinchilla & Beyond
  3.5 Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP
  3.6 Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism
  3.7 Megatron-LM, DeepSpeed & Parallelism in Practice
  3.8 Mixed Precision, bf16 & FP8 Training
  3.9 Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo
  3.10 Learning Rate Schedules, Warmup, Batch Size & Hyperparameters
  3.11 Training Stability, Loss Spikes & Debugging Large Runs
  3.12 Checkpointing, Fault Tolerance & Long-Running Jobs
  3.13 Long-Context Pretraining & Context Extension

## Part IV — Kernels, Efficiency & Quantization  [04-kernels-efficiency]
  4.1 The Roofline Model & Performance Engineering
  4.2 FlashAttention I: IO-Awareness & The Online Softmax
  4.3 FlashAttention 2 & 3: Work Partitioning, Warp Specialization & FP8
  4.4 Writing GPU Kernels with Triton
  4.5 CUDA Programming Essentials for ML Engineers
  4.6 PagedAttention & KV-Cache Memory Management
  4.7 Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)
  4.8 Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT
  4.9 Kernel Fusion, torch.compile, CUDA Graphs & Compilers
  4.10 Memory-Efficient Training: Checkpointing, Offloading & LoRA Math

## Part V — Post-Training & Alignment  [05-posttraining-alignment]
  5.1 Supervised Fine-Tuning & Instruction Tuning
  5.2 Chat Templates, Data Formatting & Sequence Packing
  5.3 PEFT I: LoRA, QLoRA, DoRA & The Adapter Family
  5.4 PEFT II: Prompt/Prefix Tuning, IA3, Model Merging & Soups
  5.5 The RLHF Pipeline & Reward Modeling
  5.6 Policy Gradients & PPO for Language Models
  5.7 Direct Preference Optimization & Its Variants
  5.8 GRPO, RLOO & Critic-Free RL
  5.9 RL with Verifiable Rewards (RLVR) & The Reasoning Recipe
  5.10 Reasoning, Chain-of-Thought & Test-Time Compute
  5.11 Constitutional AI, RLAIF & Self-Improvement
  5.12 Distillation, Model Compression & Knowledge Transfer
  5.13 Reward Hacking, Over-Optimization & Alignment Failures

## Part VI — RL Infrastructure (Deep Dive)  [06-rl-infra]
  6.1 The Anatomy of an RL-for-LLM System
  6.2 The Generation–Training Loop & Rollout Engines
  6.3 TRL: HuggingFace's RL Library
  6.4 veRL: HybridFlow & The Single-Controller Architecture
  6.5 OpenRLHF, NeMo-Aligner & Ray-Based Systems
  6.6 Prime-RL, Async RL & Decentralized Training
  6.7 Colocated vs Disaggregated RL & Weight Synchronization
  6.8 Reward Engineering, Verifiers & Sandboxes
  6.9 Advantage Estimation, KL Control & Stability Tricks
  6.10 Agentic & Multi-Turn RL
  6.11 Scaling RL: Throughput, Load Balancing & The Latest Tricks

## Part VII — Inference & Serving  [07-inference-serving]
  7.1 The Anatomy of LLM Inference: Prefill, Decode & The KV Cache
  7.2 Continuous Batching & Request Scheduling
  7.3 vLLM: Architecture, PagedAttention & Internals
  7.4 SGLang: RadixAttention & Structured Programs
  7.5 TensorRT-LLM, TGI & Other Serving Stacks
  7.6 Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead
  7.7 Prefix Caching & KV-Cache Reuse
  7.8 Disaggregated Prefill/Decode & Chunked Prefill
  7.9 Sampling Strategies & Decoding Algorithms
  7.10 Structured & Constrained Generation
  7.11 Multi-GPU & Multi-Node Inference
  7.12 Inference Economics: Latency, Throughput & Cost

## Part VIII — Agents & Harness Engineering  [08-agents-harness]
  8.1 Tool Use & Function Calling
  8.2 The Agentic Loop: ReAct, Plan-Execute & Reflection
  8.3 Harness Engineering: Building a Coding Agent
  8.4 Context Engineering & Management
  8.5 Memory Systems for Agents
  8.6 The Model Context Protocol (MCP)
  8.7 Multi-Agent Systems & Orchestration
  8.8 Agent Evaluation & Benchmarks
  8.9 Prompt Engineering as Engineering

## Part IX — Retrieval & RAG  [09-rag-retrieval]
  9.1 Embeddings & Representation Learning
  9.2 Vector Databases & Approximate Nearest Neighbor Search
  9.3 Retrieval-Augmented Generation Architectures
  9.4 Chunking, Reranking & Hybrid Search
  9.5 Advanced RAG: GraphRAG, Agentic RAG & Long-Context vs RAG

## Part X — Multimodal & Generative Frontiers  [10-multimodal-and-arch]
  10.1 Vision Transformers & Image Encoders
  10.2 Vision-Language Models
  10.3 Audio, Speech & Multimodal Fusion
  10.4 Diffusion Models & Generative Modeling (Breadth)
  10.5 Unified & Any-to-Any Models

## Part XI — Evaluation  [11-evaluation]
  11.1 The Evaluation Problem & Benchmark Landscape
  11.2 LLM-as-a-Judge & Automated Evaluation
  11.3 Building Eval Harnesses
  11.4 Reasoning, Coding & Agentic Evals
  11.5 Red-Teaming, Safety & Robustness Evaluation

## Part XII — Production, Systems & MLOps  [12-production-mlops]
  12.1 Designing an LLM Serving System
  12.2 Observability, Logging & LLMOps
  12.3 Caching, Routing & Cost Control in Production
  12.4 Safety, Guardrails & Content Moderation
  12.5 Data Flywheels & Continuous Improvement
  12.6 Security: Prompt Injection, Jailbreaks & Defenses

## Appendix  [99-appendix]
  Glossary of Terms
  The Math Reference Sheet
  Key Papers: An Annotated Reading List
  Tooling & Environment Setup Cheatsheet
  From-Scratch Code Index

===== INTERVIEW: ML/LLM Interview Companion =====

## The ML/LLM Interview — Companion Prep  [13-interview-prep]
  1.1 The ML Engineer Interview: Formats & What's Tested
  1.2 ML Breadth: Rapid-Fire Concepts & Model Answers
  1.3 ML System Design: A Framework
  1.4 ML System Design: Worked Cases
  1.5 Coding for ML Interviews
  1.6 LLM-Specific Deep-Dive Questions
  1.7 Behavioral, Leadership & Project Deep-Dive
  1.8 The 10-Day Study Plan & Final Checklist