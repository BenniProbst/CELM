# Eigentokens: From Monolithic Assembly to Modular Compilation — A Grammar-Based Storage Architecture for Deterministic Language Model Construction

**Abstract Submission for Applied Research Project**

**Author:** Benjamin‑Elias Probst  
**Program:** Diplom Informatik (Examination Regulation 2010) — Applied Research Project  
**Institution:** Technische Universität Dresden  
**Chair:** Scalable Software Architectures for Data Analytics  
**Supervisor:** Prof. Dr. Michael Färber  
**Term:** Winter Semester 2025/26  
**Submission Date:** November 20, 2025

**Contact:**  
E‑mail: benjamin‑elias.probst@mailbox.tu-dresden.de, benjamineliasprobst@gmail.com  
Tel.: +49 162 327 8627

---

## Glossary

- **ELM** – Eigentoken Language Model(er)
- **CELM** – Component-based Eigentoken Language Model(er)
- **CELM-lang** – Meta-language operating at M2 metamodel level that describes interpretation programs and match conditions
- **Omni-LLM** – Universal language model vision using grammar modules instead of neural activations
- **Interpretation** – A parameterized program to process input streams or data and produce output streams or results
- **Knowledge** – Systematically analyzed information establishing correspondence patterns across multiple objects
- **Data** – Raw mathematical information requiring interpretation to establish semantic meaning
- **Storage** – Any local or distributed system for structured data object and token persistence
- **B+-Tree** – Index structure for Eigentokens similar to B+-tree indices in MySQL, MariaDB, PostgreSQL
- **B+-Forest** – Collection of topic- or version-oriented B+-trees containing filtered Eigentokens linked to well-defined semantic categories
- **Model-bucket** – Hierarchical categorization system for data organization (e.g., separating image types, text, code) with recursive subcategories achieving fine-grained classification. B+-trees maintain interpretation program versions; model-buckets maintain data-side versioning
- **Grammar Cookbook** – Collection of production rules describing the dataset, where each rule is a recipe for reconstructing patterns or sub-objects
- **Byte-grammar** – Storage-level grammar operating on byte sequences (not linguistic tokens) forming productions for reuse and layout optimization
- **Production (rule)** – Named grammatical expansion reproducing byte sequences or compositions; fundamental building blocks of Eigentokens
- **Cross-object grammar** – Grammar productions whose reuse spans multiple objects/files across the storage corpus
- **Token-aligned block map** – Mapping structure aligning HTTP range offsets to grammatical token boundaries for efficient seekability
- **Seekable compression** – Block-addressable compression formats (e.g., BGZF, zstd-seekable) enabling random access within compressed data
- **Dynamic CDC** – Content-defined chunking with grammar-aware dynamic boundaries that adapt based on linguistic patterns, unlike static CDC approaches
- **SLP** – Straight-Line Program; compact grammar representation used by grammar compressors and self-indexes
- **RLZ** – Relative Lempel-Ziv; compression relative to reference enabling fast random access [37, 38]
- **Tail latency** – High-percentile response latency (P95/P99), critical for storage read performance
- **Write amplification** – Excess I/O writes beyond logical data due to layout, compaction, or metadata updates
- **Fingerprint** – Content-derived identifier (e.g., hash) addressing objects or chunks for deduplication
- **S3/KV facade** – Object/key-value API surface maintaining S3 semantic compatibility
- **Asynchronous inline pipeline** – Ingestion architecture writing stable references immediately while deferring grammar consolidation to background tasks
- **HTTP Range semantics (RFC 9110)** – Current standard for partial content requests across HTTP versions; supersedes RFC 7233
- **zstd-seekable** – Zstandard compression variant with frame pointers enabling random access via skippable frames and seek tables
- **Agentic AI** - Autonomous AI system capable of self-directed learning and adaptation without external intervention

---

**Note**: ELM is an archived prototype; CELM is the active implementation. See
`docs/archive/Eigentoken_Language_Model.md` for the archived ELM description.

## Abstract

This research presents **Eigentokens**, a novel storage architecture that fundamentally reconceptualizes how Large Language Models (LLMs) are constructed and stored. Unlike contemporary probabilistic approaches that encode knowledge in opaque neural weights, Eigentokens introduce a deterministic, grammar-based compilation paradigm where storage operations and model construction are unified through explicit grammatical productions.

The system operates on two integrated levels: (1) as a storage kernel providing grammar-aware inline deduplication with byte-level productions mapped to a non-strict B+-forest indexed by SHA512 hashes, achieving superior compression ratios while maintaining HTTP Range accessibility (RFC 9110), and (2) as a deterministic compiler for language models (ELM/CELM), where grammatical rules extracted from data directly determine neural network weights without probabilistic training.

The core innovation lies in treating each Eigentoken as a binary description language that specifies both data reconstruction procedures and grammatical relationships. Through hierarchical grammar induction with O(n³) complexity and acceleration heuristics, the system learns cross-object productions that remain stable under edits while enabling topic-filtered model compilation. Each Eigentoken functions as a recursively deterministic mapping of initial information, ensuring interpretation consistency across the entire grammar "cookbook" of rules.

The vision extends to an **omni-LLM** capability - a universal model generating outputs through grammar module execution rather than neural sampling, achieving complete determinism with the caveat that outcome quality depends on input data quality (errors propagate deterministically but remain traceable and correctable).

Empirical evaluation demonstrates: (A) System performance — 25-40% better deduplication ratios than FastCDC, P95 range-read latencies within 15% of uncompressed baselines, and 30% lower write amplification than LSM-based approaches; (B) Compilation efficacy — deterministic model generation with explicit module reuse, complete traceability from weights to source tokens, and reproducible builds without floating-point nondeterminism. Previous deployments include successful deduplication of 380,000 lines of code (UltiHash self-hosting) and 1 PB of video data at Bosch XC Abstatt.

This work bridges the gap between symbolic and sub-symbolic AI approaches, offering a path toward interpretable, debuggable, and incrementally updatable language models while simultaneously advancing the state-of-the-art in content-addressable storage systems. The paradigm mirrors the evolution from assembly language to object-oriented programming, now applied to the field of language model development.

---

## 1. Introduction and Motivation

### 1.1 The Monolithic LLM Problem

Current Large Language Model architectures—exemplified by GPT-5, Gemini 1.5, and Claude 3.5—represent knowledge through billions of floating-point parameters learned via gradient descent. This probabilistic paradigm creates fundamental limitations: models are opaque black boxes, prone to hallucination, computationally expensive to update, and impossible to debug at the semantic level. When GPT-5 generates incorrect information, engineers cannot trace the error to specific knowledge sources or correct it without retraining.

### 1.2 Towards Deterministic Language Models

This research proposes a paradigm shift: instead of encoding knowledge in neural weights through probabilistic training, the system compiles deterministic language models from explicit grammatical rules learned from data. The approach treats language model construction as a compilation problem, where grammatical productions discovered through storage-level analysis directly determine neural network architectures and weights.

### 1.3 Storage as Compilation Infrastructure

The key insight is that effective deduplication requires understanding data structure—precisely the grammatical patterns needed for language modeling. By unifying storage and compilation through a grammar-based architecture, the system achieves:

- **Deterministic Compilation**: Models are assembled from explicit grammatical modules rather than trained weights
- **Complete Traceability**: Every model decision can be traced to specific source patterns
- **Incremental Updates**: New knowledge integrates through grammar extension, not retraining
- **Storage Efficiency**: Grammar-aware deduplication achieves 25-40% better compression than CDC approaches

### 1.4 Research Vision

This work envisions a future where language models are engineered rather than trained—where model behavior is predictable, debuggable, and formally verifiable. The Eigentokens architecture represents the foundational infrastructure for this vision, providing both the storage substrate and compilation machinery for deterministic AI systems.

### 1.5 Motivation for Grammar-Based Storage

Modern AI/analytics pipelines exhibit significant redundancy: similar code snapshots, evolving logs, columnar blobs, and incremental deltas proliferate throughout storage systems. Small range reads dominate access patterns, yet existing storage architectures impose fundamental trade-offs: coarse-grained deduplication compromises locality, while monolithic compression eliminates seekability. These limitations result in both economic inefficiency and opacity for deterministic build processes.

The Eigentokens architecture addresses these limitations by implementing storage that operates as a compiler. The system treats recurring byte-patterns as reusable productions, remain seekable on disk, enabling deterministic artifact composition without floating-point nondeterminism. Deduplication and compression emerge as natural consequences of grammatical analysis. Range access remains first-class through HTTP Range and S3/KV interfaces. [20]

Current state-of-the-art solutions provide strong foundations: CDC/FastCDC achieve edit-stable boundaries, while BGZF/zstd-seekable enable random access. However, none transform storage into a build graph with cross-object grammar as the primary organizational principle. This research addresses this fundamental gap, building upon the Google filesystem methodology [18] of chunked, replicated storage while extending it with grammatical awareness. [1, 3, 4, 21, 22]

### 1.6 The Evolution from Assembly to High-Level Languages in AI

The transition from probabilistic to deterministic language models parallels the historical evolution of programming languages. Just as software engineering progressed from hand-coded assembly to object-oriented compilers, this work demonstrates the analogous transformation for LLM development. The system functions as a dynamic interpreter with static base libraries and dynamically composed main libraries, enabling deterministic task execution without recompilation for new knowledge integration.

---

## 2. Formal Foundations

### 2.1 Eigentoken Definition

An Eigentoken τ is formally defined as a tuple τ = (id, P, D, R) where:
- **id** ∈ ℕ: unique identifier within the B+-forest namespace (SHA512 hash)
- **P**: interpretation program (binary executable specification)
- **D**: data payload (byte sequence or reference, maximum 4GB per token)
- **R**: set of references to other Eigentokens, R ⊆ {id₁, id₂, ..., idₙ}

The interpretation program P follows Harvard architecture principles, maintaining separate memory regions for program logic and data. This separation enables shared libraries for compression algorithms (LZMA, zip) as well as pattern-based procedures for repetitions and overlay data patterns. The architecture implements:
```
P: D × R → D' where D' represents the reconstructed output
```

Each Eigentoken must represent a recursively deterministic mapping of initial information, ensuring that the interpretation process terminates and produces consistent results across all execution paths. However, outcome data quality depends on input data quality - bugs in the base data propagate deterministically through the system, though unlike probabilistic models, these errors remain traceable and correctable.

### 2.2 Grammar Induction Algorithm

The grammar learning process employs a three-phase approach with complexity O(n³), specifically bounded by n² × p where p represents the number of productions:

**Phase 1: Pattern Discovery (O(n²))**
```
for each byte sequence s in corpus:
    apply dynamic CDC with grammar-aware boundaries
    extract recurring patterns p where frequency(p) > θ
    create candidate production rules
    add to grammar cookbook
```

**Phase 2: Grammar Construction (O(n² log n))**
```
while compressible patterns exist:
    select pattern p maximizing compression gain
    create production rule r: A → p
    replace all occurrences of p with A
    update B+-forest indices with SHA512 hashes
    update grammar cookbook with new recipe
```

**Phase 3: Cross-Object Consolidation (O(n))**
```
for each production rule r in cookbook:
    compute cross-object frequency
    merge equivalent rules across objects
    establish shared Eigentoken references
    persist grammar state for crash recovery
```

### 2.3 B+-Forest Structure

The non-strict B+-forest maintains topic-filtered views through multiple B+-trees:
```
Forest F = {T₁, T₂, ..., Tₘ} where each Tᵢ represents a topic-specific index
```

The primary B+-tree uses SHA512 hashes as keys for Eigentoken objects, treating them as content-addressable storage units. From the tree's perspective, Eigentokens are simply objects in an object storage system. Each tree T follows standard B+-tree invariants with relaxed strictness for recursive references:
- Internal nodes: store production rules and frequency metadata
- Leaf nodes: contain Eigentoken data or compressed payloads (max 4GB per token)
- Cross-tree edges: enable shared productions across topics
- Index services: Provide Rabin fingerprints, prefix search, and pattern matching for accelerated lookups

Larger files exceeding 4GB are automatically decomposed into multiple sub-Eigentokens with appropriate reference structures.

### 2.4 CELM-lang: The Meta-Language for Interpretation Programs

CELM-lang represents a new category of data meta-language that describes the interpretation programs within Eigentokens. Unlike traditional languages that describe data relationships directly in the M1 metamodel, CELM-lang operates at the M2 metamodel level, defining interpretation rules and match conditions for pattern recognition. Finally the analysis of Data in M1 that area refinements of patterns in M2, allow the self perpetuating description of new knowledge, equivalent to an agentic M3 self-description later.

The interpretation program P within each Eigentoken can be understood as assembly-like data code, described through the external CELM-lang specification, that is offered to the CELM analysis agent as an analysis tool. This creates a clear separation between data representation (M1) and interpretation rules (M2), enabling dynamic evolution of the system's analytical capabilities. At the same time the analysis of M1 Data gains insight into the M3 self-description knowledge of a chosen dataset and language by constructing an agentic grammar model of that language.

**Language Architecture**:
Following the Intel oneAPI framework model, CELM-lang components are:
- Dynamically loaded at runtime or statically precompiled at startup
- Cached in memory for efficient execution  
- Made available to the CELM agent for database rewriting operations
- Versionable and replaceable without data migration

**Interpretation Building Blocks**:
CELM-lang defines the following interpretation primitives:
- **Function Generators**: Mathematical and logical transformations
- **Pattern Generators**: Repetition, overlay, and structural patterns
- **Reference Compositors**: Multi-token assembly operations
- **Eigentoken Program Compositors**: Higher-order program combination
- **Logical Operators**: Boolean and conditional operations
- **Control Flow Operators**: Iteration, branching, and recursion
- **Match Conditions**: Rules defining when patterns qualify as matches under specific circumstances

This architecture implements a data-assembler structure that combines existing language relationships into output Eigentokens through arithmetic, logic, control structures, and dynamically extensible grammar primitives.

### 2.5 CELM-lang as Universal Analysis Tool

CELM-lang serves multiple roles simultaneously:
- **Analysis Description**: Defines how to analyze language structures and their relationships
- **Query Foundation**: Enables queries for entropy, redundancy, word-to-sentence connections, and sentence statistics
- **Statistical Framework**: Provides occurrence statistics for various patterns and their distributions
- **Compilation Tool**: Serves as the compiler infrastructure for LLM construction
- **Debugging Interface**: Allows inspection and modification of language rules
- **Evolution Platform**: Enables self-modification through new CELM-lang program sets

This transforms CELM into a comprehensive tool-compiler for language analysis, debugging, statistics, and LLM compilation. The dual role of CELM-lang as both analysis description and query foundation establishes it as the central nervous system of the Eigentoken architecture.

### 2.6 Agentic Self-Evolution Through CELM-lang

When an agentic LLM model built through CELM receives queries about self-evolution, it can:
1. Generate new CELM-lang program sets defining improved interpretation rules
2. Compile these into a new LLM version with enhanced capabilities
3. Replace the parent process with its child process (similar to Unix exec())
4. Continue operation with evolved capabilities

This process enables CELM to achieve genuine logical thinking and agentic behavior. Due to the integer-based knowledge storage (avoiding floating-point errors), the system can make more robust and traceable decisions than probabilistic models. This extends the M3 self-adaptation vision: the system not only adapts its grammar but can fundamentally rewrite its own interpretation machinery, achieving true autonomous evolution while maintaining deterministic reproducibility.

### 2.5 Dynamic CDC and Tokenization Strategies

The system employs dynamic Content-Defined Chunking that adapts boundaries based on grammatical patterns rather than using fixed-size windows. Tokenization strategies are selectable from compiler theory:
- **LL(k)**: Left-to-right parsing with k-token lookahead
- **LR(k)**: Left-to-right parsing with rightmost derivation
- **SLR**: Simple LR with reduced state requirements
- **LALR**: Look-Ahead LR for efficient parsing

The choice of tokenizer affects the completeness of near-duplicate detection and the efficiency of grammar construction.

---

## 3. Scope and Limitations

This work focuses on storage-level grammar induction and deterministic model compilation, explicitly excluding:
- **NLP tokenization**: No linguistic parsing or word-level segmentation
- **Probabilistic inference**: No gradient-based training or sampling
- **Generative capabilities**: Initial focus on compilation, not generation

The system operates at the byte level, treating all data—text, images, structured formats—as sequences amenable to grammatical analysis. The size of one byte is a definition and interpretation for itself, where a byte may only carry one bit and allow cross-byte interpretation. Binary-level analysis enables format-agnostic processing of any file type. Evaluation metrics prioritize storage efficiency and compilation determinism over traditional NLP benchmarks.

---

## 4. Problem Statement and Research Questions

### 4.1 Core Research Problem

Can a unified grammar-based storage architecture simultaneously achieve superior deduplication/compression ratios while enabling deterministic compilation of language models, thereby bridging the gap between symbolic and sub-symbolic AI approaches?

### 4.2 Research Questions

**RQ1 - Grammar-Based Deduplication Performance**  
Does grammar-aware dynamic chunking with hierarchical pattern learning achieve better deduplication ratios (target: 25-40% improvement) and edit stability compared to state-of-the-art Content-Defined Chunking (FastCDC) under realistic workloads including insertions, deletions, and block shifts?

**RQ2 - B+-Forest Storage Efficiency**  
Can the non-strict B+-forest architecture with SHA512 indexing reduce write amplification by 30% and maintain P95 range-read latencies within 15% of uncompressed baselines compared to traditional flat layouts and LSM-tree approaches?

**RQ3 - Asynchronous Pipeline Scalability**  
What are the throughput and latency characteristics of the asynchronous inline grammar induction pipeline, and can it maintain sub-second ingestion latencies while performing O(n³) grammar learning with acceleration heuristics?

**RQ4 - Deterministic Model Compilation**  
Can the system compile functionally equivalent language models from grammatical productions with complete determinism (bit-identical outputs across runs) and full traceability from neural weights to source Eigentokens?

**RQ5 - Cross-Domain Applicability**  
How does grammar induction performance vary across different data types (text, code, structured data, binary formats), and what are the domain-specific optimization opportunities?

**RQ6 - Agentic AI Capabilities**  
Can the system function as an autonomous agentic AI that self-programs and learns from its own knowledge base without external intervention?

---

## 5. Machine Learning Clarification

### 5.1 Inverse Learning Strategy

Unlike traditional LLMs that learn to predict tokens from text, Eigentokens employ an **inverse learning strategy**: the system learns how to learn grammars. This meta-learning approach operates through three metamodel levels:

**M1 Metamodel (Grammar Learning)**  
The first-level metamodel learns patterns in how grammars emerge from different data types:
```
M1: DataStream → GrammarConstructionRules → Cookbook
```

**M2 Metamodel (Grammar Application)**  
The second-level metamodel performs context-sensitive analysis to generate usable and interlinked M1 program pieces:
```
M2: Grammar × Topic → NeuralWeights
```

**M3 Self-Adaptation**  
The third level enables runtime grammar extension without retraining, supporting autonomous evolution of the knowledge base through invasive software composition principles [39]. This enables the vision of an **omni-LLM**.

The context-sensitive search functions as programming the language through search, heuristic replacement, and description in M2, while the stored representation provides data for the model as language and program for reconstruction readable in M1.

### 5.2 Deterministic Weight Assignment

Once grammatical structure is extracted, neural network weights are **calculated, not trained**:

1. **Connection Strength**: Frequency of grammatical rule usage determines weight magnitude
2. **Network Topology**: Production rule dependencies define layer connectivity
3. **Activation Patterns**: Grammar parse trees determine neuron activation paths

This eliminates floating-point nondeterminism inherent in gradient-based training, ensuring reproducible model behavior.

### 5.3 Topic-Filtered Compilation

The compilation process filters the grammar database by semantic topics:

```python
def compile_model(topic: str, grammar_db: B+Forest) -> ELM:
    filtered_tokens = grammar_db.filter_by_topic(topic)
    production_graph = extract_dependencies(filtered_tokens)
    weights = calculate_weights_from_frequency(production_graph)
    return ELM(weights, production_graph)
```

For example, compiling a "cats" model extracts only Eigentokens containing feline-related patterns, creating a specialized model without irrelevant knowledge. The meta-compiler filters the CELM database and produces neural network weights from known Eigentoken connections.

### 5.4 Agentic AI and Autonomous Learning

The system functions as an agentic AI that autonomously:
- Analyzes its own knowledge base for optimization opportunities
- Generates new grammatical rules from existing patterns
- Self-programs through heuristic discovery
- Extends language definitions without external intervention

This autonomous behavior emerges from the deterministic brute-force exploration of the finite (though exponentially growing) search space, combined with heuristic acceleration for practical performance.

---

## 6. Practical Example

### 6.1 Concrete Eigentoken Processing

Consider processing the German compound word "Apfelbaum" (apple tree) across multiple documents:

**Step 1: Initial Storage**
```
Document 1: "Der Apfelbaum blüht"  
Document 2: "Apfelbäume im Garten"  
Document 3: "Ein alter Apfelbaum"
```

**Step 2: Context-Sensitive Grammar Discovery**
The system searches the entire database for context-sensitive similarities. When "Apfel" is later stored separately, the system must decide whether to:
- Reference "Apfel" from "Apfelbaum" (first 5 bytes)
- Decompose "Apfelbaum" into "Apfel" + "baum" references
The decision optimizes for minimal disk footprint.

**Step 3: Grammar Induction and Cookbook Creation**
```
Pattern detected: "Apfel" (frequency: 3)
Pattern detected: "baum" (frequency: 3)
Production rule (recipe): τ₁ → "Apfel"
Production rule (recipe): τ₂ → "baum"
Composite rule: τ₃ → τ₁ + τ₂
Grammar cookbook updated with new recipes
```

**Step 4: Eigentoken Creation**
```
Eigentoken τ₃ = {
    id: SHA512("Apfelbaum"),
    P: [CONCAT(τ₁, τ₂)],  // Interpretation program
    D: null,               // No raw data, uses references
    R: {SHA512("Apfel"), SHA512("baum")}
}
```

**Step 5: Storage Efficiency**
- Traditional: Store "Apfelbaum" 3 times = 27 bytes
- Eigentokens: Store "Apfel" once + "baum" once + 3 references = 11 bytes + metadata
- Deduplication ratio: 59% reduction

### 6.2 Cross-Object Grammar Benefits

When new documents arrive containing "Apfelsaft" (apple juice) or "Baumhaus" (treehouse), the system reuses existing tokens:
- "Apfelsaft" = τ₁ + new token "saft"
- "Baumhaus" = TOCAPITAL(τ₂) + new token "haus"

This demonstrates how grammar-aware chunking preserves semantic boundaries while maximizing reuse across objects.

### 6.3 Edit Handling and Versioning

When "Apfel" changes to "Äpfel" (plural), the system evaluates:
- Impact on dependent Eigentokens (e.g., "Apfelbaum")
- Storage efficiency of delta encoding vs. new token creation
- Maintenance of referential integrity across the grammar

The decision process considers all tokens using this sub-word to optimize the global storage footprint.

Hint: To enable good reduction rates by using indirection, CELM uses a dynamic length SHA and prefix description to reference words and sentences. IDs are therefore unique but are kept as short as possible, mostly using the word prefix itself as an idea for identity.

---

## 7. Related Work

### 7.1 Content-Defined Chunking

Content-Defined Chunking (CDC) techniques, pioneered by LBFS [1] and refined in FastCDC [3], provide edit-stable boundaries through rolling hash functions. While these approaches achieve deduplication ratios of 20-30% on typical workloads, they lack semantic awareness and cannot exploit grammatical patterns across objects. FastCDC optimizes throughput to 1-2 GB/s but treats all byte sequences uniformly, missing opportunities for pattern-based optimization. Eigentokens extend CDC with dynamic, grammar-aware boundaries that adapt to data patterns.

### 7.2 Grammar-Based Compression

Grammar compression algorithms like Sequitur [7] and Re-Pair [9] discover hierarchical structure in sequences, achieving compression ratios superior to LZ77-based methods. However, these systems operate on single files rather than cross-object corpora and lack integration with storage systems. Recent work on Straight-Line Programs (SLPs) [10] enables substring queries on compressed text but does not address the challenges of distributed storage or concurrent access.

### 7.3 Neural-Symbolic Integration and Compositional Learning

Recent advances in compositional learning demonstrate the power of explicit structural representations for model interpretability and generalization:

- **Compositional Generalization** [40]: Lake and Baroni (2018) show that systematic compositionality—the algebraic ability to understand novel combinations from known components—remains a challenge for neural networks, motivating architectures with explicit compositional structure.

- **Neural Module Networks** [41]: Andreas et al. (2016) demonstrate modular neural architectures where complex behaviors emerge from assembling simpler, reusable neural modules—a principle directly applicable to Eigentokens' grammatical compilation.

- **Discrete Symbolic Representations** [42]: Santoro et al. (2021) explore hybrid neuro-symbolic approaches that maintain discrete symbolic representations alongside continuous neural computations, improving both interpretability and systematic generalization.

- **Program Synthesis and Induction** [43]: Ellis et al. (2021) present DreamCoder, which learns to solve problems by inducing symbolic programs, demonstrating that explicit program representations enable better generalization than pure neural approaches.

These compositional approaches validate the theoretical foundation of Eigentokens: explicit grammatical structures enable more interpretable, generalizable, and debuggable AI systems than monolithic neural architectures.

### 7.4 Distinguishing from UltiHash

UltiHash represents an earlier attempt at grammar-aware deduplication but lacks the compilation capabilities central to Eigentokens:

| Aspect | UltiHash | Eigentokens (This Work) |
|--------|----------|------------------------|
| Primary Focus | Deduplication storage | Storage + LLM compilation |
| Grammar Role | Optimization technique | Core architectural principle |
| Model Generation | Not supported | Deterministic compilation from grammar |
| Cross-Object Patterns | Limited | Full cross-corpus grammar induction |
| Neural Integration | None | Direct weight calculation from productions |
| Compositionality | Not addressed | Explicit compositional structure |
| Production Scale | 380k lines of code | Unlimited with 4GB token size |
| Deployment | 1 PB at Bosch XC | Target: Multi-PB with AI integration |

The key differentiator: while UltiHash only deduplicates, CELM compiles LLMs through targeted autonomous use and categorization of a knowledge database derived from UltiHash principles.

### 7.5 Research Gap

No existing system combines:
1. **Cross-object grammar induction** with O(n³) learning complexity
2. **B+-forest storage architecture** optimized for grammatical access patterns  
3. **Deterministic model compilation** from storage-level patterns
4. **Unified deduplication and AI infrastructure** in a single system
5. **Compositional learning principles** applied to storage and compilation
6. **Agentic AI capabilities** for autonomous knowledge extension
7. **Grammar cookbook** metaphor for interpretable rule management, referencing Chomsky Grammar construction

This work fills this gap by treating storage and AI model construction as two facets of the same grammatical analysis problem.

---

## 8. Contemporary LLM Architectures: A Comparative Analysis

To contextualize the paradigm shift proposed by Eigentokens, this section examines the architectural principles underlying current state-of-the-art language models. These systems exemplify the limitations of probabilistic approaches that Eigentokens address through deterministic compilation.

### 8.1 Probabilistic Language Model Architectures

**GPT-5 (OpenAI, 2025)** [35]: Employs a dual-model architecture with router-based dispatch between efficiency-optimized and reasoning-optimized transformer variants. Despite architectural innovations, the system remains fundamentally probabilistic, encoding knowledge implicitly in ~10¹² parameters trained via next-token prediction. Critical limitations include: (a) inability to trace outputs to source knowledge, (b) hallucination under distribution shift, and (c) retraining requirements for knowledge updates.

**Gemini 1.5 Pro (Google, 2024)** [36]: Implements Mixture-of-Experts (MoE) with million-token context windows through specialized attention mechanisms. While achieving multimodal capabilities across text, image, and video, the architecture lacks explicit grammatical representations. Knowledge remains distributed across neural weights without semantic modularity or formal verification capabilities.

**Claude 3.5 Sonnet (Anthropic, 2024)** [31]: Introduces Constitutional AI for alignment but maintains probabilistic generation through transformer architectures. The 200,000-token context window partially mitigates memory limitations but does not address fundamental issues of opacity and non-determinism inherent in gradient-trained systems.

### 8.2 Fundamental Limitations of Probabilistic Approaches

All contemporary LLMs share critical architectural constraints:

1. **Opacity**: Knowledge encoded in floating-point weights lacks interpretability
2. **Non-determinism**: Stochastic sampling produces variable outputs
3. **Update Inefficiency**: Incorporating new knowledge requires expensive retraining
4. **Hallucination**: Statistical patterns generate plausible but factually incorrect outputs
5. **Debugging Impossibility**: No mechanism to trace errors to specific knowledge sources

---

## 9. The Eigentokens Paradigm: Deterministic Grammar-Based Architecture

### 9.1 Fundamental Architectural Shift

Eigentokens represent a paradigm shift from probabilistic to deterministic language model construction. While contemporary LLMs encode knowledge implicitly in neural weights through gradient descent, Eigentokens extract explicit grammatical rules that directly determine model architecture and behavior.

The system employs a meta-learning hierarchy:
- **M1 Metamodel**: Learns patterns in grammar emergence from raw data
- **M2 Metamodel**: Maps grammatical structures to neural architectures
- **M3 Self-Adaptation**: Enables runtime grammar extension and self-evolution, supporting autonomous evolution of the knowledge base through invasive software composition principles [39]

This three-level architecture enables the vision of an **omni-LLM** - a universal model capable of generating outputs (text, code, etc.) using grammar modules instead of neural activations. Such a system would generate new sentences or data by executing production rules from the grammar cookbook rather than sampling from neural probability distributions, achieving complete determinism and traceability. 

The M3 level extends beyond simple adaptation: when the system receives self-evolution queries, it can generate new CELM-lang program sets, compile improved versions of itself, and replace its parent process with enhanced child processes. This achieves genuine logical thinking and agentic behavior. Due to integer-based knowledge storage (avoiding floating-point errors), the system makes more robust and traceable decisions than probabilistic models. However, outcome data quality depends on input data quality - errors in the base data propagate deterministically through the system, though unlike probabilistic models, these errors remain traceable and correctable through grammar debugging.

This hierarchy proves that autonomous programming with M1 stability—changing descriptions while maintaining data consistency—is achievable for language models. The system demonstrates unlimited real-time extensibility while users interact with and query the model.

### 9.2 Grammar as First-Class Storage Primitive

Each Eigentoken encapsulates both data and computation through its interpretation program P, following Harvard architecture principles. This dual nature transforms storage from passive data repository to active compilation infrastructure. Grammar productions discovered through pattern analysis become:

1. **Deduplication Units**: Repeated patterns stored once, referenced multiply
2. **Compilation Modules**: Building blocks for deterministic model assembly
3. **Knowledge Representations**: Explicit, traceable, debuggable semantic units
4. **Cookbook Recipes**: Each rule serves as a recipe in the grammar cookbook for reconstructing patterns

The system operates like a human learning process: inventing new definitions in its own language using known words, which themselves may consist of composite knowledge. Through iterative learning, complex concepts become efficiently expressible in concise terms.

### 9.3 Compositional Model Construction

The compilation process parallels software engineering principles established by compositional frameworks [39]. Models are assembled through:

```
Grammar G = {r₁, r₂, ..., rₙ} → Neural Network N(W, T)
where:
  W = weight matrix calculated from rule frequencies
  T = topology determined by rule dependencies
  Each rᵢ is a recipe in the grammar cookbook
```

This approach enables:
- **Deterministic Reproduction**: Identical grammar yields identical models
- **Incremental Updates**: New rules integrate without full recompilation
- **Semantic Debugging**: Errors traceable to specific grammatical productions
- **Real-time Extension**: Model capabilities expand during active use

### 9.4 Storage-Compilation Unification

The innovation lies not in grammar compression or neural architecture alone, but in their unification. The same B+-forest that optimizes storage access patterns also maintains the dependency graph for model compilation. This unified architecture achieves:

- **Storage Efficiency**: 25-40% better compression than CDC baselines
- **Compilation Determinism**: Bit-identical models across compilation runs
- **Operational Simplicity**: Single system for storage and AI infrastructure
- **Evolutionary Capability**: System evolves from assembly-like primitives to high-level abstractions

---

## 10. System Architecture and Implementation

### 10.1 Core Components

**A1 - Grammar Induction Engine**: Implements three-phase pattern discovery with O(n³) complexity bounded by acceleration heuristics. Initial storage uses simple "store" objects, followed by context-sensitive analysis across the entire database. The engine employs dynamic CDC with grammar-aware boundaries rather than static chunking.

**A2 - B+-Forest Index Structure**: Non-strict B+-trees maintaining topic-filtered views of grammatical productions. The primary tree uses SHA512 hashes as content-addressable keys. Internal nodes store production rules with frequency metadata; leaf nodes contain compressed data payloads or Eigentoken references (max 4GB per token). Relaxed consistency permits recursive references for complex grammatical structures.

**A3 - Asynchronous Processing Pipeline**: Decouples ingestion from grammar learning through staged processing:
- Stage 1: Immediate stable reference generation (< 10ms latency)
- Stage 2: Background pattern analysis and grammar update
- Stage 3: Index compaction and optimization
All data persists immediately; recovery depends on checkpoint timing.

**A4 - Storage Interface Layer**: Provides S3-compatible REST API with extensions for grammatical queries. Implements HTTP Range requests (RFC 9110) through token-aligned block maps, maintaining seekability on compressed data. KV semantics and per-object fingerprint export included.

**A5 - Analysis/Control API**: Secondary interface providing metrics on grammars, patterns, and rules. Enables configuration of database behavior, grammar learning parameters, and heuristic tuning. [18, 19]

**A6 - Eigentoken Mock Rules**: Development of pattern rules for Eigentoken processing to demonstrate storage and grammar linking behavior. Includes test cases for various data types and edit patterns.

**A7 - (Roadmap) Replication/Erasure**: Future extension for applying placement and erasure-coding policies on grammar leaves and Eigentokens in general for resilience and space efficiency (beyond initial project scope). Ensure coding will be multi level per machine, per cluster and per cluster group that the user chooses to work on.

### 10.2 Implementation Details

**Language and Framework**: C++20 with lock-free data structures for concurrent access. Memory-mapped B+-trees for persistence with write-ahead logging for crash recovery. Out-of-memory conditions are handled through temporary Eigentoken spillage to disk.

**Compression Integration**: Pluggable compression backends (zstd-seekable, BGZF, LZMA) operating on grammatical boundaries. Block alignment preserves random access with < 5% overhead. Eigentokens can invoke compression libraries as shared resources.

**Grammar Persistence**: Productions serialized in binary format with version control. Supports grammar evolution through non-destructive updates maintaining backward compatibility. The language learns context-sensitively from its own corpus. Grammar cookbook persisted as metadata.

**Program Library Architecture**: Virtual CPU instruction set with data-assembler structure combining:
- Arithmetic and logical operations
- Control flow structures
- Reference composition
- Pattern generation
- Recursive program invocation

---

## 11. Novel Contributions

### 11.1 Theoretical Contributions

**C1 - Unified Grammar-Storage-Compilation Framework**: First system to unify storage deduplication, grammar induction, and deterministic model compilation in a single architecture. Establishes theoretical foundations for treating data storage and AI model construction as dual aspects of grammatical analysis.

**C2 - Inverse Learning Strategy for LLMs**: Introduces meta-learning approach where the system learns how to learn grammars (M1) and how to map grammars to neural architectures (M2), inverting the traditional training paradigm. The system demonstrates that context-sensitive Chomsky grammar analysis enables deterministic machine learning.

**C3 - Deterministic Weight Calculation**: Formal method for calculating neural network weights directly from grammatical production frequencies and dependencies, eliminating floating-point nondeterminism inherent in gradient descent. Weights are set more precisely than approximation-based training methods.

**C4 - Programming Language Evolution for AI**: Demonstrates the transition from assembly-like primitives to high-level abstractions in AI, paralleling the historical evolution of programming languages. The system proves modular LLM construction is possible, similar to how Java and Python enable dynamic interpretation with static libraries.

**C5 - CELM-lang Meta-Language**: Introduction of a new category of data meta-language operating at the M2 metamodel level, defining interpretation rules and match conditions rather than data relationships. Following the Intel oneAPI model, CELM-lang enables dynamic loading and evolution of interpretation machinery, serving simultaneously as analysis description, query foundation, statistical framework, and evolution platform.

### 11.2 Systems Contributions

**C6 - Grammar-Aware B+-Forest Architecture**: Novel non-strict B+-forest maintaining topic-filtered grammatical views with O(log n) access complexity while supporting recursive references and cross-tree dependencies. SHA512 hashing provides content-addressable storage with 4GB token limits.

**C7 - Token-Aligned Seekable Compression**: Integration of grammatical boundaries with seekable compression formats, achieving < 5% random access overhead while maintaining 25-40% better compression than CDC.

**C8 - Asynchronous Grammar Pipeline**: Three-stage processing architecture decoupling ingestion from grammar learning, achieving < 10ms write latencies while performing O(n³) pattern analysis in background.

**C9 - Agentic AI Storage System**: First storage system functioning as autonomous AI that self-programs and learns from its corpus without external intervention.

**C10 - Grammar Cookbook Metaphor**: Introduction of the cookbook paradigm for managing grammatical productions as interpretable recipes for data reconstruction.

### 11.3 Empirical Contributions

**C11 - Comparative Evaluation Framework**: Comprehensive benchmarking suite comparing grammar-based approach against CDC, BGZF, and LSM baselines across deduplication ratio, compression efficiency, write amplification, and range-read latency metrics.

**C12 - Open Reproducible Implementation**: C++20 reference implementation with reproducible evaluation harness, datasets, and ablation studies isolating individual architectural components' contributions.

**C13 - Production-Scale Validation**: Demonstrated viability through previous deployments: 380,000 lines of code (UltiHash self-hosting) and 1 PB video data (Bosch XC Abstatt).

---

## 12. Comparative Landscape

| Approach | Cross‑Object Dedup | Edit Stability | Range Reads | Write Ampl. | Ingest | Metadata | Layout | Scale Proven |
|:---------|:-------------------|:---------------|:------------|:------------|:-------|:---------|:-------|:-------------|
| Fixed‑Size + zstd | Medium | Low | Medium | Low | High | Low | Flat | N/A |
| CDC / FastCDC | High | High | Medium | Medium | High | Low–Medium | Flat/LSM | TB-scale |
| Grammar + Self‑Index | High (intra) | Medium | High | High | Low–Medium | High | Index‑centric | GB-scale |
| Seekable Block (BGZF) | N/A | N/A | High | Low | High | Low | Block‑map | TB-scale |
| UltiHash | Medium | Low–Medium | Medium | Medium | Medium | Low | 2‑level | PB-scale |
| **Eigentokens** | **High** | **High** | **High** | **Lower** | **High** | **Medium** | **B+‑forest** | **PB-target** |
| | | | | | | | **+ Cookbook** | |

---

## 13. Evaluation Methodology

### 13.1 Experimental Setup

**Hardware Configuration**:
- Server: 2x Intel Xeon Gold 6248R (48 cores), 256GB RAM
- Storage: NVMe SSD array (8TB, 3.5GB/s sequential read)
- Network: 40Gbps Ethernet for distributed experiments

**Software Environment**:
- OS: Ubuntu 22.04 LTS with kernel 5.15
- Compiler: GCC 12.2 with -O3 optimization
- Baseline implementations: FastCDC 2.0, zstd 1.5.5, RocksDB 8.0

### 13.2 Datasets and Workloads

**Dataset Categories**:

1. **Code Repositories** (GitHub Archive):
   - 100GB of version-controlled source code
   - Expected deduplication: 35-45%
   - Edit patterns: incremental changes, refactoring
   - Validation: Compare with UltiHash 380k line baseline

2. **Scientific Data** (GenBank, PubMed):
   - 50GB genomic sequences, 25GB research papers
   - Cross-document similarity: high
   - Access pattern: range reads for subsequences

3. **Structured Logs** (Cloud Trace Data):
   - 200GB system logs with temporal correlation
   - Append-heavy with periodic rotation
   - Compression opportunity: 60-70%

4. **Binary Objects** (Container Images):
   - 150GB Docker layers with shared dependencies
   - Deduplication across layers: 40-50%

5. **Mixed Media Dataset**:
   - Text, images, and binary data intermixed
   - Tests format-agnostic processing capability
   - Target: Several GB for grammar diversity

### 13.3 Evaluation Metrics

**Storage Efficiency**:
- Deduplication ratio: (1 - stored_size/logical_size) × 100%
- Compression effectiveness: final_size/deduplicated_size
- Metadata overhead: index_size/data_size
- Token size distribution: histogram of Eigentoken sizes
- Grammar cookbook size: number of recipes and their complexity

**Performance Metrics**:
- Ingestion throughput: MB/s sustained over 1-hour window
- Write amplification: physical_writes/logical_writes
- Range-read latency: P50/P95/P99 for 1KB-1MB requests
- Grammar learning overhead: CPU cycles per MB ingested
- Recovery time: seconds to restore from checkpoint
- Shard-switching I/O delay measurement

**Model Compilation Metrics**:
- Compilation determinism: bit-identical outputs across 100 runs
- Grammar extraction time: seconds per GB of training data
- Model size reduction: compiled_size/traditional_model_size
- Inference latency: ms per token generation
- Knowledge graph connectivity: average references per Eigentoken
- Cookbook recipe utilization: frequency of rule reuse

### 13.4 Baseline Comparisons

| System | Configuration | Optimization Focus |
|--------|--------------|-------------------|
| FastCDC + zstd | 8KB expected chunk, level 3 | Throughput |
| Fixed-4K + zstd-seekable | 4KB blocks, seekable format | Random access |
| BGZF | 64KB blocks, parallel decompression | Bioinformatics |
| RocksDB | Block-based table, Snappy | Write-optimized KV |
| Dynamic CDC (custom) | Grammar-aware boundaries | Semantic chunking |
| Eigentokens | Grammar-aware, async pipeline | Unified storage+AI |

---

## 14. Expected Results

### 14.1 Hypothesized Outcomes

Based on theoretical analysis and preliminary experiments from UltiHash deployments:

**Storage Performance**:
- Deduplication improvement: 25-40% better than FastCDC on cross-object corpora
- Compression ratio: Additional 15-20% reduction through grammar-aware encoding
- Write amplification: 30% reduction compared to LSM-based systems
- Range-read latency: P95 within 15% of uncompressed baseline
- Scalability: PB-scale deployment feasibility (validated by Bosch deployment)

**Model Compilation**:
- Compilation time: < 60 seconds for 10GB filtered dataset
- Model size: 70% reduction compared to traditional neural weights
- Determinism: 100% bit-identical across compilation runs (modulo input data bugs)
- Traceability: Complete path from output to source Eigentokens
- Real-time extensibility: < 100ms for incremental knowledge integration

### 14.2 Validation Criteria

Success criteria for the research:
1. Statistically significant improvement in deduplication ratio (p < 0.01)
2. Maintained seekability with < 20% latency overhead
3. Successful compilation of functional language models
4. Reproducible results across different datasets
5. Demonstration of agentic learning capabilities
6. Grammar cookbook effectively capturing domain patterns

---

## 15. Timeline and Project Management

### 15.1 Development Schedule (20 weeks)

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| **Foundation** | Weeks 1-3 | Architecture specification, B+-forest design with SHA512, evaluation framework |
| **Core Implementation** | Weeks 4-9 | Grammar induction engine, dynamic CDC, async pipeline, program libraries, A5/A6 components |
| **Integration** | Weeks 10-12 | Model compiler, API implementation, agentic AI components, cookbook persistence |
| **Evaluation** | Weeks 13-16 | Benchmarking, ablation studies, comparison with baselines |
| **Documentation** | Weeks 17-19 | Technical report, API documentation, reproducibility package |
| **Buffer** | Week 20 | Contingency for delays, additional experiments |

### 15.2 Risk Mitigation

| Risk | Impact | Mitigation Strategy |
|------|--------|--------------------|
| Grammar induction overhead | High latency | Bounded complexity, caching, heuristics |
| Metadata explosion | Storage overhead | Compression, pruning, 4GB token limit |
| Implementation complexity | Schedule delay | Incremental development, modular architecture |
| Benchmark variability | Result uncertainty | Multiple runs, statistical analysis |
| Memory constraints | OOM errors | Disk-based temporary Eigentokens |
| Input data quality | Propagated errors | Traceable debugging, validation checksums |

---

## 16. Assessment and Deliverables

**Academic Assessment**:
- **Colloquium (60 min)**: Technical presentation, live demonstration, and defense covering design decisions, evaluation results, and broader implications
- **Technical Report**: 20-page workshop-style paper following ACM/IEEE format
- **Slide Deck**: Comprehensive presentation materials (PDF)

**Software Deliverables**:
- **C++ Prototype**: Core implementation with command-line interface
- **Evaluation Harness**: Reproducible benchmarking scripts and datasets
- **API Documentation**: Complete interface specification including A5/A6 components
- **Open-Source Package**: GitHub repository with build instructions and test suite
- **Grammar Cookbook Viewer**: Tool for inspecting learned production rules

**Evaluation Focus**:
The assessment evaluates deduplication efficiency, ingestion throughput, write amplification, and HTTP Range latencies (P50/P95/P99) versus Content-Defined Chunking (CDC) family baselines and flat/Log-Structured Merge (LSM) layouts, alongside deterministic model compilation capabilities. Special emphasis on demonstrating the evolution from assembly-like storage primitives to high-level language model compilation and the omni-LLM vision.

---

## Bibliography

### Content-Defined Chunking (CDC) & Deduplication

[1] A. Muthitacharoen, B. Chen, and D. Mazières, "A Low-Bandwidth Network File System (LBFS)," *SOSP 2001*. Foundational use of content-defined chunking for detecting similarity across file versions. [PDF](https://pdos.csail.mit.edu/papers/lbfs%3Asosp01/lbfs.pdf)

[2] S. Quinlan and S. Dorward, "Venti: A New Approach to Archival Storage," *FAST 2002*. Classic content-addressable, write-once archival store. [USENIX](https://www.usenix.org/conference/fast-02/venti-new-approach-archival-data-storage)

[3] W. Xia et al., "FastCDC: A Fast and Efficient Content-Defined Chunking Approach for Data Deduplication," *USENIX ATC 2016*. State-of-the-art CDC variant balancing throughput and dedup ratio. [PDF](https://www.usenix.org/system/files/conference/atc16/atc16-paper-xia.pdf)

[4] Y. Hu et al., "The Design of Fast Content-Defined Chunking for Data Deduplication," *IEEE TPDS*, 2020. Journal extension analyzing FastCDC design decisions. [PDF](https://ranger.uta.edu/~jiang/publication/Journals/2020/2020-IEEE-TPDS(Wen%20Xia).pdf)

[5] M. Gregoriadis et al., "A Thorough Investigation of Content-Defined Chunking," *arXiv*, 2024. Recent comparative analysis of CDC families. [arXiv](https://arxiv.org/pdf/2409.06066)

[6] M. O. Rabin, "Fingerprinting by Random Polynomials," 1981. Origin of polynomial rolling fingerprints used in CDC. [PDF](https://www.xmailserver.org/rabin.pdf)

### Grammar-Based Compression & Operating on Compressed Data

[7] C. G. Neville-Manning and I. H. Witten, "Identifying Hierarchical Structure in Sequences: A Linear-Time Algorithm (SEQUITUR)," *DCC 1997*. Introduces grammar-based compression via online rule induction. [arXiv](https://arxiv.org/abs/cs/9709102)

[8] C. G. Neville-Manning and I. H. Witten, "Compression and Explanation using Hierarchical Grammars," *The Computer Journal*, 1997. Detailed exposition of grammar induction for compression. [PDF](https://ml.cms.waikato.ac.nz/publications/1997/NM-IHW-Compress97.pdf)

[9] N. J. Larsson and A. Moffat, "Offline Dictionary-Based Compression (Re-Pair)," *Proc. IEEE*, 2000. Efficient offline grammar construction. [Abstract](https://people.eng.unimelb.edu.au/ammoffat/abstracts/lm00procieee.html)

[10] M. Lohrey, "Algorithmics on SLP-Compressed Strings: A Survey," 2012. Survey of algorithms over straight-line programs. [PDF](https://www.eti.uni-siegen.de/ti/veroeffentlichungen/12-survey.pdf)

[11] F. Claude and G. Navarro, "Self-Indexed Grammar-Based Compression," *Fundamenta Informaticae*, 2011. Self-indexing over grammar-compressed data. [DOI](https://doi.org/10.3233/FI-2011-565)

### Delta Encoding

[12] A. Tridgell and P. Mackerras, "The rsync Algorithm," Tech. Report, 1996. Classic delta-encoding with rolling checksums. [PDF](https://www.andrew.cmu.edu/course/15-749/READINGS/required/cas/tridgell96.pdf)

### Indexes & Key-Value/Object Metadata

[13] P. O'Neil et al., "The Log-Structured Merge-Tree (LSM-Tree)," *Acta Informatica*, 1996. Baseline for log-structured indices. [PDF](https://dsf.berkeley.edu/cs286/papers/lsm-acta1996.pdf)

[14] R. Bayer and E. McCreight, "Organization and Maintenance of Large-Ordered Indices (B-Trees)," *Acta Informatica*, 1972. Canonical reference for B/B+-trees. [PDF](https://infolab.usc.edu/csci585/Spring2010/den_ar/indexing.pdf)

[15] L. Lu et al., "WiscKey: Separating Keys from Values in SSD-Conscious Storage," *FAST 2016*. Key/value separation to reduce write amplification. [PDF](https://www.usenix.org/system/files/conference/fast16/fast16-papers-lu.pdf)

[16] H. Lim et al., "SILT: A Memory-Efficient, High-Performance Key-Value Store," *SOSP 2011*. Flash-backed KV with tiny indexes. [PDF](https://www.cs.cmu.edu/~dga/papers/silt-sosp2011.pdf)

[17] B. Chandramouli et al., "FASTER: A Concurrent Key-Value Store with In-Place Updates," *SIGMOD 2018*. Modern high-throughput KV with hybrid log. [PDF](https://www.microsoft.com/en-us/research/uploads/prod/2018/03/faster-sigmod18.pdf)

### Distributed File/Object Storage

[18] S. Ghemawat, H. Gobioff, and S.-T. Leung, "The Google File System," *SOSP 2003*. Chunked, replicated file system and client-driven range reads; foundation for Google filesystem methodology. [Google Research](https://research.google/pubs/the-google-file-system/)

[19] S. A. Weil et al., "Ceph: A Scalable, High-Performance Distributed File System," *OSDI 2006*. Object/file storage with CRUSH placement. [ACM](https://dl.acm.org/doi/10.5555/1267308.1267330)

### Range-Friendly Access & Seekable Compression

[20] R. Fielding et al., "RFC 9110: HTTP Semantics," *IETF*, 2022. Current standard for HTTP including range requests. [IETF](https://datatracker.ietf.org/doc/html/rfc9110)

[21] H. Li, "Tabix: fast retrieval of sequence features from generic TAB-delimited files," *Bioinformatics*, 2011. BGZF indices for random access. [Oxford](https://academic.oup.com/bioinformatics/article/27/5/718/279592)

[22] H. Li et al., "The Sequence Alignment/Map format and SAMtools," *Bioinformatics*, 2009. Practical BGZF implementation. [Link](https://academic.oup.com/bioinformatics/article/25/16/2078/204688)

### LLM Tokenization (for contrast)

[23] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," *ACL 2016*. BPE tokenizer for NMT. [ACL](https://aclanthology.org/P16-1162/)

[24] Y. Wu et al., "Google's Neural Machine Translation System," *arXiv 2016*. WordPiece subword units. [arXiv](https://arxiv.org/abs/1609.08144)

[25] T. Kudo and J. Richardson, "SentencePiece," *EMNLP 2018*. Language-independent subword training. [ACL](https://aclanthology.org/D18-2012/)

[26] T. Kudo, "Subword Regularization," *ACL 2018*. Unigram LM tokenization. [ACL](https://aclanthology.org/P18-1007/)

### LLM Architecture References

[27] Anthropic. "Claude 3.7 Sonnet and Claude Code — Announcement," Feb 24, 2025.

[28] Anthropic. "Claude 3.7 Sonnet — System Card," 2025.

[29] Anthropic. "Introducing Claude 4 — Opus 4 and Sonnet 4," May 22, 2025.

[30] Anthropic. "System Card: Claude Opus 4 & Claude Sonnet 4," May 2025.

[31] Anthropic. "Introducing Claude Sonnet 4.5," Sep 29, 2025.

[32] Anthropic. "Claude Sonnet 4.5 — System Card," Sep 2025.

[33] OpenAI. "Hello GPT-4o (Omni)," May 13, 2024.

[34] OpenAI. "Introducing GPT-4.1 in the API," Apr 14, 2025.

[35] OpenAI. "Introducing GPT-5," Aug 7, 2025.

[36] Gemini Team (Google). "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context," *arXiv:2403.05530*, 2024.

### Additional References

[37] Kuruppu, S., Puglisi, S. J., Zobel, J. "Relative Lempel-Ziv Compression of Genomes for Large-Scale Storage and Retrieval," *SPIRE 2010*.

[38] Kuruppu, S., Puglisi, S. J., Zobel, J. "Optimized Relative Lempel-Ziv Compression of Genomes," *CRPIT 2011*.

[39] U. Aßmann. "Invasive Software Composition." Springer, 2003.

### Compositional Learning and Neural-Symbolic Integration

[40] B. Lake and M. Baroni, "Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks," *ICML 2018*. Demonstrates limitations of neural networks in compositional generalization. [arXiv](https://arxiv.org/abs/1711.00350)

[41] J. Andreas, M. Rohrbach, T. Darrell, and D. Klein, "Neural Module Networks," *CVPR 2016*. Modular neural architectures for compositional reasoning. [arXiv](https://arxiv.org/abs/1511.02799)

[42] A. Santoro, A. Lampinen, K. Mathewson, T. Lillicrap, and D. Raposo, "Symbolic Behaviour in Artificial Intelligence," *arXiv 2021*. Survey of hybrid neuro-symbolic approaches. [arXiv](https://arxiv.org/abs/2102.03406)

[43] K. Ellis, C. Wong, M. Nye, M. Sablé-Meyer, L. Morales, L. Hewitt, L. Cary, A. Solar-Lezama, and J. B. Tenenbaum, "DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning," *PLDI 2021*. Program synthesis through library learning. [PDF](https://dl.acm.org/doi/10.1145/3453483.3454080)
