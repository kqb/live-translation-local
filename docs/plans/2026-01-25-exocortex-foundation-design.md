# Exocortex Foundation: Personal Memory Externalization System

**Date**: January 25, 2026
**Status**: Design Complete - Ready for Implementation
**Vision**: Building a fully local, privacy-first system for externalizing biological memory with semantic search, multi-modal linkage, and psychological health monitoring.

---

## 1. Executive Summary

This system transforms the live-translation project into the foundation of a personal exocortex - an externalized memory system that captures, indexes, consolidates, and retrieves life experiences. Unlike simple note-taking or logging systems, this architecture treats memories as interconnected, multi-modal experiences that evolve over time, with active consolidation processes that mirror biological memory formation.

**Core Principles:**
- **Privacy First**: All data stored and processed locally, with encryption for sensitive memories
- **Multi-Modal**: Audio, visual, digital, and biometric data unified under single memory model
- **Edge-Ready**: Designed from ground up for distributed deployment across edge nodes (G2 glasses, phone, desktop)
- **Psychologically Aware**: Monitors linguistic patterns to detect manipulation, abuse, or mental health decline
- **Future-Proof**: Extensible architecture supporting decades of memory accumulation

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT SOURCES (Modular)                      │
├─────────────────────────────────────────────────────────────────┤
│ • Translation Pipeline (audio → transcription → translation)     │
│ • Continuous Audio Capture (ambient recording)                   │
│ • [Future] Screen Capture (periodic screenshots + OCR)           │
│ • [Future] Photo/Camera (timestamped images)                     │
│ • [Future] Digital Activity (browser, calendar, emails)          │
│ • [Future] Biometric Context (location, activity, health)        │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  MEMORY CAPTURE PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│ • Raw data ingestion                                             │
│ • Source-specific preprocessing (transcription, OCR, etc.)       │
│ • Entity extraction (people, places, projects, concepts)         │
│ • Topic modeling & classification                                │
│ • Embedding generation (sentence-transformers)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  MEMORY PROCESSING ENGINE                        │
├─────────────────────────────────────────────────────────────────┤
│ • Entity linking (cross-modal, e.g., voice + face + email)       │
│ • Relationship detection (builds on, contradicts, relates to)    │
│ • Temporal clustering (group into coherent experiences)          │
│ • Semantic bridging (link across modalities by topic)            │
│ • Importance scoring (initial + dynamic rescoring)               │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED MEMORY STORE                          │
├─────────────────────────────────────────────────────────────────┤
│ • Qdrant (vector embeddings for semantic search)                 │
│   - Local embedded mode                                          │
│   - 384-dim vectors (all-MiniLM-L6-v2)                          │
│   - Collections: memories, experiences, entities                 │
│                                                                  │
│ • Neo4j (knowledge graph for relationships)                      │
│   - Embedded mode (local instance)                               │
│   - Nodes: Memory, Experience, Entity, Topic, Temporal           │
│   - Edges: PART_OF, MENTIONS, ABOUT, RELATES_TO, etc.           │
│                                                                  │
│ • SQLite (structured metadata & temporal index)                  │
│   - Full memory objects                                          │
│   - Fast temporal queries                                        │
│   - Backup/export capability                                     │
│                                                                  │
│ • File System (raw audio, images, documents)                     │
│   - Organized by date: ./data/memories/YYYY/MM/DD/               │
│   - Deduplication via content hashing                            │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              MEMORY CONSOLIDATION (Background)                   │
├─────────────────────────────────────────────────────────────────┤
│ • Hourly: Experience clustering, entity linking                  │
│ • Daily: Summaries, pattern detection, importance rescoring      │
│ • Weekly: Topic evolution, knowledge synthesis                   │
│ • Monthly: Long-term trends, linguistic health analysis          │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL INTERFACE                          │
├─────────────────────────────────────────────────────────────────┤
│ • Semantic Search (natural language queries)                     │
│ • Temporal Navigation (timeline browsing)                        │
│ • Entity-Centric (person/topic focused)                          │
│ • Graph Traversal (associative recall)                           │
│ • Real-Time Context (G2 glasses integration)                     │
│ • Export & RAG (markdown, audio, LLM integration)                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

**Memory Creation Flow:**
1. Input source captures raw data (e.g., audio chunk)
2. Source-specific processing (e.g., Whisper transcription, speaker diarization)
3. Entity extraction (NER for people, places, projects)
4. Embedding generation (sentence-transformers on text)
5. Storage: Qdrant (embedding), Neo4j (graph relationships), SQLite (metadata)
6. Background: Temporal clustering into experiences

**Memory Retrieval Flow:**
1. User query (natural language or structured)
2. Query embedding generation
3. Qdrant semantic search (top-k similar memories)
4. Neo4j graph traversal (relationship filtering)
5. SQLite metadata filtering (time, speaker, etc.)
6. Result ranking & presentation
7. Optional: Audio playback, related memory suggestions

---

## 3. Core Components

### 3.1 Unified Memory Model

**Schema:**
```python
@dataclass
class Memory:
    # Core Identity
    memory_id: str                    # UUID
    timestamp: datetime               # Capture time
    source_type: str                  # "audio", "screen", "photo", "digital", "biometric"
    source_id: str                    # Capture session/device ID

    # Content (Multi-Modal)
    text: Optional[str]               # Transcription, OCR, extracted text
    audio_path: Optional[str]         # Path to audio file
    image_path: Optional[str]         # Path to image/screenshot
    raw_data: Optional[dict]          # Source-specific data

    # Semantic Understanding (Auto-Extracted)
    entities: List[Entity]            # People, places, projects, concepts
    topics: List[str]                 # What this memory is about
    sentiment: Optional[float]        # Emotional tone (-1 to 1)
    importance: float                 # Relevance score (0-1, dynamic)

    # Context
    location: Optional[tuple]         # GPS coordinates
    activity: Optional[str]           # What you were doing
    related_memories: List[str]       # Connected memory IDs

    # Retrieval
    embedding: np.ndarray             # 384-dim vector
    summary: Optional[str]            # LLM-generated summary

@dataclass
class Entity:
    entity_id: str
    type: str                         # "person", "place", "project", "concept"
    name: str
    appearances: List[tuple]          # (memory_id, modality, identifier)
    # For people: voice_id, face_embedding, email, etc.

@dataclass
class Experience:
    """Clustered memories forming coherent episode."""
    cluster_id: str
    start_time: datetime
    end_time: datetime
    primary_activity: str             # "meeting", "coding", "reading"

    audio_memories: List[Memory]
    visual_memories: List[Memory]
    digital_memories: List[Memory]
    biometric_context: dict

    title: str                        # "Meeting with John about Project X"
    summary: str                      # LLM-generated
    key_entities: List[Entity]
```

**Storage Strategy:**
- **Qdrant**: Embeddings + minimal metadata for fast vector search
- **Neo4j**: Full graph relationships (Memory → Entity, Experience → Topic)
- **SQLite**: Complete memory objects, temporal index, metadata queries
- **File System**: Raw files (audio/images) organized by date

### 3.2 Experience Graph (Neo4j)

**Node Types:**
- **Memory**: Individual memory points
- **Experience**: Clustered memories (coherent episodes)
- **Entity**: People, places, projects, concepts
- **Topic**: Semantic themes
- **Temporal**: Dates, time periods

**Edge Types:**
- `PART_OF`: Memory → Experience
- `MENTIONS`: Memory → Entity
- `ABOUT`: Memory → Topic
- `PRECEDES/FOLLOWS`: Memory → Memory (temporal sequence)
- `CO_OCCURS`: Entity ↔ Entity (seen together)
- `EVOLVES_FROM`: Topic → Topic (idea evolution)
- `RELATES_TO`: Experience ↔ Experience (thematic connection)
- `SPEAKER_IN`: Person → Memory (who said it)
- `DISCUSSES`: Person → Topic (frequency tracked)

**Example Queries (Cypher):**
```cypher
// "What did John say about AI Safety?"
MATCH (p:Person {name: "John"})-[:SPEAKER_IN]->(m:Memory)-[:ABOUT]->(t:Topic {name: "AI Safety"})
RETURN m.text, m.timestamp, m.audio_path
ORDER BY m.timestamp DESC

// "Show evolution of my thinking on Project X"
MATCH path = (t1:Topic {name: "Project X"})-[:EVOLVES_FROM*]->(t2:Topic)
RETURN path

// "Find meetings where Sarah and John were both present"
MATCH (e:Experience)-[:INVOLVES]->(p1:Person {name: "Sarah"}),
      (e)-[:INVOLVES]->(p2:Person {name: "John"})
WHERE e.primary_activity = "meeting"
RETURN e.title, e.start_time, e.summary
```

### 3.3 Memory Consolidation Engine

**Background Processing Schedule:**

**Real-Time (during capture):**
- Basic indexing (embeddings, graph nodes)
- Entity extraction (NER)
- Speaker identification (voice embeddings)

**Hourly:**
- Experience clustering (temporal proximity + semantic similarity)
- Cross-modal entity linking (voice + face + email)
- Relationship detection (semantic bridges across modalities)

**Daily (midnight):**
- Daily summary generation (LLM-powered)
- Pattern detection (temporal routines, social patterns)
- Memory importance rescoring
- Linguistic health analysis (vocabulary drift tracking)

**Weekly (Sunday):**
- Topic evolution analysis
- Knowledge synthesis (LLM generates topic summaries)
- Social graph analysis (who you interact with, about what)

**Monthly:**
- Long-term trend analysis
- Memory consolidation (reduce redundancy, strengthen important memories)
- Comprehensive linguistic health report

**Consolidation Tasks:**

1. **Daily Summaries**
```python
{
    "date": "2024-01-25",
    "key_experiences": [
        "Meeting with John about Project X deadline",
        "Research session on AI Safety",
        "Conversation with Sarah about blockchain"
    ],
    "people_met": ["John", "Sarah"],
    "topics_explored": ["Project X", "AI Safety", "blockchain"],
    "emotional_tone": 0.3,  # Slightly positive
    "productivity_score": 0.8,
    "insights": "Discovered connection between AI Safety and blockchain governance models"
}
```

2. **Topic Evolution Tracking**
```python
{
    "topic": "AI Safety",
    "first_mention": "2024-01-10",
    "frequency_trend": [2, 5, 3, 8, 12],  # Weekly mentions
    "perspective_shifts": [
        {
            "date": "2024-01-15",
            "shift": "Changed from skepticism to cautious optimism",
            "evidence": [memory_ids]
        }
    ],
    "related_topics_over_time": {
        "2024-01-10": ["machine learning"],
        "2024-01-20": ["machine learning", "blockchain", "governance"]
    }
}
```

3. **Pattern Detection**
- Weekly patterns: "Every Monday you discuss Project X"
- Social patterns: "You meet John every 2-3 weeks"
- Topic clustering: "AI Safety often discussed with blockchain"
- Temporal patterns: "You're most productive 9-11 AM"

### 3.4 Linguistic Health Monitoring

**Critical Feature: Abuse & Manipulation Detection**

Monitors vocabulary drift to detect psychological manipulation, abusive relationships, or mental health decline.

**Tracked Indicators:**
```python
concern_indicators = {
    "assertiveness_decline": [
        "I think", "maybe", "I guess", "sort of", "kind of",
        "I don't know", "whatever", "doesn't matter"
    ],
    "apologetic_increase": [
        "sorry", "my bad", "my fault", "I apologize",
        "didn't mean to", "excuse me"
    ],
    "self_doubt": [
        "am I wrong?", "did I?", "was I?", "I'm confused",
        "I must be", "I'm probably"
    ],
    "isolation_language": [
        "alone", "nobody", "no one understands", "by myself",
        "they all", "everyone thinks"
    ],
    "cognitive_distortion": [
        "always", "never", "everyone", "no one", "terrible",
        "awful", "disaster", "ruined"
    ]
}
```

**Analysis:**
- Establish 30-day baseline linguistic profile
- Track drift over time (weekly/monthly)
- Alert if multiple red flags (30%+ assertiveness decline + 50%+ apologetic increase)
- Person-specific analysis: "Do I speak differently around this person?"
- Temporal correlation: "Did this shift coincide with a relationship/event?"

**Privacy-First Design:**
- All analysis runs **locally only**
- Alerts shown only to you (never shared)
- Option to encrypt memories involving specific people
- Complete user control over monitoring settings

**Integration:**
- Subtle G2 glasses notifications for health check-ins
- Weekly linguistic health reports
- Correlation with sleep, activity, social patterns

### 3.5 Retrieval & Query Interface

**Query Modes:**

**1. Semantic Search (Natural Language)**
```bash
exo recall "what did John say about the deadline?"
exo recall "conversations about AI safety"
exo recall "when did I last see Sarah?"
```

**2. Temporal Navigation**
```bash
exo timeline 2024-01-25          # All experiences from that day
exo timeline week                # This week's summary
exo timeline month --topic "Project X"  # Monthly filtered view
```

**3. Entity-Centric Queries**
```bash
# Person-focused
exo person "John" --summary           # All interactions, summarized
exo person "John" --topics            # Discussion topics
exo person "John" --language-drift    # Linguistic health check

# Topic-focused
exo topic "AI Safety" --evolution     # How thinking evolved
exo topic "Project X" --timeline      # Chronological view
```

**4. Graph Traversal (Associative Recall)**
```bash
exo graph explore "blockchain"
# → Interactive navigation through related topics, people, time clusters

exo query "MATCH (p:Person)-[:DISCUSSES]->(t:Topic {name: 'AI'}) RETURN p, count(*)"
# → Cypher queries for power users
```

**5. Real-Time Contextual Recall (G2 Integration)**
During conversations, provide relevant context on glasses:
- "Last talked to John 3 days ago about deadline"
- "Sarah mentioned this topic 2 weeks ago with different opinion"
- "Reminder: You wanted to ask about budget"

**6. Export & RAG Integration**
```bash
exo export --topic "AI" --format markdown > ai_research.md
exo export --person "John" --with-audio --zip > john_conversations.zip
exo rag "Summarize my understanding of quantum computing" --memories 20
```

**7. Privacy Controls**
```bash
exo encrypt --person "Therapist" --password <pwd>
exo encrypt --topic "Medical" --password <pwd>
exo redact --contains "password" --contains "credit card"
exo isolate --person "Ex-partner"  # Quarantine memories
```

---

## 4. Technical Stack

### 4.1 Core Dependencies

**Vector Database:**
- **Qdrant** (embedded mode): Local vector storage, semantic search
- Version: Latest stable (0.11+)
- Storage: `./data/memories/qdrant/`

**Graph Database:**
- **Neo4j** (embedded mode): Knowledge graph relationships
- Version: Latest community edition (5.x+)
- Storage: `./data/memories/neo4j/`

**Relational Database:**
- **SQLite**: Metadata, temporal index, backup
- Storage: `./data/memories/memory.db`

**Embedding Model:**
- **sentence-transformers/all-MiniLM-L6-v2**
- Size: 80MB
- Dimensions: 384
- Local inference (CPU/GPU)
- Future: Export to ONNX for edge deployment

**NLP & Processing:**
- **spaCy** (en_core_web_sm): Entity extraction
- **transformers**: Topic modeling (optional)
- **Local LLM** (llama.cpp/Ollama): Summaries, synthesis (optional)

**Existing Infrastructure:**
- **Whisper** (faster-whisper): Audio transcription
- **pyannote.audio**: Speaker diarization
- **Speaker recognition**: Voice embeddings (existing system)

### 4.2 New Dependencies

```toml
# Add to pyproject.toml
dependencies = [
    # ... existing dependencies ...
    "qdrant-client>=1.7.0",          # Vector database
    "neo4j>=5.15.0",                 # Graph database
    "sentence-transformers>=2.2.0",   # Embeddings
    "spacy>=3.7.0",                  # NER
]
```

### 4.3 File Structure

```
live-translation-local/
├── src/
│   ├── exocortex/                   # New exocortex package
│   │   ├── __init__.py
│   │   ├── memory.py                # Memory, Entity, Experience models
│   │   ├── capture.py               # Input source plugins
│   │   ├── indexer.py               # Embedding + graph indexing
│   │   ├── graph.py                 # Neo4j graph operations
│   │   ├── consolidation.py         # Background processing
│   │   ├── linguistic_health.py     # Vocabulary drift monitoring
│   │   ├── retrieval.py             # Query interface
│   │   └── cli.py                   # CLI commands (exo)
│   ├── pipeline.py                  # Existing (integrate with capture)
│   ├── session_logger.py            # Existing (integrate with indexer)
│   └── speaker_recognition.py       # Existing (integrate with entities)
├── data/
│   ├── memories/
│   │   ├── qdrant/                  # Vector DB storage
│   │   ├── neo4j/                   # Graph DB storage
│   │   ├── memory.db                # SQLite metadata
│   │   └── YYYY/MM/DD/              # Raw files by date
│   ├── sessions/                    # Existing session logs
│   └── speakers.json                # Existing speaker DB
├── docs/
│   └── plans/
│       └── 2026-01-25-exocortex-foundation-design.md
└── config.yaml
```

---

## 5. Privacy & Security

### 5.1 Local-First Architecture

**All processing happens locally:**
- Embedding generation: Local sentence-transformers
- Vector search: Local Qdrant instance
- Graph queries: Local Neo4j instance
- LLM synthesis: Local llama.cpp/Ollama (optional)
- No cloud dependencies after initial model downloads

### 5.2 Encryption

**Selective Encryption:**
```python
# Encrypt memories involving specific people/topics
exo encrypt --person "Therapist" --password <pwd>
exo encrypt --topic "Medical" --password <pwd>

# Encrypted memories stored with AES-256
# Decryption required for retrieval
# Key derived from user password (PBKDF2)
```

**At-Rest Encryption:**
- Option to encrypt entire `./data/memories/` directory
- LUKS/FileVault/BitLocker integration
- Passphrase-protected database files

### 5.3 Access Control

**Memory Isolation:**
- Quarantine specific memories (hidden from search)
- Delete without trace (secure wipe)
- Export restrictions (prevent accidental sharing)

**Audit Logging:**
- Track all memory access (who/what/when)
- Detect unusual query patterns
- Alert on potential data exfiltration

---

## 6. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Basic memory capture and retrieval working

1. Set up storage infrastructure
   - Initialize Qdrant (embedded mode)
   - Initialize Neo4j (embedded mode)
   - Create SQLite schema

2. Implement Memory model
   - Core dataclasses (Memory, Entity, Experience)
   - Serialization/deserialization

3. Integrate with existing translation pipeline
   - Capture audio memories from SessionLogger
   - Generate embeddings (sentence-transformers)
   - Store in Qdrant + SQLite

4. Basic CLI retrieval
   - `exo recall "query"` → semantic search
   - Display results with context

**Deliverable**: Can capture audio memories and search them semantically

### Phase 2: Graph & Entity Linking (Weeks 3-4)
**Goal**: Multi-modal memory linkage via knowledge graph

1. Neo4j graph integration
   - Define node/edge schemas
   - Create graph indexing pipeline

2. Entity extraction
   - Integrate spaCy NER
   - Link speaker recognition to Person entities
   - Cross-modal entity linking (voice → graph)

3. Experience clustering
   - Temporal proximity grouping
   - Semantic similarity bridging
   - Auto-generate experience summaries

4. Graph queries
   - Cypher query interface
   - Entity-centric retrieval

**Deliverable**: Memories linked via knowledge graph, entity-based queries work

### Phase 3: Consolidation Engine (Weeks 5-6)
**Goal**: Active memory processing and pattern detection

1. Background processing framework
   - Scheduler (hourly/daily/weekly tasks)
   - Task queue and execution

2. Daily summaries
   - Aggregate day's memories
   - LLM-generated insights (optional)

3. Topic evolution tracking
   - Compare embeddings over time
   - Detect perspective shifts

4. Pattern detection
   - Temporal routines
   - Social patterns
   - Topic co-occurrence

**Deliverable**: System actively consolidates and analyzes memories

### Phase 4: Linguistic Health Monitoring (Week 7)
**Goal**: Psychological safety features

1. Baseline establishment
   - First 30 days of vocabulary profiling
   - Category frequency tracking

2. Drift detection
   - Weekly vocabulary analysis
   - Person-specific comparisons
   - Alert triggering

3. Health reporting
   - Weekly summaries
   - G2 glasses integration for check-ins

**Deliverable**: System detects and alerts on concerning linguistic patterns

### Phase 5: Enhanced Retrieval (Week 8)
**Goal**: Rich query interface and G2 integration

1. Advanced CLI commands
   - Timeline navigation
   - Graph exploration
   - Export functionality

2. Real-time contextual recall
   - G2 glasses notifications during conversations
   - Relevant memory suggestions

3. Privacy controls
   - Encryption commands
   - Memory isolation
   - Access audit logs

**Deliverable**: Full-featured retrieval interface with privacy controls

### Phase 6: Future Input Sources (Future)
**Goal**: Extend beyond audio

1. Screen capture plugin
   - Periodic screenshots
   - OCR extraction
   - Link to concurrent audio

2. Photo/camera integration
   - Timestamped images
   - Face recognition
   - Location tagging

3. Digital activity tracking
   - Browser history
   - Calendar events
   - Email metadata

**Deliverable**: Multi-modal memory capture

---

## 7. Future Extensions

### 7.1 Distributed Edge Deployment

**Architecture:**
```
Edge Node 1 (G2 + Phone)          Edge Node 2 (Desktop)
├─ Local Qdrant                   ├─ Local Qdrant
├─ Local embeddings                ├─ Local embeddings
├─ Continuous capture              ├─ Screen capture
└─ Real-time retrieval             └─ Heavy processing
        ↓                                  ↓
        └──────── Sync via gRPC ──────────┘
                        ↓
              Central Node (optional)
              └─ Aggregated Qdrant cluster
```

**Sync Strategy:**
- Embeddings are version-locked (same model across nodes)
- Periodic sync via Qdrant's built-in replication
- Conflict resolution via timestamp + device priority
- Offline-capable (sync when connected)

### 7.2 Advanced Features

**Emotional Intelligence:**
- Sentiment analysis over time
- Stress detection (voice prosody + vocabulary)
- Mood tracking correlated with memories

**Predictive Insights:**
- "You usually feel stressed on Monday mornings"
- "This topic tends to lead to important insights"
- "People who discuss X also mention Y"

**Collaborative Memories:**
- Shared experiences with consent
- Different perspectives on same event
- Privacy-preserving: Only metadata shared

**Memory Replay:**
- Audio playback of past conversations
- Timeline visualization of experiences
- "What was I doing last year at this time?"

---

## 8. Success Metrics

**Phase 1-2 (Foundation):**
- Successfully capture and retrieve 1000+ memories
- Sub-second semantic search latency
- Graph queries execute in <100ms

**Phase 3-4 (Consolidation):**
- Daily summaries generated within 5 minutes
- Pattern detection identifies 3+ meaningful routines
- Linguistic health baseline established

**Phase 5 (Retrieval):**
- 90%+ retrieval accuracy for known memories
- Real-time G2 notifications within 500ms
- Zero privacy breaches or accidental data exposure

**Long-Term:**
- System scales to 10+ years of continuous capture
- Edge deployment successfully syncs across 3+ devices
- Linguistic health alerts prevent 1+ harmful situations

---

## 9. Ethical Considerations

**Consent & Privacy:**
- Never record others without explicit consent
- Clear indicators when system is capturing (LED, notification)
- Easy opt-out for specific people/situations

**Data Ownership:**
- You own all memories (no third-party access)
- Export capability in open formats (JSON, markdown)
- Right to delete (secure wipe, no recovery)

**Psychological Impact:**
- Linguistic health monitoring is opt-in
- Alerts are informational, not diagnostic
- Encourage professional help when needed
- Avoid over-reliance on exocortex (cognitive atrophy risk)

**Security Responsibility:**
- User must secure device (encryption, strong passwords)
- Backup strategy for irreplaceable memories
- Incident response plan for device loss/theft

---

## 10. Conclusion

This exocortex foundation transforms the live-translation system into a comprehensive personal memory externalization platform. By combining local vector search (Qdrant), knowledge graphs (Neo4j), active consolidation, and psychological safety features, it creates a system that not only stores memories but actively helps you understand your life, protect your mental health, and build knowledge over decades.

The architecture is designed for edge deployment, privacy-first operation, and extensibility to multi-modal input sources. Starting with audio/translation as the primary input source provides immediate value while establishing infrastructure for future expansion.

**Next Steps:**
1. Review and approve this design
2. Create implementation plan with detailed tasks
3. Set up git worktree for isolated development
4. Begin Phase 1 implementation

---

**Design Status**: ✅ Complete - Ready for Implementation
**Reviewed By**: User + Claude
**Approved**: Pending
