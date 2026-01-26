#!/usr/bin/env python3
"""Exocortex Dashboard - Comprehensive monitoring and management interface.

Provides an interactive interface to:
- System overview with statistics and charts
- Full-text memory search with filters
- Speaker enrollment and management
- Live pipeline status monitoring
- Audio playback with segment controls
"""

import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request, send_file
from rich.console import Console

from src.speaker_recognition import SpeakerRecognizer

console = Console()

app = Flask(__name__)

# Configuration
EXOCORTEX_PATH = Path.home() / ".exocortex" / "memories"
SPEAKER_DB_PATH = Path("./data/speakers.json")


def get_db_connection():
    """Get SQLite database connection."""
    db_path = EXOCORTEX_PATH / "metadata.db"
    return sqlite3.connect(str(db_path))


def get_memories_by_speaker() -> dict[str, list[dict[str, Any]]]:
    """Group memories by speaker label."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT data FROM memories")
    rows = cursor.fetchall()
    conn.close()

    grouped = {}
    for (data_json,) in rows:
        data = json.loads(data_json)
        # Handle None values - use "or" to catch both missing keys and null values
        speaker_label = data.get("speaker_label") or "UNKNOWN"

        if speaker_label not in grouped:
            grouped[speaker_label] = []

        metadata = data.get("metadata", {})
        grouped[speaker_label].append(
            {
                "memory_id": data.get("memory_id") or "",
                "text": data.get("text") or "",
                "audio_path": data.get("audio_path") or "",
                "timestamp": metadata.get("timestamp") or "",
                "start_time": metadata.get("start_time", 0),
                "end_time": metadata.get("end_time", 0),
                "speaker_label": speaker_label,
                "speaker_name": data.get("speaker_name") or "",
            }
        )

    return grouped


def update_speaker_name(speaker_label: str, speaker_name: str) -> int:
    """Update all memories with a speaker label to have the speaker name."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT memory_id, data FROM memories")
    rows = cursor.fetchall()

    updated_count = 0
    for memory_id, data_json in rows:
        data = json.loads(data_json)
        if data.get("speaker_label") == speaker_label:
            data["speaker_name"] = speaker_name
            cursor.execute(
                "UPDATE memories SET data = ? WHERE memory_id = ?",
                (json.dumps(data), memory_id),
            )
            updated_count += 1

    conn.commit()
    conn.close()

    return updated_count


def get_system_stats() -> dict[str, Any]:
    """Get comprehensive system statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total memories
    cursor.execute("SELECT COUNT(*) FROM memories")
    total_memories = cursor.fetchone()[0]

    # Memories with speaker labels
    cursor.execute(
        "SELECT COUNT(*) FROM memories WHERE json_extract(data, '$.speaker_label') IS NOT NULL"
    )
    with_speakers = cursor.fetchone()[0]

    # Unique audio files
    cursor.execute(
        "SELECT COUNT(DISTINCT json_extract(data, '$.audio_path')) FROM memories WHERE json_extract(data, '$.audio_path') IS NOT NULL"
    )
    unique_files = cursor.fetchone()[0]

    # Speaker distribution
    cursor.execute(
        """
        SELECT
            json_extract(data, '$.speaker_label') as speaker,
            COUNT(*) as count
        FROM memories
        WHERE speaker IS NOT NULL
        GROUP BY speaker
        ORDER BY count DESC
    """
    )
    speaker_dist = [{"label": row[0], "count": row[1]} for row in cursor.fetchall()]

    # Recent memories (last 24 hours)
    cursor.execute(
        """
        SELECT COUNT(*) FROM memories
        WHERE datetime(json_extract(data, '$.metadata.timestamp')) > datetime('now', '-1 day')
    """
    )
    recent_count = cursor.fetchone()[0]

    # Language distribution
    cursor.execute(
        """
        SELECT
            json_extract(data, '$.language') as lang,
            COUNT(*) as count
        FROM memories
        WHERE lang IS NOT NULL
        GROUP BY lang
        ORDER BY count DESC
        LIMIT 10
    """
    )
    lang_dist = [{"language": row[0] or "unknown", "count": row[1]} for row in cursor.fetchall()]

    conn.close()

    # Storage size
    db_size = (EXOCORTEX_PATH / "metadata.db").stat().st_size if (EXOCORTEX_PATH / "metadata.db").exists() else 0
    qdrant_path = EXOCORTEX_PATH / "qdrant"
    qdrant_size = sum(f.stat().st_size for f in qdrant_path.rglob("*") if f.is_file()) if qdrant_path.exists() else 0

    return {
        "total_memories": total_memories,
        "with_speakers": with_speakers,
        "unique_audio_files": unique_files,
        "speaker_distribution": speaker_dist,
        "language_distribution": lang_dist,
        "recent_memories": recent_count,
        "storage": {
            "database_mb": round(db_size / (1024 * 1024), 2),
            "vectors_mb": round(qdrant_size / (1024 * 1024), 2),
            "total_mb": round((db_size + qdrant_size) / (1024 * 1024), 2),
        },
    }


def search_memories(query: str, limit: int = 50, speaker_filter: str = None) -> list[dict]:
    """Search memories by text content."""
    conn = get_db_connection()
    cursor = conn.cursor()

    if speaker_filter and speaker_filter != "all":
        cursor.execute(
            """
            SELECT data FROM memories
            WHERE json_extract(data, '$.text') LIKE ?
            AND json_extract(data, '$.speaker_label') = ?
            ORDER BY json_extract(data, '$.metadata.timestamp') DESC
            LIMIT ?
        """,
            (f"%{query}%", speaker_filter, limit),
        )
    else:
        cursor.execute(
            """
            SELECT data FROM memories
            WHERE json_extract(data, '$.text') LIKE ?
            ORDER BY json_extract(data, '$.metadata.timestamp') DESC
            LIMIT ?
        """,
            (f"%{query}%", limit),
        )

    results = []
    for (data_json,) in cursor.fetchall():
        data = json.loads(data_json)
        metadata = data.get("metadata", {})
        results.append(
            {
                "memory_id": data.get("memory_id") or "",
                "text": data.get("text") or "",
                "audio_path": data.get("audio_path") or "",
                "timestamp": metadata.get("timestamp") or "",
                "start_time": metadata.get("start_time", 0),
                "end_time": metadata.get("end_time", 0),
                "speaker_label": data.get("speaker_label") or "UNKNOWN",
                "speaker_name": data.get("speaker_name") or "",
                "language": data.get("language") or "",
            }
        )

    conn.close()
    return results


def get_pipeline_status() -> dict[str, Any]:
    """Check status of running background tasks."""
    try:
        # Check for running Python processes
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )

        processes = []
        for line in result.stdout.split("\n"):
            if "ingest_audio_files.py" in line or "exocortex" in line:
                parts = line.split()
                if len(parts) >= 11:
                    processes.append({
                        "pid": parts[1],
                        "cpu": parts[2],
                        "mem": parts[3],
                        "command": " ".join(parts[10:])[:100]
                    })

        # Check task output files
        tasks_dir = Path("/private/tmp/claude/-Users-yuzucchi-Documents-live-translation-local/tasks")
        active_tasks = []

        if tasks_dir.exists():
            for task_file in sorted(tasks_dir.glob("*.output"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                try:
                    with open(task_file) as f:
                        lines = f.readlines()
                        last_lines = lines[-3:] if len(lines) > 3 else lines

                    active_tasks.append({
                        "task_id": task_file.stem,
                        "last_update": datetime.fromtimestamp(task_file.stat().st_mtime).isoformat(),
                        "recent_output": "".join(last_lines).strip()
                    })
                except:
                    pass

        return {
            "processes": processes,
            "active_tasks": active_tasks,
            "has_activity": len(processes) > 0 or len(active_tasks) > 0
        }
    except Exception as e:
        console.print(f"[red]Error checking pipeline status: {e}[/red]")
        return {"processes": [], "active_tasks": [], "has_activity": False, "error": str(e)}


# HTML template with safe DOM manipulation
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Exocortex Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            background: #2d2d2d;
            padding: 20px;
            border-bottom: 2px solid #404040;
        }

        h1 {
            color: #ffffff;
            font-size: 24px;
            margin-bottom: 8px;
        }

        .subtitle {
            color: #999;
            font-size: 14px;
        }

        .container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        .sidebar {
            width: 300px;
            background: #252525;
            border-right: 2px solid #404040;
            overflow-y: auto;
            padding: 20px;
        }

        .main-content {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }

        .speaker-card {
            background: #2d2d2d;
            border: 2px solid #404040;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .speaker-card:hover {
            border-color: #0066cc;
            background: #333;
        }

        .speaker-card.active {
            border-color: #0066cc;
            background: #1a3a5a;
        }

        .speaker-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .speaker-label {
            font-weight: 600;
            font-size: 16px;
            color: #4da6ff;
        }

        .memory-count {
            background: #404040;
            color: #fff;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }

        .speaker-name {
            color: #66cc66;
            font-size: 14px;
        }

        .memory-item {
            background: #2d2d2d;
            border: 1px solid #404040;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }

        .memory-text {
            color: #e0e0e0;
            margin-bottom: 12px;
            line-height: 1.5;
        }

        .memory-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #999;
            font-size: 13px;
        }

        .audio-controls {
            display: flex;
            gap: 8px;
        }

        button {
            background: #0066cc;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s;
        }

        button:hover {
            background: #0052a3;
        }

        button:disabled {
            background: #404040;
            cursor: not-allowed;
        }

        .enrollment-form {
            background: #2d2d2d;
            border: 2px solid #0066cc;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .form-group {
            margin-bottom: 16px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #ccc;
            font-weight: 500;
        }

        input[type="text"] {
            width: 100%;
            padding: 10px;
            background: #1a1a1a;
            border: 1px solid #404040;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 14px;
        }

        input[type="text"]:focus {
            outline: none;
            border-color: #0066cc;
        }

        .status-message {
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 16px;
            display: none;
        }

        .status-message.success {
            background: #1a4d1a;
            color: #66ff66;
            border: 1px solid #2d7a2d;
        }

        .status-message.error {
            background: #4d1a1a;
            color: #ff6666;
            border: 1px solid #7a2d2d;
        }

        .empty-state {
            text-align: center;
            color: #666;
            padding: 40px;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #999;
        }

        /* Audio Player */
        .audio-player {
            background: #1a1a1a;
            border: 1px solid #404040;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            display: none;
        }

        .audio-player.active {
            display: block;
        }

        .player-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .player-title {
            color: #4da6ff;
            font-weight: 600;
            font-size: 14px;
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .player-close {
            background: #404040;
            color: #ccc;
            padding: 4px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }

        .player-close:hover {
            background: #505050;
        }

        .player-controls {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }

        .play-pause-btn {
            background: #0066cc;
            color: white;
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .play-pause-btn:hover {
            background: #0052a3;
        }

        .time-display {
            color: #999;
            font-size: 13px;
            min-width: 100px;
        }

        .seek-bar {
            flex: 1;
            height: 6px;
            background: #404040;
            border-radius: 3px;
            cursor: pointer;
            position: relative;
        }

        .seek-progress {
            height: 100%;
            background: #0066cc;
            border-radius: 3px;
            width: 0%;
            transition: width 0.1s;
        }

        .volume-control {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .volume-slider {
            width: 80px;
            height: 4px;
            -webkit-appearance: none;
            appearance: none;
            background: #404040;
            border-radius: 2px;
            outline: none;
        }

        .volume-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 14px;
            height: 14px;
            background: #0066cc;
            border-radius: 50%;
            cursor: pointer;
        }

        .volume-slider::-moz-range-thumb {
            width: 14px;
            height: 14px;
            background: #0066cc;
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }

        .waveform-visual {
            height: 40px;
            background: linear-gradient(to right, #1a3a5a 0%, #2d5a8a 50%, #1a3a5a 100%);
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }

        .waveform-visual::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            width: 0%;
            background: rgba(77, 166, 255, 0.3);
            animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 0.6; }
        }

        /* Pagination */
        .pagination {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;
            margin-top: 20px;
            padding: 20px 0;
        }

        .pagination button {
            padding: 8px 16px;
            background: #2d2d2d;
            border: 1px solid #404040;
            color: #ccc;
        }

        .pagination button:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }

        .pagination button:not(:disabled):hover {
            background: #404040;
            border-color: #0066cc;
        }

        .page-info {
            color: #999;
            font-size: 14px;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Exocortex Dashboard</h1>
        <p class="subtitle">Monitor, search, and manage your external memory system</p>
    </div>

    <div class="tabs" id="tabBar" style="display: flex; gap: 8px; background: #1a1a1a; padding: 12px 30px 0 30px; border-bottom: 1px solid #2d2d2d;">
        <!-- Tabs will be created dynamically -->
    </div>

    <div class="container">
        <!-- Dashboard Tab -->
        <div id="dashboardTab" class="tab-content" style="display: none; padding: 30px;">
            <div id="statsContent" class="loading">Loading statistics...</div>
        </div>

        <!-- Search Tab -->
        <div id="searchTab" class="tab-content" style="display: none; padding: 30px;">
            <div style="margin-bottom: 30px;">
                <div style="display: flex; gap: 12px; margin-bottom: 16px;">
                    <input type="text" id="searchInput" placeholder="Search your memories..." style="flex: 1; padding: 14px 20px; background: #1a1a2e; border: 2px solid #2d2d4a; border-radius: 8px; color: #e0e0e0; font-size: 15px;">
                    <button onclick="searchMemories()" style="padding: 14px 24px;">Search</button>
                </div>
                <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                    <select id="speakerFilter" style="padding: 10px 16px; background: #1a1a2e; border: 1px solid #2d2d4a; border-radius: 6px; color: #e0e0e0;">
                        <option value="all">All Speakers</option>
                    </select>
                    <select id="resultLimit" style="padding: 10px 16px; background: #1a1a2e; border: 1px solid #2d2d4a; border-radius: 6px; color: #e0e0e0;">
                        <option value="20">20 results</option>
                        <option value="50" selected>50 results</option>
                        <option value="100">100 results</option>
                    </select>
                </div>
            </div>
            <div id="searchResults"></div>
        </div>

        <!-- Speakers Tab (original content) -->
        <div id="speakersTab" class="tab-content">
            <div class="sidebar">
                <div id="speakerList" class="loading">Loading speakers...</div>
            </div>

            <div class="main-content">
                <div id="statusMessage" class="status-message"></div>
                <div id="mainContent" class="empty-state">
                    Select a speaker from the sidebar to begin enrollment
                </div>
            </div>
        </div>

        <!-- Pipeline Tab -->
        <div id="pipelineTab" class="tab-content" style="display: none; padding: 30px;">
            <div id="pipelineStatus" class="loading">Checking pipeline status...</div>
        </div>
    </div>

    <script>
        let currentSpeaker = null;
        let speakersData = {};
        let currentPage = 0;
        const MEMORIES_PER_PAGE = 10;
        let audioPlayer = null;
        let currentAudioElement = null;
        let isPlaying = false;

        // Fetch speakers on load
        async function loadSpeakers() {
            try {
                const response = await fetch('/api/speakers');
                speakersData = await response.json();
                renderSpeakerList();
            } catch (error) {
                showStatus('Failed to load speakers: ' + error.message, 'error');
            }
        }

        function renderSpeakerList() {
            const container = document.getElementById('speakerList');
            container.innerHTML = '';

            const labels = Object.entries(speakersData)
                .map(([label, memories]) => ({
                    label,
                    count: memories.length,
                    hasName: memories[0]?.speaker_name ? true : false,
                    speakerName: memories[0]?.speaker_name
                }))
                .sort((a, b) => b.count - a.count);

            if (labels.length === 0) {
                const empty = document.createElement('div');
                empty.className = 'empty-state';
                empty.textContent = 'No speakers found';
                container.appendChild(empty);
                return;
            }

            labels.forEach(speaker => {
                const card = document.createElement('div');
                card.className = 'speaker-card';
                if (currentSpeaker === speaker.label) {
                    card.classList.add('active');
                }
                card.onclick = () => selectSpeaker(speaker.label);

                const header = document.createElement('div');
                header.className = 'speaker-header';

                const label = document.createElement('span');
                label.className = 'speaker-label';
                label.textContent = speaker.label;

                const count = document.createElement('span');
                count.className = 'memory-count';
                count.textContent = speaker.count;

                header.appendChild(label);
                header.appendChild(count);
                card.appendChild(header);

                if (speaker.hasName) {
                    const name = document.createElement('div');
                    name.className = 'speaker-name';
                    name.textContent = '→ ' + speaker.speakerName;
                    card.appendChild(name);
                }

                container.appendChild(card);
            });
        }

        async function selectSpeaker(label) {
            currentSpeaker = label;
            currentPage = 0;
            closeAudioPlayer();
            renderSpeakerList();
            renderMemories();
        }

        function renderMemories() {
            const container = document.getElementById('mainContent');
            container.innerHTML = '';

            const memories = speakersData[currentSpeaker] || [];
            const totalPages = Math.ceil(memories.length / MEMORIES_PER_PAGE);
            const start = currentPage * MEMORIES_PER_PAGE;
            const end = start + MEMORIES_PER_PAGE;
            const pageMemories = memories.slice(start, end);

            // Create audio player (hidden by default)
            const player = document.createElement('div');
            player.className = 'audio-player';
            player.id = 'audioPlayer';
            container.appendChild(player);

            // Create enrollment form
            const form = document.createElement('div');
            form.className = 'enrollment-form';

            const formTitle = document.createElement('h2');
            formTitle.textContent = 'Enroll Speaker: ' + currentSpeaker;
            formTitle.style.marginBottom = '16px';
            formTitle.style.color = '#4da6ff';
            form.appendChild(formTitle);

            const formGroup = document.createElement('div');
            formGroup.className = 'form-group';

            const label = document.createElement('label');
            label.textContent = 'Speaker Name';
            formGroup.appendChild(label);

            const input = document.createElement('input');
            input.type = 'text';
            input.id = 'speakerNameInput';
            input.placeholder = 'Enter speaker name (e.g., Alice, Bob)';
            if (memories[0]?.speaker_name) {
                input.value = memories[0].speaker_name;
            }
            formGroup.appendChild(input);

            form.appendChild(formGroup);

            const submitBtn = document.createElement('button');
            submitBtn.textContent = 'Update All Memories (' + memories.length + ')';
            submitBtn.onclick = enrollSpeaker;
            form.appendChild(submitBtn);

            container.appendChild(form);

            // Create memories list header
            const memoriesTitle = document.createElement('h3');
            memoriesTitle.textContent = 'Memories (' + memories.length + ' total)';
            memoriesTitle.style.marginBottom = '16px';
            memoriesTitle.style.color = '#ccc';
            container.appendChild(memoriesTitle);

            // Render current page memories
            pageMemories.forEach(memory => {
                const item = document.createElement('div');
                item.className = 'memory-item';

                const text = document.createElement('div');
                text.className = 'memory-text';
                text.textContent = memory.text;
                item.appendChild(text);

                const meta = document.createElement('div');
                meta.className = 'memory-meta';

                const timestamp = document.createElement('span');
                timestamp.textContent = memory.timestamp || 'No timestamp';
                meta.appendChild(timestamp);

                const controls = document.createElement('div');
                controls.className = 'audio-controls';

                if (memory.audio_path) {
                    const playBtn = document.createElement('button');
                    playBtn.textContent = '▶️ Play Audio';
                    playBtn.onclick = () => playAudio(
                        memory.audio_path,
                        memory.text,
                        memory.start_time,
                        memory.end_time
                    );
                    controls.appendChild(playBtn);
                }

                meta.appendChild(controls);
                item.appendChild(meta);
                container.appendChild(item);
            });

            // Add pagination controls if needed
            if (totalPages > 1) {
                const pagination = document.createElement('div');
                pagination.className = 'pagination';

                const prevBtn = document.createElement('button');
                prevBtn.textContent = '← Previous';
                prevBtn.disabled = currentPage === 0;
                prevBtn.onclick = () => {
                    if (currentPage > 0) {
                        currentPage--;
                        renderMemories();
                    }
                };
                pagination.appendChild(prevBtn);

                const pageInfo = document.createElement('span');
                pageInfo.className = 'page-info';
                pageInfo.textContent = `Page ${currentPage + 1} of ${totalPages}`;
                pagination.appendChild(pageInfo);

                const nextBtn = document.createElement('button');
                nextBtn.textContent = 'Next →';
                nextBtn.disabled = currentPage >= totalPages - 1;
                nextBtn.onclick = () => {
                    if (currentPage < totalPages - 1) {
                        currentPage++;
                        renderMemories();
                    }
                };
                pagination.appendChild(nextBtn);

                container.appendChild(pagination);
            }
        }

        async function enrollSpeaker() {
            const nameInput = document.getElementById('speakerNameInput');
            const speakerName = nameInput.value.trim();

            if (!speakerName) {
                showStatus('Please enter a speaker name', 'error');
                return;
            }

            try {
                const response = await fetch('/api/enroll', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        speaker_label: currentSpeaker,
                        speaker_name: speakerName,
                    }),
                });

                const result = await response.json();

                if (result.success) {
                    showStatus('Successfully updated ' + result.count + ' memories!', 'success');
                    await loadSpeakers();
                    selectSpeaker(currentSpeaker);
                } else {
                    showStatus('Enrollment failed: ' + result.error, 'error');
                }
            } catch (error) {
                showStatus('Enrollment failed: ' + error.message, 'error');
            }
        }

        function playAudio(audioPath, transcription, startTime = 0, endTime = 0) {
            const player = document.getElementById('audioPlayer');
            player.className = 'audio-player active';
            player.innerHTML = '';

            // Stop any currently playing audio
            if (currentAudioElement) {
                currentAudioElement.pause();
                currentAudioElement = null;
            }

            const encodedPath = encodeURIComponent(audioPath);
            const audioUrl = '/api/audio?path=' + encodedPath;
            const fileName = audioPath.split('/').pop();

            // Create audio element
            currentAudioElement = new Audio(audioUrl);
            currentAudioElement.volume = 0.8;

            // Store segment boundaries
            const segmentStart = startTime || 0;
            const segmentEnd = endTime || 0;
            let hasSegmentBounds = segmentStart > 0 || segmentEnd > 0;

            // Player header
            const header = document.createElement('div');
            header.className = 'player-header';

            const title = document.createElement('div');
            title.className = 'player-title';
            title.textContent = '🎵 ' + fileName;
            header.appendChild(title);

            const closeBtn = document.createElement('div');
            closeBtn.className = 'player-close';
            closeBtn.textContent = '✕';
            closeBtn.onclick = closeAudioPlayer;
            header.appendChild(closeBtn);

            player.appendChild(header);

            // Waveform visual
            const waveform = document.createElement('div');
            waveform.className = 'waveform-visual';
            player.appendChild(waveform);

            // Controls container
            const controls = document.createElement('div');
            controls.className = 'player-controls';

            // Play/Pause button
            const playPauseBtn = document.createElement('button');
            playPauseBtn.className = 'play-pause-btn';
            playPauseBtn.innerHTML = '▶';
            playPauseBtn.onclick = togglePlayPause;
            controls.appendChild(playPauseBtn);

            // Time display
            const timeDisplay = document.createElement('span');
            timeDisplay.className = 'time-display';
            timeDisplay.textContent = '0:00 / 0:00';
            controls.appendChild(timeDisplay);

            // Seek bar
            const seekBar = document.createElement('div');
            seekBar.className = 'seek-bar';
            seekBar.onclick = (e) => seek(e, seekBar);

            const seekProgress = document.createElement('div');
            seekProgress.className = 'seek-progress';
            seekProgress.id = 'seekProgress';
            seekBar.appendChild(seekProgress);

            controls.appendChild(seekBar);

            // Volume control
            const volumeControl = document.createElement('div');
            volumeControl.className = 'volume-control';

            const volumeIcon = document.createElement('span');
            volumeIcon.textContent = '🔊';
            volumeControl.appendChild(volumeIcon);

            const volumeSlider = document.createElement('input');
            volumeSlider.type = 'range';
            volumeSlider.className = 'volume-slider';
            volumeSlider.min = '0';
            volumeSlider.max = '100';
            volumeSlider.value = '80';
            volumeSlider.oninput = (e) => {
                currentAudioElement.volume = e.target.value / 100;
            };
            volumeControl.appendChild(volumeSlider);

            controls.appendChild(volumeControl);
            player.appendChild(controls);

            // Segment info and transcription display
            const infoDiv = document.createElement('div');
            infoDiv.style.marginTop = '12px';
            infoDiv.style.paddingTop = '12px';
            infoDiv.style.borderTop = '1px solid #404040';

            if (hasSegmentBounds && segmentEnd > 0) {
                const segmentInfo = document.createElement('div');
                segmentInfo.style.color = '#4da6ff';
                segmentInfo.style.fontSize = '12px';
                segmentInfo.style.marginBottom = '8px';
                segmentInfo.textContent = '🎯 Playing segment: ' + formatTime(segmentStart) + ' → ' + formatTime(segmentEnd);
                infoDiv.appendChild(segmentInfo);

                // Add button to play full audio
                const fullAudioBtn = document.createElement('button');
                fullAudioBtn.textContent = '🎵 Play Full Audio';
                fullAudioBtn.style.padding = '4px 12px';
                fullAudioBtn.style.fontSize = '12px';
                fullAudioBtn.style.marginBottom = '8px';
                fullAudioBtn.onclick = () => {
                    hasSegmentBounds = false;
                    currentAudioElement.currentTime = 0;
                    if (!isPlaying) {
                        currentAudioElement.play();
                        isPlaying = true;
                        playPauseBtn.innerHTML = '⏸';
                    }
                    showStatus('Playing full audio', 'success');
                };
                infoDiv.appendChild(fullAudioBtn);
            }

            if (transcription) {
                const transcriptDiv = document.createElement('div');
                transcriptDiv.style.color = '#999';
                transcriptDiv.style.fontSize = '13px';
                transcriptDiv.style.fontStyle = 'italic';
                transcriptDiv.style.marginTop = hasSegmentBounds ? '8px' : '0';
                transcriptDiv.textContent = '"' + transcription + '"';
                infoDiv.appendChild(transcriptDiv);
            }

            if (infoDiv.children.length > 0) {
                player.appendChild(infoDiv);
            }

            // Audio event listeners
            currentAudioElement.addEventListener('loadedmetadata', () => {
                // Seek to segment start time
                if (hasSegmentBounds && segmentStart > 0) {
                    currentAudioElement.currentTime = segmentStart;
                    showStatus('Playing segment: ' + formatTime(segmentStart) + ' - ' + formatTime(segmentEnd || currentAudioElement.duration), 'success');
                }
                updateTimeDisplay();
            });

            currentAudioElement.addEventListener('timeupdate', () => {
                // Stop at segment end time if specified
                if (hasSegmentBounds && segmentEnd > 0 && currentAudioElement.currentTime >= segmentEnd) {
                    currentAudioElement.pause();
                    isPlaying = false;
                    playPauseBtn.innerHTML = '▶';
                    showStatus('Segment playback finished', 'success');
                    // Reset to start of segment
                    currentAudioElement.currentTime = segmentStart;
                    return;
                }
                updateTimeDisplay();
                updateSeekBar();
            });

            currentAudioElement.addEventListener('ended', () => {
                isPlaying = false;
                playPauseBtn.innerHTML = '▶';
                showStatus('Playback finished', 'success');
            });

            currentAudioElement.addEventListener('error', (e) => {
                showStatus('Error playing audio', 'error');
                console.error('Audio error:', e);
            });

            // Auto-play
            currentAudioElement.play().then(() => {
                isPlaying = true;
                playPauseBtn.innerHTML = '⏸';
                showStatus('Playing: ' + fileName, 'success');
            }).catch(err => {
                showStatus('Playback failed: ' + err.message, 'error');
            });

            // Helper functions
            function togglePlayPause() {
                if (isPlaying) {
                    currentAudioElement.pause();
                    playPauseBtn.innerHTML = '▶';
                    isPlaying = false;
                } else {
                    currentAudioElement.play();
                    playPauseBtn.innerHTML = '⏸';
                    isPlaying = true;
                }
            }

            function seek(e, seekBar) {
                const rect = seekBar.getBoundingClientRect();
                const clickX = e.clientX - rect.left;
                const percentage = clickX / rect.width;

                if (hasSegmentBounds && segmentEnd > 0) {
                    // Seek within segment bounds
                    const segmentDuration = segmentEnd - segmentStart;
                    currentAudioElement.currentTime = segmentStart + (percentage * segmentDuration);
                } else {
                    // Seek in full audio
                    currentAudioElement.currentTime = percentage * currentAudioElement.duration;
                }
            }

            function updateTimeDisplay() {
                if (hasSegmentBounds && segmentEnd > 0) {
                    // Show time within segment
                    const current = formatTime(Math.max(0, currentAudioElement.currentTime - segmentStart));
                    const duration = formatTime(segmentEnd - segmentStart);
                    timeDisplay.textContent = current + ' / ' + duration + ' (segment)';
                } else {
                    // Show full audio time
                    const current = formatTime(currentAudioElement.currentTime);
                    const duration = formatTime(currentAudioElement.duration);
                    timeDisplay.textContent = current + ' / ' + duration;
                }
            }

            function updateSeekBar() {
                let percentage;
                if (hasSegmentBounds && segmentEnd > 0) {
                    // Show progress within segment
                    const segmentDuration = segmentEnd - segmentStart;
                    const segmentProgress = currentAudioElement.currentTime - segmentStart;
                    percentage = (segmentProgress / segmentDuration) * 100;
                } else {
                    // Show progress in full audio
                    percentage = (currentAudioElement.currentTime / currentAudioElement.duration) * 100;
                }
                document.getElementById('seekProgress').style.width = Math.max(0, Math.min(100, percentage)) + '%';
            }

            function formatTime(seconds) {
                if (isNaN(seconds)) return '0:00';
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return mins + ':' + (secs < 10 ? '0' : '') + secs;
            }
        }

        function closeAudioPlayer() {
            if (currentAudioElement) {
                currentAudioElement.pause();
                currentAudioElement = null;
            }
            isPlaying = false;
            const player = document.getElementById('audioPlayer');
            if (player) {
                player.className = 'audio-player';
            }
        }

        function showStatus(message, type) {
            const statusEl = document.getElementById('statusMessage');
            statusEl.textContent = message;
            statusEl.className = 'status-message ' + type;
            statusEl.style.display = 'block';

            setTimeout(() => {
                statusEl.style.display = 'none';
            }, 5000);
        }

        // Tab Management
        const tabs = [
            { id: 'dashboard', label: '📊 Dashboard', content: 'dashboardTab' },
            { id: 'search', label: '🔍 Search', content: 'searchTab' },
            { id: 'speakers', label: '🎙️ Speakers', content: 'speakersTab' },
            { id: 'pipeline', label: '⚙️ Pipeline', content: 'pipelineTab' }
        ];

        let currentTab = 'speakers';
        let refreshInterval = null;

        function initTabs() {
            const tabBar = document.getElementById('tabBar');
            tabs.forEach(tab => {
                const button = document.createElement('button');
                button.className = 'tab';
                button.textContent = tab.label;
                button.onclick = () => switchTab(tab.id);
                button.style.cssText = 'padding: 12px 24px; background: transparent; border: none; color: #999; cursor: pointer; font-size: 14px; font-weight: 500; border-radius: 8px 8px 0 0;';
                tabBar.appendChild(button);
            });
        }

        function switchTab(tabId) {
            // Stop any refresh intervals
            if (refreshInterval) {
                clearInterval(refreshInterval);
                refreshInterval = null;
            }

            currentTab = tabId;

            // Update tab buttons
            document.querySelectorAll('.tab').forEach((btn, index) => {
                if (tabs[index].id === tabId) {
                    btn.style.background = '#2d2d4a';
                    btn.style.color = '#4da6ff';
                } else {
                    btn.style.background = 'transparent';
                    btn.style.color = '#999';
                }
            });

            // Show/hide tab content
            tabs.forEach(tab => {
                const content = document.getElementById(tab.content);
                content.style.display = tab.id === tabId ? (tab.id === 'speakers' ? 'flex' : 'block') : 'none';
            });

            // Load data for tab
            if (tabId === 'dashboard') {
                loadDashboard();
            } else if (tabId === 'search') {
                loadSearchFilters();
            } else if (tabId === 'speakers') {
                loadSpeakers();
            } else if (tabId === 'pipeline') {
                loadPipelineStatus();
                refreshInterval = setInterval(loadPipelineStatus, 5000);
            }
        }

        // Dashboard functions
        async function loadDashboard() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                renderDashboard(stats);
            } catch (error) {
                console.error('Failed to load stats:', error);
            }
        }

        function renderDashboard(stats) {
            const container = document.getElementById('statsContent');
            container.className = '';
            container.innerHTML = '';

            // Stats grid
            const grid = document.createElement('div');
            grid.style.cssText = 'display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;';

            const cards = [
                { label: 'Total Memories', value: stats.total_memories },
                { label: 'With Speakers', value: stats.with_speakers },
                { label: 'Audio Files', value: stats.unique_audio_files },
                { label: 'Storage', value: stats.storage.total_mb + ' MB' }
            ];

            cards.forEach(card => {
                const div = document.createElement('div');
                div.style.cssText = 'background: linear-gradient(135deg, #1a1a2e 0%, #16162a 100%); border: 1px solid #2d2d4a; border-radius: 12px; padding: 20px;';

                const value = document.createElement('div');
                value.textContent = card.value;
                value.style.cssText = 'font-size: 36px; font-weight: 700; color: #4da6ff; margin-bottom: 8px;';

                const label = document.createElement('div');
                label.textContent = card.label;
                label.style.cssText = 'color: #999; font-size: 13px; text-transform: uppercase;';

                div.appendChild(value);
                div.appendChild(label);
                grid.appendChild(div);
            });

            container.appendChild(grid);
        }

        // Search functions
        async function searchMemories() {
            const query = document.getElementById('searchInput').value;
            const speaker = document.getElementById('speakerFilter').value;
            const limit = document.getElementById('resultLimit').value;

            if (!query) return;

            try {
                const response = await fetch('/api/search?q=' + encodeURIComponent(query) + '&speaker=' + speaker + '&limit=' + limit);
                const results = await response.json();
                renderSearchResults(results);
            } catch (error) {
                console.error('Search failed:', error);
            }
        }

        function renderSearchResults(results) {
            const container = document.getElementById('searchResults');
            container.innerHTML = '';

            if (results.length === 0) {
                container.textContent = 'No results found';
                return;
            }

            results.forEach(result => {
                const item = document.createElement('div');
                item.style.cssText = 'background: #1a1a2e; border: 1px solid #2d2d4a; border-radius: 8px; padding: 18px; margin-bottom: 12px;';

                const text = document.createElement('div');
                text.textContent = result.text;
                text.style.cssText = 'color: #e0e0e0; margin-bottom: 12px;';

                const meta = document.createElement('div');
                meta.style.cssText = 'display: flex; justify-content: space-between; color: #999; font-size: 13px;';

                const tags = document.createElement('div');
                tags.textContent = result.speaker_name || result.speaker_label;

                meta.appendChild(tags);

                if (result.audio_path) {
                    const btn = document.createElement('button');
                    btn.textContent = '▶️ Play';
                    btn.onclick = () => playAudio(result.audio_path, result.text, result.start_time, result.end_time);
                    meta.appendChild(btn);
                }

                item.appendChild(text);
                item.appendChild(meta);
                container.appendChild(item);
            });
        }

        async function loadSearchFilters() {
            try {
                const response = await fetch('/api/speakers');
                const speakers = await response.json();

                const select = document.getElementById('speakerFilter');
                select.innerHTML = '<option value="all">All Speakers</option>';

                Object.keys(speakers).forEach(label => {
                    const option = document.createElement('option');
                    option.value = label;
                    option.textContent = label;
                    select.appendChild(option);
                });
            } catch (error) {
                console.error('Failed to load filters:', error);
            }
        }

        // Pipeline functions
        async function loadPipelineStatus() {
            try {
                const response = await fetch('/api/pipeline');
                const status = await response.json();
                renderPipelineStatus(status);
            } catch (error) {
                console.error('Failed to load pipeline status:', error);
            }
        }

        function renderPipelineStatus(status) {
            const container = document.getElementById('pipelineStatus');
            container.className = '';
            container.innerHTML = '';

            const header = document.createElement('h2');
            header.textContent = 'Pipeline Status: ' + (status.has_activity ? '🟢 Active' : '⚪ Idle');
            header.style.cssText = 'color: #ccc; margin-bottom: 20px;';
            container.appendChild(header);

            if (status.processes && status.processes.length > 0) {
                const title = document.createElement('h3');
                title.textContent = 'Running Processes';
                title.style.cssText = 'color: #999; margin-bottom: 12px;';
                container.appendChild(title);

                status.processes.forEach(proc => {
                    const item = document.createElement('div');
                    item.textContent = 'PID ' + proc.pid + ' | CPU ' + proc.cpu + '% | MEM ' + proc.mem + '% | ' + proc.command;
                    item.style.cssText = 'background: #1a1a2e; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 12px; margin-bottom: 8px;';
                    container.appendChild(item);
                });
            }

            if (status.active_tasks && status.active_tasks.length > 0) {
                const title = document.createElement('h3');
                title.textContent = 'Recent Tasks';
                title.style.cssText = 'color: #999; margin: 20px 0 12px 0;';
                container.appendChild(title);

                status.active_tasks.forEach(task => {
                    const item = document.createElement('div');
                    item.style.cssText = 'background: #1a1a2e; padding: 12px; border-radius: 6px; margin-bottom: 8px;';

                    const taskId = document.createElement('div');
                    taskId.textContent = 'Task: ' + task.task_id;
                    taskId.style.cssText = 'font-weight: 600; margin-bottom: 4px;';

                    const time = document.createElement('div');
                    time.textContent = 'Last update: ' + new Date(task.last_update).toLocaleString();
                    time.style.cssText = 'font-size: 12px; color: #999;';

                    item.appendChild(taskId);
                    item.appendChild(time);
                    container.appendChild(item);
                });
            }

            if (!status.has_activity) {
                const msg = document.createElement('div');
                msg.textContent = 'No active pipeline processes';
                msg.style.cssText = 'text-align: center; color: #666; padding: 40px;';
                container.appendChild(msg);
            }
        }

        // Initialize on load
        initTabs();
        switchTab('speakers');
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Serve the main UI."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/speakers")
def api_speakers():
    """Get all speakers and their memories."""
    try:
        console.print("[dim]Fetching speakers from database...[/dim]")
        speakers = get_memories_by_speaker()
        console.print(f"[green]Found {len(speakers)} speaker labels[/green]")
        return jsonify(speakers)
    except Exception as e:
        console.print(f"[red]Error fetching speakers: {e}[/red]")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    """Enroll a speaker by updating all memories with the speaker name."""
    try:
        data = request.json
        speaker_label = data.get("speaker_label")
        speaker_name = data.get("speaker_name")

        if not speaker_label or not speaker_name:
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        count = update_speaker_name(speaker_label, speaker_name)

        return jsonify({"success": True, "count": count})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/audio")
def api_audio():
    """Serve audio file for playback."""
    try:
        audio_path = request.args.get("path")
        if not audio_path:
            return jsonify({"error": "No audio path provided"}), 400

        audio_file = Path(audio_path)
        if not audio_file.exists():
            return jsonify({"error": f"Audio file not found: {audio_path}"}), 404

        # Determine mimetype based on file extension
        ext = audio_file.suffix.lower()
        mimetype_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
        }
        mimetype = mimetype_map.get(ext, "audio/wav")

        return send_file(str(audio_file), mimetype=mimetype)
    except Exception as e:
        console.print(f"[red]Error serving audio: {e}[/red]")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """Get system statistics."""
    try:
        console.print("[dim]Fetching system statistics...[/dim]")
        stats = get_system_stats()
        return jsonify(stats)
    except Exception as e:
        console.print(f"[red]Error getting stats: {e}[/red]")
        return jsonify({"error": str(e)}), 500


@app.route("/api/search")
def api_search():
    """Search memories."""
    try:
        query = request.args.get("q", "")
        speaker = request.args.get("speaker", "all")
        limit = int(request.args.get("limit", 50))

        if speaker == "all":
            speaker = None

        console.print(f"[dim]Searching for: {query}[/dim]")
        results = search_memories(query, limit, speaker)
        console.print(f"[green]Found {len(results)} results[/green]")
        return jsonify(results)
    except Exception as e:
        console.print(f"[red]Error searching: {e}[/red]")
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline")
def api_pipeline():
    """Get pipeline status."""
    try:
        status = get_pipeline_status()
        return jsonify(status)
    except Exception as e:
        console.print(f"[red]Error getting pipeline status: {e}[/red]")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    console.print("\n[bold cyan]🧠 Exocortex Dashboard[/bold cyan]\n")

    # Check database exists
    db_path = EXOCORTEX_PATH / "metadata.db"
    if not db_path.exists():
        console.print(f"[red]Error: Database not found at {db_path}[/red]")
        console.print("[yellow]Have you run exocortex ingestion yet?[/yellow]")
        exit(1)

    console.print(f"[green]✓ Database found: {db_path}[/green]")

    # Test database connection
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        count = cursor.fetchone()[0]
        conn.close()
        console.print(f"[green]✓ Database accessible: {count} memories found[/green]")
    except Exception as e:
        console.print(f"[red]Error accessing database: {e}[/red]")
        exit(1)

    console.print("\n[bold]Open in browser:[/bold] http://localhost:5001\n")
    console.print("[green]Starting server...[/green]")

    app.run(host="0.0.0.0", port=5001, debug=True)
