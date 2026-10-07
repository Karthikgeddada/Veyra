let currentRunId = null;
let pollInterval = null;
let allGeneratedFiles = [];

document.getElementById('generateBtn').addEventListener('click', async () => {
    const prompt = document.getElementById('promptInput').value.trim();
    if (!prompt) return alert('Please enter a prompt');

    // Reset UI
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('downloadBtn').style.display = 'none';
    document.querySelectorAll('.step').forEach(el => {
        el.className = 'step';
    });
    document.getElementById('progressBarFill').style.width = '0%';
    document.getElementById('progressText').innerText = '0% Complete';
    document.getElementById('logsTerminal').innerHTML = 'Initializing agents...<br>';
    document.getElementById('generateBtn').disabled = true;
    document.getElementById('generateBtn').innerHTML = 'Generating... <span class="sparkle">⚙️</span>';

    try {
        const res = await fetch('/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({prompt})
        });
        const data = await res.json();
        currentRunId = data.run_id;
        
        pollInterval = setInterval(pollStatus, 2000);
    } catch (err) {
        alert('Error starting generation');
        document.getElementById('generateBtn').disabled = false;
        document.getElementById('generateBtn').innerHTML = 'Generate Project <span class="sparkle">✨</span>';
    }
});

async function pollStatus() {
    if (!currentRunId) return;

    try {
        const res = await fetch(`/status/${currentRunId}`);
        if (!res.ok) return;
        const data = await res.json();

        updateLogs(data.logs);
        updateTracker(data.status);

        if (data.status === 'completed' || data.status === 'failed') {
            clearInterval(pollInterval);
            document.getElementById('generateBtn').disabled = false;
            document.getElementById('generateBtn').innerHTML = 'Generate Project <span class="sparkle">✨</span>';
            
            if (data.status === 'completed') {
                fetchFiles();
            }
        }
    } catch (e) {
        console.error(e);
    }
}

function updateLogs(logs) {
    const term = document.getElementById('logsTerminal');
    if(logs && logs.length > 0){
        term.innerHTML = logs.map(l => `> ${l}`).join('<br>');
        term.scrollTop = term.scrollHeight;
    }
}

function updateTracker(status) {
    const statusMap = {
        'starting': {id: 'step-requirements', pct: 5},
        'analyzing_requirements': {id: 'step-requirements', pct: 15},
        'designing_architecture': {id: 'step-architecture', pct: 28},
        'generating_code': {id: 'step-coding', pct: 45},
        'validating': {id: 'step-coding', pct: 55},
        'testing': {id: 'step-testing', pct: 62},
        'running_tests': {id: 'step-testing', pct: 72},
        'debugging': {id: 'step-review', pct: 80},
        'reviewing': {id: 'step-review', pct: 88},
        'documenting': {id: 'step-docs', pct: 95},
        'completed': {id: 'step-packaging', pct: 100}
    };

    const stepData = statusMap[status];
    if (!stepData) return;

    const activeId = stepData.id;

    // Update Progress Bar
    document.getElementById('progressBarFill').style.width = `${stepData.pct}%`;
    document.getElementById('progressText').innerText = `${stepData.pct}% Complete`;

    let foundActive = false;
    document.querySelectorAll('.step').forEach(el => {
        if (el.id === activeId) {
            el.className = 'step active';
            foundActive = true;
        } else if (!foundActive) {
            el.className = 'step done';
        } else {
            el.className = 'step';
        }
    });

    if (status === 'completed') {
        document.querySelectorAll('.step').forEach(el => el.className = 'step done');
    }
}

async function fetchFiles() {
    try {
        const res = await fetch(`/files/${currentRunId}`);
        if (!res.ok) return;
        const data = await res.json();
        
        allGeneratedFiles = data.files;
        document.getElementById('resultsSection').classList.remove('hidden');
        document.getElementById('downloadBtn').href = `/download/${currentRunId}`;
        document.getElementById('downloadBtn').style.display = 'inline-block';

        renderFileList();
    } catch (e) {
        console.error(e);
    }
}

function renderFileList() {
    const list = document.getElementById('fileList');
    list.innerHTML = '';
    allGeneratedFiles.forEach((f, idx) => {
        const li = document.createElement('li');
        li.textContent = f.path;
        li.onclick = () => previewCode(idx);
        list.appendChild(li);
    });
    if (allGeneratedFiles.length > 0) {
        previewCode(0);
    }
}

function previewCode(index) {
    const file = allGeneratedFiles[index];
    const viewer = document.getElementById('codeViewer');
    
    // Auto detect language based on extension
    let lang = 'python';
    if(file.path.endsWith('.js')) lang = 'javascript';
    else if(file.path.endsWith('.html')) lang = 'html';
    else if(file.path.endsWith('.css')) lang = 'css';
    else if(file.path.endsWith('.md')) lang = 'markdown';
    else if(file.path.endsWith('.json')) lang = 'json';
    
    viewer.className = `language-${lang}`;
    viewer.textContent = file.content;
    
    if (window.hljs) {
        hljs.highlightElement(viewer);
    }
}
