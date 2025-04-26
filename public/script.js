// public/script.js

// --- Global Variables ---
let cy; // Cytoscape instance
let currentConfig = null; // Represents the currently loaded/active config object { id, name, graph_structure }
let defaultGraphStructure = null; // To store the structure fetched from backend
let sessionLog = []; // Logs for the current browser session
let edgeHandlesInstance = null; // Reference to edge handles extension API
let contextMenuInstance = null; // Reference to context menus extension API
let newNodeCounter = 0; // Counter for generating unique default node IDs

// --- DEFINE HELPER FUNCTIONS FIRST ---
function nodeLabelFunc(node) {
    const id = node.data('id');
    const fullName = node.data('fullName') || id;
    const currentLabel = node.data('currentProbLabel');
    return `${id}: ${fullName}\n${currentLabel || '(N/A)'}`;
}

// Client-side cycle check helper
function simulateIsDag(graph) {
    const adj = {}; const nodesSet = new Set();
    graph.nodes.forEach(n => { adj[n.id] = []; nodesSet.add(n.id); });
    graph.edges.forEach(e => { if(e.source in adj && nodesSet.has(e.target)) adj[e.source].push(e.target); });
    const path = new Set(); const visited = new Set();
    function dfs(node) {
        path.add(node); visited.add(node);
        for (const neighbor of adj[node] || []) {
            if (!nodesSet.has(neighbor)) continue;
            if (path.has(neighbor)) return false; // Cycle
            if (!visited.has(neighbor)) { if (!dfs(neighbor)) return false; }
        }
        path.delete(node); return true;
    }
    for (const node of nodesSet) { if (!visited.has(node)) { if (!dfs(node)) return false; } }
    return true;
}

// --- Utility Functions ---
function selectConfigInDropdown(configId) { const select = document.getElementById('load-config-select'); select.value = configId; }
function enableUI(enable) { const buttons = document.querySelectorAll('button'); const inputs = document.querySelectorAll('input, select'); buttons.forEach(btn => btn.disabled = !enable); inputs.forEach(inp => inp.disabled = !enable); document.getElementById('gradient-toggle').disabled = false; document.body.style.cursor = enable ? 'default' : 'wait'; }
function showSpinner(show) { document.getElementById('loading-spinner').classList.toggle('hidden', !show); }
function runLayout() { if (!cy) return; let l = 'cola'; try { cy.layout({ name: 'cola', animate:true, nodeSpacing: 50, edgeLength: 180, padding: 30 }).run(); } catch (e) { console.warn('Cola failed, using dagre'); l = 'dagre'; try { cy.layout({ name: 'dagre', rankDir:'TB', spacingFactor: 1.2 }).run(); } catch (e) { console.error('Layouts failed'); l = 'grid'; try { cy.layout({ name: 'grid' }).run(); } catch(e) { console.error("Grid layout failed too!")}}} console.log('Using layout:', l); }
function updateInputControls(nodes) { const c = document.getElementById('input-controls-container'); c.innerHTML = ''; const iN = nodes.filter(n => n.nodeType === 'input'); if (iN.length === 0) { c.innerHTML = '<p>No input nodes defined.</p>'; return; } iN.forEach(n => { const d = document.createElement('div'); const l = document.createElement('label'); l.htmlFor = `input-${n.id}`; l.textContent = `${n.id} (${n.fullName || n.id}):`; const i = document.createElement('input'); i.type = 'number'; i.id = `input-${n.id}`; i.name = n.id; i.min = "0"; i.max = "1"; i.step = "0.01"; i.value = "0.5"; d.appendChild(l); d.appendChild(i); c.appendChild(d); }); }
async function fetchAndUpdateLLM() { /* ... (Prediction logic - unchanged from previous good version) ... */ if (!currentConfig || !currentConfig.graph_structure || !currentConfig.graph_structure.nodes.length === 0) { alert("Load config first."); return; } if (!cy) { alert("Graph not ready."); return; } setStatusMessage('predict-status', "Gathering inputs...", "loading"); let inputs = { input_values: {} }; let invalid = false; currentConfig.graph_structure.nodes.filter(n => n.nodeType === 'input').forEach(n => { const el = document.getElementById(`input-${n.id}`); const cont = el?.parentElement; let v = 0.5; if (el) { v = parseFloat(el.value); if (isNaN(v) || v < 0 || v > 1) { setStatusMessage('predict-status', `Invalid input for ${n.id}.`, "error"); cont?.classList.add('invalid-input'); invalid = true; } else { cont?.classList.remove('invalid-input'); } } inputs.input_values[n.id] = isNaN(v)?0.5:Math.max(0, Math.min(1, v)); }); if (invalid) { setStatusMessage('predict-status', "Fix invalid inputs (0-1).", "error"); return; } const payload = { ...inputs, graph_structure: currentConfig.graph_structure, config_id: currentConfig.id, config_name: currentConfig.name }; setStatusMessage('predict-status', "Running prediction...", "loading"); showSpinner(true); enableUI(false); clearLLMOutputs(); await retryFetch(async () => { const r = await fetch('/api/predict_openai_bn_single_call', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }); const d = await r.json(); if (!r.ok) throw new Error(d.detail || `HTTP error ${r.status}`); updateNodeProbabilities(d.probabilities); displayLLMReasoning(d.llm_reasoning); displayLLMContext(d.llm_context); setStatusMessage('predict-status', "Prediction complete.", "success"); logPrediction(inputs.input_values, d.probabilities); }, 3, () => setStatusMessage('predict-status', "Prediction failed. Retrying...", "error")).catch(e => { console.error("Predict error:", e); setStatusMessage('predict-status', `Prediction failed: ${e.message}`, "error"); clearLLMOutputs(); }).finally(() => { enableUI(true); showSpinner(false); }); }
function updateNodeProbabilities(probabilities) { /* ... (Node styling logic - unchanged) ... */ if (!cy) return; const useGradient = document.getElementById('gradient-toggle').checked; cy.nodes().forEach(node => { const nodeId = node.id(); const nodeType = node.data('nodeType'); let probState1 = null; if (probabilities && probabilities[nodeId] && probabilities[nodeId]["1"] !== undefined) { probState1 = probabilities[nodeId]["1"]; node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`); } else if (nodeType === 'input') { const inputElement = document.getElementById(`input-${nodeId}`); const currentVal = inputElement ? (parseFloat(inputElement.value) || 0.5) : 0.5; probState1 = currentVal; node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`); } else { node.data('currentProbLabel', '(N/A)'); } let baseBgColor = nodeType === 'input' ? '#add8e6' : '#f0e68c'; let textColor = '#333'; let finalBgColor = baseBgColor; if (probState1 !== null) { if (useGradient) { finalBgColor = `rgb(${Math.round(255 * (1 - probState1))}, ${Math.round(255 * probState1)}, 0)`; textColor = '#333'; } else { finalBgColor = '#4B0082'; textColor = '#FFFFFF'; } } else if (!useGradient && nodeType !== 'input') { finalBgColor = '#4B0082'; textColor = '#FFFFFF'; } node.style({ 'background-color': finalBgColor, 'color': textColor }); }); cy.style().update(); }
function displayLLMReasoning(reasoningText) { /* ... (Display reasoning - unchanged) ... */ const d = document.getElementById('llm-reasoning-content'); d.textContent = reasoningText || "N/A"; }
function displayLLMContext(context) { /* ... (Display context - unchanged) ... */ if (!context) return; const iD = document.getElementById('input-context'); const sD = document.getElementById('structure-context'); const dD = document.getElementById('node-descriptions-context'); let iH='<ul>';(context.input_states||[]).forEach(s=>{iH+=`<li>${s.node}(${s.description}): ${s.state} (p=${s.value.toFixed(2)})</li>`;});iH+='</ul>';iD.innerHTML=iH||'N/A'; let sH='<ul>';Object.entries(context.node_dependencies||{}).forEach(([n,p])=>{sH+=`<li>${n}: ${p.join(',')||'None'}</li>`;});sH+='</ul>';sD.innerHTML=sH||'N/A'; let dH='<ul>';Object.entries(context.node_descriptions||{}).forEach(([n,d])=>{dH+=`<li>${n}: ${d}</li>`;});dH+='</ul>';dD.innerHTML=dH||'N/A';}
function setStatusMessage(elementId, message, type) { /* ... (Set status text/class - unchanged) ... */ const el = document.getElementById(elementId); if (el) { el.textContent = message; el.className = `status-message ${type||''}`; } }
async function retryFetch(fetchFn, maxRetries, onRetry) { /* ... (Retry logic - unchanged) ... */ let lastError; for (let attempt = 1; attempt <= maxRetries; attempt++) { try { await fetchFn(); return; } catch (error) { lastError = error; console.warn(`Attempt ${attempt} failed: ${error.message}`); if (attempt < maxRetries) { if(onRetry) onRetry(); await new Promise(resolve => setTimeout(resolve, 1000 * attempt)); } } } throw lastError; }
function logPrediction(inputs, probabilities) { /* ... (Session logging - unchanged) ... */ const logEntry = { timestamp: new Date().toISOString(), configId: currentConfig ? currentConfig.id : "unknown", configName: currentConfig ? currentConfig.name : "Unknown", inputs: { ...inputs }, probabilities: {} }; for (const n in probabilities) { logEntry.probabilities[n] = probabilities[n]["1"]; } for (const i in inputs) { if (!(i in logEntry.probabilities)) { logEntry.probabilities[i] = inputs[i]; } } sessionLog.push(logEntry); document.getElementById('log-count').textContent = `Session logs: ${sessionLog.length}`; }
function clearSessionLog() { /* ... (Clear log array/UI - unchanged) ... */ sessionLog = []; document.getElementById('log-count').textContent = `Session logs: 0`; }
function clearLLMOutputs() { /* ... (Clear reasoning/context/status - unchanged) ... */ document.getElementById('llm-reasoning-content').textContent = 'Run prediction...'; document.getElementById('input-context').innerHTML = '<p>N/A</p>'; document.getElementById('structure-context').innerHTML = '<p>N/A</p>'; document.getElementById('node-descriptions-context').innerHTML = '<p>N/A</p>'; setStatusMessage('predict-status', "", ""); }
function downloadSessionLog() { /* ... (Download session CSV - unchanged) ... */ if (sessionLog.length === 0) { alert("No logs."); return; } const h=['Timestamp','ConfigID','ConfigName','NodeID','ProbP1']; const r=[]; sessionLog.forEach(l=>{ const probs = l.probabilities || {}; Object.entries(probs).forEach(([n,p])=>{ if(typeof p === 'number') r.push([l.timestamp,l.configId,l.configName,n,p.toFixed(4)]); else console.warn("Skipping non-numeric probability in session log for node:", n); }); }); const csv=Papa.unparse({fields:h,data:r}); triggerCsvDownload(csv, `session_log_${(currentConfig?.name||'unsaved').replace(/[^a-z0-9]/gi,'_')}`); }
async function downloadAllLogs() { /* ... (Download all logs CSV - unchanged) ... */ if (!currentConfig || !currentConfig.id || currentConfig.id === "unknown" || currentConfig.id === "default-config-001") { alert("Load saved config."); return; } setStatusMessage('predict-status', "Downloading logs...", "loading"); showSpinner(true); enableUI(false); await retryFetch(async () => { const r = await fetch(`/api/download_log/${currentConfig.id}`); if (!r.ok) { if (r.status === 404) throw new Error(`No logs found for '${currentConfig.name}'.`); const e = await r.json().catch(()=>({detail:`HTTP ${r.status}`})); throw new Error(e.detail); } const b = await r.blob(); triggerCsvDownload(b, `all_logs_${currentConfig.name.replace(/[^a-z0-9]/gi,'_')}`); setStatusMessage('predict-status', "Logs downloaded.", "success"); }, 3, () => setStatusMessage('predict-status', "Download failed. Retrying...", "error")).catch(e=>{setStatusMessage('predict-status',`Log download failed: ${e.message}`,"error");}).finally(() => { enableUI(true); showSpinner(false); }); }
function triggerCsvDownload(csvDataOrBlob, baseFilename) { /* ... (Trigger browser download - unchanged) ... */ const blob = (csvDataOrBlob instanceof Blob) ? csvDataOrBlob : new Blob([csvDataOrBlob], { type: 'text/csv;charset=utf-8;' }); const link = document.createElement("a"); const url = URL.createObjectURL(blob); link.setAttribute("href", url); const timestampStr = new Date().toISOString().replace(/[:.]/g, '-'); link.setAttribute("download", `${baseFilename}_${timestampStr}.csv`); link.style.visibility = 'hidden'; document.body.appendChild(link); link.click(); document.body.removeChild(link); URL.revokeObjectURL(url); }

function markConfigUnsaved() {
    const configNameDisplay = document.getElementById('current-config-name');
    if (configNameDisplay && !configNameDisplay.textContent.endsWith('*')) {
        configNameDisplay.textContent += '*';
        setStatusMessage('config-status', "Graph modified. Save changes.", "loading");
    }
}

function clearUnsavedMark() {
     const configNameDisplay = document.getElementById('current-config-name');
     if (configNameDisplay && configNameDisplay.textContent.endsWith('*')) {
         configNameDisplay.textContent = configNameDisplay.textContent.slice(0, -1);
     }
}

// --- Initialization Sequence ---
document.addEventListener('DOMContentLoaded', async () => {
    console.log("DOM Content Loaded. Starting initialization...");

    // Check core library and container first
    if (typeof cytoscape !== 'function') {
        alert("Error: Cytoscape library failed to load.");
        setStatusMessage('config-status', "Error: Core graph library failed.", "error");
        showSpinner(false); return;
    }
    if (!document.getElementById('cy')) {
        alert("Error: Graph container element 'cy' not found.");
        setStatusMessage('config-status', "Error: Graph container missing.", "error");
        showSpinner(false); return;
    }

    // 1. Initialize Cytoscape Instance
    initializeCytoscape(); // Creates 'cy' instance

    if (!cy) {
        // Error handling already inside initializeCytoscape
        showSpinner(false); return;
    }

    // 2. Initialize Editing Extensions (includes registration check)
    initializeEditingExtensions(); // Tries to register and init extensions on 'cy'

    // 3. Initialize UI Button Listeners
    initializeUI();

    // 4. Load Data (Default & Saved)
    showSpinner(true);
    setStatusMessage('config-status', "Loading config data...", "loading");
    try {
        await loadDefaultConfig(); // Fetch default structure
        if (defaultGraphStructure) {
            loadGraphData(defaultGraphStructure, true); // Load default into cy
            addDefaultToDropdown();
            // Select default only if no other config ID is selected/remembered
            const select = document.getElementById('load-config-select');
            if(!select.value || select.value === "") {
                selectConfigInDropdown(defaultGraphStructure.id);
            }
        } else { /* Handle error */ throw new Error("Default config data unavailable."); }

        await loadConfigList(); // Fetch saved configs
        const finalStatus = currentConfig ? `Config '${currentConfig.name}' loaded.` : "Ready.";
        setStatusMessage('config-status', finalStatus, "success");

    } catch (error) {
        console.error("Initialization Data Load Error:", error);
        setStatusMessage('config-status', `Initialization failed: ${error.message}`, "error");
        // Ensure UI is somewhat usable even if data load fails
         if (!currentConfig) { // If no config loaded at all
             updateInputControls([]);
             document.getElementById('current-config-name').textContent = "None";
         }
    } finally {
        showSpinner(false);
    }
});

// --- Cytoscape Core Initialization ---
function initializeCytoscape() {
    // ... (Uses nodeLabelFunc defined above) ...
     if (cy) { console.warn("Cytoscape already initialized."); return; }
    console.log("Attempting to initialize Cytoscape core...");
    try {
        cy = cytoscape({
            container: document.getElementById('cy'),
            elements: [],
            style: [ // Now nodeLabelFunc is defined
                { selector: 'node', style: { 'background-color': '#ccc', 'label': nodeLabelFunc, 'width': 120, 'height': 120, 'shape': 'ellipse', 'text-valign': 'center', 'text-halign': 'center', 'font-size': '10px', 'font-weight': '100', 'text-wrap': 'wrap', 'text-max-width': 110, 'text-outline-color': '#fff', 'text-outline-width': 1, 'color': '#333', 'transition-property': 'background-color, color', 'transition-duration': '0.5s' } },
                { selector: 'node[nodeType="input"]', style: { 'shape': 'rectangle', 'width': 130, 'height': 70 } },
                { selector: 'node[nodeType="hidden"]', style: { 'shape': 'ellipse' } },
                { selector: 'edge', style: { 'width': 2, 'line-color': '#666', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#666', 'curve-style': 'bezier' } },
                { selector: '.edge-source-active', style: { 'border-width': 3, 'border-color': '#00ff00' } },
                { selector: '.eh-grabbed', style: { 'border-width': 3, 'border-color': '#007bff' } }
            ],
            layout: { name: 'preset' }
        });
        console.log("Cytoscape core initialized successfully.");
    } catch(error) {
        console.error("Failed to initialize Cytoscape Core:", error);
        alert(`Critical Error: Failed to initialize the graph visualization library.\n\nError: ${error.message}\n\nPlease check the browser console for details and refresh the page.`);
        cy = null;
    }
}

// --- Editing Extensions Initialization & REGISTRATION ---
function initializeEditingExtensions() {
    console.log("Initializing and Registering editing extensions...");
    if (!cy) { console.error("Cannot init extensions: Cytoscape not ready."); return; }

    let edgeHandlesRegistered = false;
    let contextMenusRegistered = false;

    // --- Register & Initialize Edge Handles ---
    try {
        // Check if library is loaded AND core cytoscape is available
        if (typeof cytoscapeEdgehandles === 'function' && typeof cytoscape === 'function') {
             cytoscape.use( cytoscapeEdgehandles ); // REGISTER the extension
             console.log("Edgehandles registered.");
            edgeHandlesInstance = cy.edgehandles({ /* ... edgehandles options ... */
                 snap: true, handleNodes: 'node', handleSize: 10, handleColor: '#007bff', handlePosition: 'middle top', preview: true, hoverDelay: 150,
                 edgeType: (src, tgt) => 'flat', loopAllowed: (n) => false, nodeLoopOffset: -50,
                 nodeParams: (src, tgt) => ({}), edgeParams: (src, tgt, i) => ({ data: { source: src.id(), target: tgt.id() } }), ghostEdgeParams: () => ({}),
                 handleClass: 'eh-handle', hoverClass: 'eh-hover', snapClass: 'eh-snap', sourceNodeClass: 'eh-source', targetNodeClass: 'eh-target', ghostEdgeClass: 'eh-ghost-edge', previewClass: 'eh-preview', grabbedNodeClass: 'eh-grabbed',
                 complete: ( src, tgt, added ) => { console.log("Edge added via handles:", added); const g = { nodes: cy.nodes().map(n=>n.data()), edges: cy.edges().map(e=>({source:e.source().id(), target:e.target().id()})) }; if (!simulateIsDag(g)) { alert(`Cycle detected! Removing edge ${src.id()}->${tgt.id()}`); cy.remove(added); setStatusMessage('config-status', `Edge aborted (cycle).`, "error"); } else { markConfigUnsaved(); setStatusMessage('config-status', `Edge ${src.id()}->${tgt.id()} added.`, "success"); /* runLayout(); optional */ } }
             });
            console.log("Edgehandles initialized instance.");
            edgeHandlesRegistered = true; // Mark as successful
        } else { console.warn("cytoscape-edgehandles lib not found for registration."); }
    } catch (e) { console.error("Error initializing/registering Edgehandles:", e); }

    // --- Register & Initialize Context Menus ---
    try {
        // Check if library AND dependencies are loaded
        if (typeof cytoscapeContextMenus === 'function' && typeof tippy === 'function' && typeof Popper === 'object' && typeof cytoscape === 'function') {
             cytoscape.use( cytoscapeContextMenus ); // REGISTER the extension
             console.log("Context Menus registered.");
            contextMenuInstance = cy.contextMenus({ /* ... context menu options ... */
                 menuItems: [
                     { id: 'add-input-core', title: 'Add Input Node Here', selector: '', coreAsWell: true, onClickFunction: (evt) => addNodeFromMenu('input', evt.position || evt.cyPosition) },
                     { id: 'add-hidden-core', title: 'Add Hidden Node Here', selector: '', coreAsWell: true, onClickFunction: (evt) => addNodeFromMenu('hidden', evt.position || evt.cyPosition), hasTrailingDivider: true },
                     { id: 'convert-type', title: 'Convert Type', selector: 'node', onClickFunction: (evt) => convertNodeType(evt.target || evt.cyTarget) },
                     { id: 'delete-node', title: 'Delete Node', selector: 'node', hasTrailingDivider: true, onClickFunction: (evt) => deleteElement(evt.target || evt.cyTarget) },
                     { id: 'delete-edge', title: 'Delete Edge', selector: 'edge', onClickFunction: (evt) => deleteElement(evt.target || evt.cyTarget) }
                 ],
                 menuItemClasses: [ 'ctx-menu-item' ], contextMenuClasses: [ 'ctx-menu' ]
             });
            console.log("Context Menus initialized instance.");
            contextMenusRegistered = true; // Mark as successful
        } else { console.warn("cytoscape-context-menus lib or dependencies (Tippy/Popper) not found for registration."); }
    } catch (e) { console.error("Error initializing/registering Context Menus:", e); }

     // Update placeholder text based on registration success
     const editorPlaceholder = document.querySelector('.graph-editor p');
     if (editorPlaceholder) {
         let message = "";
         if (!contextMenusRegistered && !edgeHandlesRegistered) message = "Editing unavailable: Failed to load necessary libraries.";
         else if (!contextMenusRegistered) message = "Right-click editing disabled. Drag node handles to add edges.";
         else if (!edgeHandlesRegistered) message = "Edge drawing disabled. Use right-click menus to edit.";
         else message = "Right-click nodes/canvas to edit. Drag node handles to add edges."; // All good
          editorPlaceholder.textContent = message;
          if (!contextMenusRegistered || !edgeHandlesRegistered) editorPlaceholder.style.color = 'orange';
     }
}

// --- UI Initialization ---
function initializeUI() {
    console.log("Initializing UI event listeners...");
    // Config Buttons
    document.getElementById('save-config-button')?.addEventListener('click', saveConfiguration);
    document.getElementById('load-config-button')?.addEventListener('click', loadSelectedConfiguration);
    document.getElementById('set-default-button')?.addEventListener('click', setDefaultConfiguration);
    document.getElementById('delete-config-button')?.addEventListener('click', deleteSelectedConfiguration);
    // Prediction Buttons
    document.getElementById('update-button')?.addEventListener('click', fetchAndUpdateLLM);
    document.getElementById('gradient-toggle')?.addEventListener('change', () => updateNodeProbabilities(null));
    // Logging Buttons
    document.getElementById('download-session-log-button')?.addEventListener('click', downloadSessionLog);
    document.getElementById('download-all-log-button')?.addEventListener('click', downloadAllLogs);
    console.log("UI Listeners attached.");
}

// --- Editing Functions ---

function addNodeFromMenu(nodeType, position) {
     if (!cy) return;
    newNodeCounter++;
    const idBase = nodeType === 'input' ? 'Input' : 'Hidden';
    let id = `${idBase}_${newNodeCounter}`;
    while (cy.getElementById(id).length > 0) { newNodeCounter++; id = `${idBase}_${newNodeCounter}`; }

    const name = prompt(`Enter display name for new ${nodeType} node (ID: ${id}):`, id);
    if (name === null) return; // User cancelled

    const newNodeData = {
        group: 'nodes', data: { id: id.trim(), fullName: name.trim() || id.trim(), nodeType: nodeType }, position: position };
    cy.add(newNodeData);
    console.log(`Added ${nodeType} node: ID=${id}, Name=${newNodeData.data.fullName}`);
    if (nodeType === 'input') { updateInputControls(cy.nodes().map(n => n.data())); }
    updateNodeProbabilities({}); // Update appearance
    runLayout(); // Adjust layout
    markConfigUnsaved();
}

function deleteElement(target) {
     if (!cy || !target || (!target.isNode && !target.isEdge)) return;
    const id = target.id();
    const type = target.isNode() ? 'node' : 'edge';
    let name = target.data('fullName') || id;
    if (target.isEdge()){ name = `${target.source().id()} -> ${target.target().id()}`; }

    if (confirm(`Delete ${type} "${name}"?`)) {
        const wasInputNode = target.isNode() && target.data('nodeType') === 'input';
        cy.remove(target);
        console.log(`Removed ${type}: ${id}`);
        if(wasInputNode){ updateInputControls(cy.nodes().map(n => n.data())); }
        markConfigUnsaved();
    }
}

function convertNodeType(targetNode) {
    if (!cy || !targetNode || !targetNode.isNode()) return;
    const currentType = targetNode.data('nodeType');
    const newType = currentType === 'input' ? 'hidden' : 'input';
    const nodeId = targetNode.id();
    if (confirm(`Convert node "${targetNode.data('fullName') || nodeId}" from ${currentType} to ${newType}?`)) {
        targetNode.data('nodeType', newType);
        console.log(`Converted node ${nodeId} to ${newType}`);
        cy.style().update(); // Apply new styles
        updateInputControls(cy.nodes().map(n => n.data())); // Update input section
        updateNodeProbabilities({}); // Reset probability display
        markConfigUnsaved();
    }
}

// --- Configuration Management Functions ---

async function loadDefaultConfig() {
    // ... (Fetches default, stores in global defaultGraphStructure) ...
    console.log("Fetching default configuration structure...");
    try {
        // Use retryFetch for robustness
        await retryFetch(async () => {
            const response = await fetch('/api/configs/default');
            if (!response.ok) throw new Error(`HTTP error ${response.status} fetching default`);
            defaultGraphStructure = await response.json();
            console.log("Default config structure stored.", defaultGraphStructure);
        }, 3); // Retry 3 times
    } catch (error) {
        console.error('Error fetching default configuration structure:', error);
        defaultGraphStructure = null; // Ensure null on failure
        throw error; // Re-throw for main init block to handle
    }
}

function addDefaultToDropdown() {
    // ... (Adds default option to dropdown if not present) ...
     const select = document.getElementById('load-config-select');
    if (!defaultGraphStructure) return;
    let exists = false; for(let i=0; i < select.options.length; i++){ if (select.options[i].value === defaultGraphStructure.id) { exists = true; break; } }
    if (!exists) { const option = document.createElement('option'); option.value = defaultGraphStructure.id; option.textContent = `${defaultGraphStructure.name} (Default)`; if (select.options.length > 0 && select.options[0].value === "") { select.insertBefore(option, select.options[1]); } else { select.insertBefore(option, select.firstChild); } console.log("Added Default config to dropdown."); }
}

function loadGraphData(configData, isDefault = false) {
    // ... (Loads data into cy, updates currentConfig, UI elements) ...
     if (!cy) { console.error("Cannot load graph, Cytoscape not ready."); return; }
     if (!configData || !configData.graph_structure) { console.error("Invalid config data provided to loadGraphData."); return; }
     console.log(`Loading graph data for: ${configData.name}`);
     const graphElements = configData.graph_structure.nodes.map(n => ({ data: { id: n.id, fullName: n.fullName, nodeType: n.nodeType } })) .concat(configData.graph_structure.edges.map(e => ({ data: { source: e.source, target: e.target } })));
     cy.elements().remove(); cy.add(graphElements); runLayout();
     currentConfig = { ...configData };
     document.getElementById('config-name').value = isDefault ? '' : currentConfig.name;
     document.getElementById('current-config-name').textContent = currentConfig.name;
     updateInputControls(currentConfig.graph_structure.nodes);
     updateNodeProbabilities({}); clearSessionLog(); clearLLMOutputs(); clearUnsavedMark();
     // Selection happens *after* dropdown is populated
     selectConfigInDropdown(currentConfig.id);
}

async function saveConfiguration() {
    // ... (Gathers structure, validates name, calls POST /api/configs) ...
     const configNameInput = document.getElementById('config-name'); const configName = configNameInput.value.trim(); if (!configName) { setStatusMessage('config-status', "Please enter a name.", "error"); return; } if (!cy) { setStatusMessage('config-status', "Graph not initialized.", "error"); return; } setStatusMessage('config-status', "Saving...", "loading"); showSpinner(true); enableUI(false); const currentGraphStructure = { nodes: cy.nodes().map(n => ({ id: n.id(), fullName: n.data('fullName'), nodeType: n.data('nodeType') })), edges: cy.edges().map(e => ({ source: e.source().id(), target: e.target().id() })) };
     await retryFetch(async () => { const r = await fetch('/api/configs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ config_name: configName, graph_structure: currentGraphStructure }) }); const d = await r.json(); if (!r.ok) throw new Error(d.detail || `HTTP error ${r.status}`); currentConfig = { id: d.config_id, name: d.config_name, graph_structure: currentGraphStructure }; document.getElementById('current-config-name').textContent = currentConfig.name; configNameInput.value = ''; setStatusMessage('config-status', `Saved as '${d.config_name}'.`, "success"); await loadConfigList(); selectConfigInDropdown(d.config_id); clearSessionLog(); clearUnsavedMark(); }, 3, () => setStatusMessage('config-status', "Save failed. Retrying...", "error")) .catch(e => { console.error('Save config error:', e); setStatusMessage('config-status', `Save failed: ${e.message}`, "error"); }) .finally(() => { enableUI(true); showSpinner(false); });
}

async function loadConfigList() {
    // ... (Fetches GET /api/configs, updates dropdown, preserves selection) ...
    console.log("Updating saved configuration list...");
    try {
        // Use retryFetch
        await retryFetch(async () => {
            const response = await fetch('/api/configs');
            if (!response.ok) throw new Error(`HTTP error ${response.status} fetching list`);
            const configs = await response.json();
            const select = document.getElementById('load-config-select');
            const currentSelection = select.value;
            const optionsToRemove = []; for (let i = 0; i < select.options.length; i++) { const v = select.options[i].value; if (v !== "" && v !== "default-config-001") { optionsToRemove.push(select.options[i]); } } optionsToRemove.forEach(opt => select.removeChild(opt));
            configs.forEach(config => { const option = document.createElement('option'); option.value = config.id; option.textContent = config.name; select.appendChild(option); });
            select.value = currentSelection; let foundCurrent = false; if(currentConfig) { for(let i=0; i < select.options.length; i++){ if(select.options[i].value === currentConfig.id) { select.value = currentConfig.id; foundCurrent = true; break; } } } if(!foundCurrent && defaultGraphStructure){ select.value = defaultGraphStructure.id; } if(select.value === "") select.selectedIndex = 0;
            console.log("Saved config list updated.");
        }, 3); // Retry 3 times
    } catch (error) {
        console.error('Error loading configuration list:', error);
        setStatusMessage('config-status', `Failed to update saved list: ${error.message}`, "error");
    }
}

async function loadSelectedConfiguration() {
    // ... (Loads config based on dropdown selection, calls loadGraphData) ...
     const select = document.getElementById('load-config-select'); const configId = select.value; if (!configId) { setStatusMessage('config-status', "Select a config.", "error"); return; } if (!cy) { setStatusMessage('config-status', "Graph not ready.", "error"); return; } setStatusMessage('config-status', `Loading ${select.options[select.selectedIndex].text}...`, "loading"); showSpinner(true); enableUI(false);
     await retryFetch(async () => { if (configId === 'default-config-001' && defaultGraphStructure) { loadGraphData(defaultGraphStructure, true); setStatusMessage('config-status', "Default config loaded.", "success"); return; } const r = await fetch(`/api/configs/${configId}`); if (!r.ok) { const e = await r.json().catch(() => ({ detail: `HTTP error ${r.status}` })); throw new Error(e.detail || `HTTP error ${r.status}`); } const d = await r.json(); loadGraphData(d, false); setStatusMessage('config-status', `Config '${d.name}' loaded.`, "success"); }, 3, () => setStatusMessage('config-status', "Load failed. Retrying...", "error")) .catch(e => { console.error('Load config error:', e); setStatusMessage('config-status', `Load failed: ${e.message}`, "error"); }) .finally(() => { enableUI(true); showSpinner(false); });
}

async function setDefaultConfiguration() {
    // ... (Calls POST /api/configs/set_default) ...
     const select = document.getElementById('load-config-select'); const configId = select.value; const configName = select.options[select.selectedIndex].text; if (!configId || configId === 'default-config-001') { setStatusMessage('config-status', "Select saved config.", "error"); return; } if (!confirm(`Set "${configName}" as default?`)) return; setStatusMessage('config-status', `Setting default...`, "loading"); showSpinner(true); enableUI(false);
     await retryFetch(async () => { const r = await fetch('/api/configs/set_default', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(configId) }); const d = await r.json(); if (!r.ok) throw new Error(d.detail || `HTTP error ${r.status}`); setStatusMessage('config-status', `Set '${configName}' as default.`, "success"); await loadDefaultConfig(); /* Reload default structure in case it changed */ }, 3, () => setStatusMessage('config-status', "Set default failed. Retrying...", "error")) .catch(e => { console.error('Set default error:', e); setStatusMessage('config-status', `Set default failed: ${e.message}`, "error"); }) .finally(() => { enableUI(true); showSpinner(false); });
}

async function deleteSelectedConfiguration() {
    // ... (Calls DELETE /api/configs/{id}, reloads default if needed, updates list) ...
     const select = document.getElementById('load-config-select'); const configId = select.value; const configName = select.options[select.selectedIndex].text; if (!configId || configId === 'default-config-001') { setStatusMessage('config-status', "Select saved config.", "error"); return; } if (!confirm(`Delete "${configName}"?`)) return; setStatusMessage('config-status', `Deleting...`, "loading"); showSpinner(true); enableUI(false);
     await retryFetch(async () => { const r = await fetch(`/api/configs/${configId}`, { method: 'DELETE' }); const d = await r.json(); if (!r.ok) throw new Error(d.detail || `HTTP error ${r.status}`); setStatusMessage('config-status', `Deleted '${configName}'.`, "success"); if (currentConfig && currentConfig.id === configId) { if(defaultGraphStructure) { loadGraphData(defaultGraphStructure, true); setStatusMessage('config-status', `Deleted '${configName}'. Default loaded.`, "success"); } else { cy.elements().remove(); currentConfig = null; updateInputControls([]); clearSessionLog(); clearLLMOutputs(); document.getElementById('current-config-name').textContent = "None"; setStatusMessage('config-status', `Deleted '${configName}'. Load another.`, "success"); } } await loadConfigList(); }, 3, () => setStatusMessage('config-status', "Delete failed. Retrying...", "error")) .catch(e => { console.error('Delete config error:', e); setStatusMessage('config-status', `Delete failed: ${e.message}`, "error"); }) .finally(() => { enableUI(true); showSpinner(false); });
}

console.log("script.js loaded and executed.");
