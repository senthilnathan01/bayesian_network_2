// public/script.js
let cy;
let currentConfig = null;
let defaultGraphStructure = null;
let sessionLog = [];
let edgeHandlesInstance = null; // Reference to edge handles extension
let contextMenuInstance = null; // Reference to context menus extension


document.addEventListener('DOMContentLoaded', async () => {
    // Initialize Cytoscape first as extensions depend on it
    initializeCytoscape();

    // Check if cy was initialized before proceeding
    if (cy) {
         initializeEditingExtensions(); // Initialize editing features AFTER cy exists
    } else {
        console.error("Cytoscape core failed to initialize. Cannot proceed.");
        setStatusMessage('config-status', "Error: Graph library failed to load.", "error");
        return; // Stop initialization if core fails
    }

    // Initialize UI button listeners etc.
    initializeUI();

    // Load default config and saved list
    setStatusMessage('config-status', "Loading default graph...", "loading");
    showSpinner(true);
    try {
        await loadDefaultConfig(); // Fetch default structure
        if (defaultGraphStructure) {
            loadGraphData(defaultGraphStructure, true); // Load it into cy
            addDefaultToDropdown(); // Add option to select
            selectConfigInDropdown(defaultGraphStructure.id); // Select it by default
            setStatusMessage('config-status', "Default config loaded.", "success");
        } else {
            setStatusMessage('config-status', "Could not load default config.", "error");
            updateInputControls([]);
            document.getElementById('current-config-name').textContent = "None";
        }
    } catch (error) { console.error("Error loading default:", error); setStatusMessage('config-status', `Default load failed: ${error.message}`, "error"); updateInputControls([]); document.getElementById('current-config-name').textContent = "None"; }

    setStatusMessage('config-status', "Loading saved configurations...", "loading");
    try {
        await loadConfigList(); // Fetch saved configs
        // Update status based on what's loaded
        const finalStatus = currentConfig ? `Config '${currentConfig.name}' loaded.` : "Ready. Select or save a config.";
        setStatusMessage('config-status', finalStatus, "success");
    } catch (error) { console.error("Error loading saved list:", error); setStatusMessage('config-status', `Failed list load: ${error.message}`, "error"); }
    finally { showSpinner(false); }
});

function initializeCytoscape() {
    if (cy) { console.warn("Cytoscape already initialized."); return; }
    console.log("Initializing Cytoscape core...");
    try {
        cy = cytoscape({
            container: document.getElementById('cy'),
            elements: [],
            style: [
                { selector: 'node', style: { 'background-color': '#ccc', 'label': nodeLabelFunc, 'width': 120, 'height': 120, 'shape': 'ellipse', 'text-valign': 'center', 'text-halign': 'center', 'font-size': '10px', 'font-weight': '100', 'text-wrap': 'wrap', 'text-max-width': 110, 'text-outline-color': '#fff', 'text-outline-width': 1, 'color': '#333', 'transition-property': 'background-color, color', 'transition-duration': '0.5s' } },
                { selector: 'node[nodeType="input"]', style: { 'shape': 'rectangle', 'width': 130, 'height': 70 } },
                { selector: 'node[nodeType="hidden"]', style: { 'shape': 'ellipse' } },
                { selector: 'edge', style: { 'width': 2, 'line-color': '#666', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#666', 'curve-style': 'bezier' } }
            ],
            layout: { name: 'cola', animate: true, nodeSpacing: 50, edgeLength: 180, padding: 30 }
        });
        console.log("Cytoscape core initialized successfully.");
    } catch(error) {
        console.error("Failed to initialize Cytoscape Core:", error);
        cy = null; // Ensure cy is null if init fails
    }
}

function initializeEditingExtensions() {
    console.log("Initializing editing extensions...");
    if (!cy) { console.error("Cannot init extensions: Cytoscape not ready."); return; }

    // --- Initialize Edge Handles ---
    try {
        if (typeof cy.edgehandles === 'function') {
            edgeHandlesInstance = cy.edgehandles({
                snap: true, handleNodes: 'node', handleSize: 10, handleColor: '#007bff',
                edgeType: (sourceNode, targetNode) => 'flat',
                loopAllowed: (node) => false,
                edgeParams: (sourceNode, targetNode, i) => ({ data: { source: sourceNode.id(), target: targetNode.id() } }),
                complete: (sourceNode, targetNodes, addedEntities) => {
                    console.log("Edge added via handles:", addedEntities);
                    runLayout();
                    markConfigUnsaved(); // Mark as unsaved after adding edge
                }
            });
            console.log("Edgehandles initialized.");
        } else { console.warn("cytoscape-edgehandles not available or not registered."); }
    } catch (e) { console.error("Error initializing Edgehandles:", e); }

    // --- Initialize Context Menus ---
     // Use the setup from the user's provided code, ensuring checks
    try {
        if (typeof cy.contextMenus === 'function' && typeof tippy === 'function') {
            contextMenuInstance = cy.contextMenus({
                menuItems: [
                    // Add Node on Canvas Click
                    { id: 'add-input-node-core', title: 'Add Input Node Here', selector: '', coreAsWell: true,
                        onClickFunction: (event) => addNodeFromMenu('input', event.position || event.cyPosition) },
                    { id: 'add-hidden-node-core', title: 'Add Hidden Node Here', selector: '', coreAsWell: true, hasTrailingDivider: true,
                        onClickFunction: (event) => addNodeFromMenu('hidden', event.position || event.cyPosition) },

                    // Node actions
                    { id: 'remove-node', title: 'Delete Node', selector: 'node',
                        onClickFunction: (event) => deleteElement(event.target || event.cyTarget) },
                     { id: 'start-edge', title: 'Start Edge (Drag Handle)', selector: 'node', // Manual trigger if needed
                       onClickFunction: (event) => edgeHandlesInstance?.start(event.target || event.cyTarget) }, // Optional: Manually start edge draw

                    // Edge actions
                    { id: 'remove-edge', title: 'Delete Edge', selector: 'edge',
                        onClickFunction: (event) => deleteElement(event.target || event.cyTarget) }
                ],
                menuItemClasses: [ 'ctx-menu-item' ],
                contextMenuClasses: [ 'ctx-menu' ]
            });
            console.log("Context Menus initialized.");
        } else {
             console.warn("cytoscape-context-menus or dependencies not available or not registered.");
              // Display message in the placeholder if context menus failed specifically
              const editorPlaceholder = document.querySelector('.graph-editor p');
              if (editorPlaceholder) {
                  editorPlaceholder.textContent = "Right-click editing unavailable: Context menu library failed to load.";
                  editorPlaceholder.style.color = 'red';
              }
        }
    } catch (e) { console.error("Error initializing Context Menus:", e); }
}


// --- New Editing Functions ---

// Add a global counter near the top
let newNodeCounter = 0;

// Modify addNodeFromMenu
function addNodeFromMenu(nodeType, position) {
    if (!cy) return;
    newNodeCounter++; // Increment counter
    const idBase = nodeType === 'input' ? 'Input' : 'Hidden';
    let id = `${idBase}_${newNodeCounter}`;
    // Ensure unique ID (less likely needed with counter, but good practice)
    while (cy.getElementById(id).length > 0) {
        newNodeCounter++;
        id = `${idBase}_${newNodeCounter}`;
    }

    // Prompt for the DISPLAY NAME, using the ID as a suggestion
    const name = prompt(`Enter display name for new ${nodeType} node (ID: ${id}):`, id); // Use ID as default name
    if (name === null) return; // User cancelled

    const newNodeData = {
        group: 'nodes',
        data: {
            id: id.trim(), // Use generated ID
            fullName: name.trim() || id.trim(), // Use entered name or fallback to ID
            nodeType: nodeType
        },
        position: position
    };

    cy.add(newNodeData);
    console.log(`Added ${nodeType} node: ID=${id}, Name=${newNodeData.data.fullName}`);
    if (nodeType === 'input') { updateInputControls(cy.nodes().map(n => n.data())); }
    updateNodeProbabilities({});
    runLayout();
    markConfigUnsaved();
}

function deleteElement(target) {
     if (!cy || !target || (!target.isNode && !target.isEdge)) return;
    const id = target.id();
    const type = target.isNode() ? 'node' : 'edge';
    if (confirm(`Delete ${type} "${target.data('fullName') || id}"?`)) {
        const wasInputNode = target.isNode() && target.data('nodeType') === 'input';
        cy.remove(target);
        console.log(`Removed ${type}: ${id}`);
        if(wasInputNode){
            updateInputControls(cy.nodes().map(n => n.data())); // Refresh inputs if node was removed
        }
        markConfigUnsaved();
        // Optional: runLayout(); // Might not be needed for deletion
    }
}

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

// --- Modified Config Management ---

function loadGraphData(configData, isDefault = false) {
    // ... (Loads graph, updates UI) ...
     if (!cy) { console.error("Cannot load graph, Cytoscape not ready."); return; }
     if (!configData || !configData.graph_structure) { console.error("Invalid config data provided to loadGraphData."); return; }
     console.log(`Loading graph data for: ${configData.name}`);
     const graphElements = configData.graph_structure.nodes.map(n => ({ data: { id: n.id, fullName: n.fullName, nodeType: n.nodeType } }))
         .concat(configData.graph_structure.edges.map(e => ({ data: { source: e.source, target: e.target } })));
     cy.elements().remove(); cy.add(graphElements); runLayout();
     currentConfig = { ...configData };
     document.getElementById('config-name').value = isDefault ? '' : currentConfig.name;
     document.getElementById('current-config-name').textContent = currentConfig.name; // Set clean name
     selectConfigInDropdown(currentConfig.id);
     updateInputControls(currentConfig.graph_structure.nodes);
     updateNodeProbabilities({});
     clearSessionLog();
     clearLLMOutputs();
     clearUnsavedMark(); // Clear '*' on successful load
     // Status message set by caller
}


async function saveConfiguration() {
    // ... (Gathers structure, calls API) ...
     const configNameInput = document.getElementById('config-name');
    const configName = configNameInput.value.trim();
    if (!configName) { setStatusMessage('config-status', "Please enter a name.", "error"); return; }
    if (!cy) { setStatusMessage('config-status', "Graph not initialized.", "error"); return; }
    setStatusMessage('config-status', "Saving...", "loading"); showSpinner(true); enableUI(false);
    const currentGraphStructure = {
        nodes: cy.nodes().map(node => ({ id: node.id(), fullName: node.data('fullName'), nodeType: node.data('nodeType') })),
        edges: cy.edges().map(edge => ({ source: edge.source().id(), target: edge.target().id() }))
    };
    await retryFetch(async () => {
        const response = await fetch('/api/configs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ config_name: configName, graph_structure: currentGraphStructure }) });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);
        currentConfig = { id: result.config_id, name: result.config_name, graph_structure: currentGraphStructure };
        document.getElementById('current-config-name').textContent = currentConfig.name; // Set clean name
        configNameInput.value = ''; // Clear input
        setStatusMessage('config-status', `Saved as '${result.config_name}'.`, "success");
        await loadConfigList();
        selectConfigInDropdown(result.config_id);
        clearSessionLog();
        clearUnsavedMark(); // Clear '*' on successful save
    }, 3, () => setStatusMessage('config-status', "Save failed. Retrying...", "error"))
    .catch(error => { // Handle final fetch error
        console.error('Error saving configuration:', error);
        setStatusMessage('config-status', `Save failed: ${error.message}`, "error");
    })
    .finally(() => { enableUI(true); showSpinner(false); });
}

// --- Rest of the script ---
// (initializeUI, loadDefaultConfig, addDefaultToDropdown, loadConfigList, loadSelectedConfiguration, setDefaultConfiguration, deleteSelectedConfiguration, selectConfigInDropdown, enableUI, showSpinner, runLayout, updateInputControls, fetchAndUpdateLLM, updateNodeProbabilities, displayLLMReasoning, displayLLMContext, setStatusMessage, retryFetch, logPrediction, clearSessionLog, clearLLMOutputs, downloadSessionLog, downloadAllLogs, triggerCsvDownload)
// Ensure all these functions exist and are correct as per the previous iteration or your latest version.
// Key functions like retryFetch, showSpinner, setStatusMessage are assumed to be present from your code.

// Make sure these helper functions (used above) are defined correctly:

function nodeLabelFunc(node) { const id = node.data('id'); const fullName = node.data('fullName') || id; const currentLabel = node.data('currentProbLabel'); return `${id}: ${fullName}\n${currentLabel || '(N/A)'}`; }
function selectConfigInDropdown(configId) { const select = document.getElementById('load-config-select'); select.value = configId; }
function enableUI(enable) { const buttons = document.querySelectorAll('button'); const inputs = document.querySelectorAll('input, select'); buttons.forEach(btn => btn.disabled = !enable); inputs.forEach(inp => inp.disabled = !enable); document.getElementById('gradient-toggle').disabled = false; document.body.style.cursor = enable ? 'default' : 'wait'; }
function showSpinner(show) { document.getElementById('loading-spinner').classList.toggle('hidden', !show); }
function runLayout() { if (!cy) return; let l = 'cola'; try { cy.layout({ name: 'cola' }).run(); } catch (e) { console.warn('Cola failed, using dagre'); l = 'dagre'; try { cy.layout({ name: 'dagre' }).run(); } catch (e) { console.error('Layouts failed'); l = 'grid'; cy.layout({ name: 'grid' }).run(); } } console.log('Using layout:', l); }
function updateInputControls(nodes) { const c = document.getElementById('input-controls-container'); c.innerHTML = ''; const iN = nodes.filter(n => n.nodeType === 'input'); if (iN.length === 0) { c.innerHTML = '<p>No input nodes.</p>'; return; } iN.forEach(n => { const d = document.createElement('div'); const l = document.createElement('label'); l.htmlFor = `input-${n.id}`; l.textContent = `${n.id} (${n.fullName}):`; const i = document.createElement('input'); i.type = 'number'; i.id = `input-${n.id}`; i.name = n.id; i.min = "0"; i.max = "1"; i.step = "0.01"; i.value = "0.5"; d.appendChild(l); d.appendChild(i); c.appendChild(d); }); }
async function fetchAndUpdateLLM() { if (!currentConfig || !currentConfig.graph_structure || !currentConfig.graph_structure.nodes.length === 0) { alert("Load config first."); return; } if (!cy) { alert("Graph not ready."); return; } setStatusMessage('predict-status', "Gathering inputs...", "loading"); let inputs = { input_values: {} }; let invalid = false; currentConfig.graph_structure.nodes.filter(n => n.nodeType === 'input').forEach(n => { const el = document.getElementById(`input-${n.id}`); const cont = el?.parentElement; let v = 0.5; if (el) { v = parseFloat(el.value); if (isNaN(v) || v < 0 || v > 1) { setStatusMessage('predict-status', `Invalid input for ${n.id}.`, "error"); cont?.classList.add('invalid-input'); invalid = true; } else { cont?.classList.remove('invalid-input'); } } inputs.input_values[n.id] = isNaN(v)?0.5:Math.max(0, Math.min(1, v)); }); if (invalid) { setStatusMessage('predict-status', "Fix invalid inputs (0-1).", "error"); return; } const payload = { ...inputs, graph_structure: currentConfig.graph_structure, config_id: currentConfig.id, config_name: currentConfig.name }; setStatusMessage('predict-status', "Running prediction...", "loading"); showSpinner(true); enableUI(false); clearLLMOutputs(); await retryFetch(async () => { const r = await fetch('/api/predict_openai_bn_single_call', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }); const d = await r.json(); if (!r.ok) throw new Error(d.detail || `HTTP error ${r.status}`); updateNodeProbabilities(d.probabilities); displayLLMReasoning(d.llm_reasoning); displayLLMContext(d.llm_context); setStatusMessage('predict-status', "Prediction complete.", "success"); logPrediction(inputs.input_values, d.probabilities); }, 3, () => setStatusMessage('predict-status', "Prediction failed. Retrying...", "error")).catch(e => { console.error("Predict error:", e); setStatusMessage('predict-status', `Prediction failed: ${e.message}`, "error"); clearLLMOutputs(); }).finally(() => { enableUI(true); showSpinner(false); }); }
function updateNodeProbabilities(probabilities) { if (!cy) return; const useGradient = document.getElementById('gradient-toggle').checked; cy.nodes().forEach(node => { const nodeId = node.id(); const nodeType = node.data('nodeType'); let probState1 = null; if (probabilities && probabilities[nodeId] && probabilities[nodeId]["1"] !== undefined) { probState1 = probabilities[nodeId]["1"]; node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`); } else if (nodeType === 'input') { const inputElement = document.getElementById(`input-${nodeId}`); const currentVal = inputElement ? (parseFloat(inputElement.value) || 0.5) : 0.5; probState1 = currentVal; node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`); } else { node.data('currentProbLabel', '(N/A)'); } let baseBgColor = nodeType === 'input' ? '#add8e6' : '#f0e68c'; let textColor = '#333'; let finalBgColor = baseBgColor; if (probState1 !== null) { if (useGradient) { finalBgColor = `rgb(${Math.round(255 * (1 - probState1))}, ${Math.round(255 * probState1)}, 0)`; textColor = '#333'; } else { finalBgColor = '#4B0082'; textColor = '#FFFFFF'; } } else if (!useGradient && nodeType !== 'input') { finalBgColor = '#4B0082'; textColor = '#FFFFFF'; } node.style({ 'background-color': finalBgColor, 'color': textColor }); }); cy.style().update(); }
function displayLLMReasoning(reasoningText) { const d = document.getElementById('llm-reasoning-content'); d.textContent = reasoningText || "N/A"; }
function displayLLMContext(context) { if (!context) return; const iD = document.getElementById('input-context'); const sD = document.getElementById('structure-context'); const dD = document.getElementById('node-descriptions-context'); let iH='<ul>';(context.input_states||[]).forEach(s=>{iH+=`<li>${s.node}(${s.description}): ${s.state} (p=${s.value.toFixed(2)})</li>`;});iH+='</ul>';iD.innerHTML=iH||'N/A'; let sH='<ul>';Object.entries(context.node_dependencies||{}).forEach(([n,p])=>{sH+=`<li>${n}: ${p.join(',')||'None'}</li>`;});sH+='</ul>';sD.innerHTML=sH||'N/A'; let dH='<ul>';Object.entries(context.node_descriptions||{}).forEach(([n,d])=>{dH+=`<li>${n}: ${d}</li>`;});dH+='</ul>';dD.innerHTML=dH||'N/A';}
function setStatusMessage(elementId, message, type) { const el = document.getElementById(elementId); if (el) { el.textContent = message; el.className = `status-message ${type||''}`; } }
async function retryFetch(fetchFn, maxRetries, onRetry) { let lastError; for (let attempt = 1; attempt <= maxRetries; attempt++) { try { await fetchFn(); return; } catch (error) { lastError = error; console.warn(`Attempt ${attempt} failed: ${error.message}`); if (attempt < maxRetries) { if(onRetry) onRetry(); await new Promise(resolve => setTimeout(resolve, 1000 * attempt)); } } } throw lastError; }
function logPrediction(inputs, probabilities) { const logEntry = { timestamp: new Date().toISOString(), configId: currentConfig ? currentConfig.id : "unknown", configName: currentConfig ? currentConfig.name : "Unknown", inputs: { ...inputs }, probabilities: {} }; for (const n in probabilities) { logEntry.probabilities[n] = probabilities[n]["1"]; } for (const i in inputs) { if (!(i in logEntry.probabilities)) { logEntry.probabilities[i] = inputs[i]; } } sessionLog.push(logEntry); document.getElementById('log-count').textContent = `Session logs: ${sessionLog.length}`; }
function clearSessionLog() { sessionLog = []; document.getElementById('log-count').textContent = `Session logs: 0`; }
function clearLLMOutputs() { document.getElementById('llm-reasoning-content').textContent = 'Run prediction...'; document.getElementById('input-context').innerHTML = '<p>N/A</p>'; document.getElementById('structure-context').innerHTML = '<p>N/A</p>'; document.getElementById('node-descriptions-context').innerHTML = '<p>N/A</p>'; setStatusMessage('predict-status', "", ""); }
function downloadSessionLog() { if (sessionLog.length === 0) { alert("No logs."); return; } const h=['Timestamp','ConfigID','ConfigName','NodeID','ProbP1']; const r=[]; sessionLog.forEach(l=>{Object.entries(l.probabilities).forEach(([n,p])=>{r.push([l.timestamp,l.configId,l.configName,n,p.toFixed(4)]);});}); const csv=Papa.unparse({fields:h,data:r}); triggerCsvDownload(csv, `session_log_${(currentConfig?.name||'unsaved').replace(/[^a-z0-9]/gi,'_')}`); }
async function downloadAllLogs() { if (!currentConfig || !currentConfig.id || currentConfig.id === "unknown" || currentConfig.id === "default-config-001") { alert("Load saved config."); return; } setStatusMessage('predict-status', "Downloading logs...", "loading"); showSpinner(true); enableUI(false); await retryFetch(async () => { const r = await fetch(`/api/download_log/${currentConfig.id}`); if (!r.ok) { const e = await r.json().catch(()=>({detail:`HTTP ${r.status}`})); throw new Error(e.detail); } const b = await r.blob(); triggerCsvDownload(b, `all_logs_${currentConfig.name.replace(/[^a-z0-9]/gi,'_')}`); setStatusMessage('predict-status', "Logs downloaded.", "success"); }, 3, () => setStatusMessage('predict-status', "Download failed. Retrying...", "error")).catch(e=>{setStatusMessage('predict-status',`Log download failed: ${e.message}`,"error");}).finally(() => { enableUI(true); showSpinner(false); }); }
function triggerCsvDownload(csvDataOrBlob, baseFilename) { const blob = (csvDataOrBlob instanceof Blob) ? csvDataOrBlob : new Blob([csvDataOrBlob], { type: 'text/csv;charset=utf-8;' }); const link = document.createElement("a"); const url = URL.createObjectURL(blob); link.setAttribute("href", url); const timestampStr = new Date().toISOString().replace(/[:.]/g, '-'); link.setAttribute("download", `${baseFilename}_${timestampStr}.csv`); link.style.visibility = 'hidden'; document.body.appendChild(link); link.click(); document.body.removeChild(link); URL.revokeObjectURL(url); }
