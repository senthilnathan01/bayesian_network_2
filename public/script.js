// public/script.js

// --- Global Variables ---
let cy; // Cytoscape instance
let currentConfig = null; // Represents the currently loaded/active config object { id, name, graph_structure }
let defaultGraphStructure = null; // To store the structure fetched from backend
let sessionLog = []; // Logs for the current browser session
let edgeHandlesInstance = null; // Reference to edge handles extension API
let contextMenuInstance = null; // Reference to context menus extension API
let newNodeCounter = 0; // Counter for generating unique default node IDs

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initialize Cytoscape Instance
    initializeCytoscape();

    // Check if cy was initialized before proceeding
    if (cy) {
         // 2. Initialize Editing Extensions (Context Menus, Edge Handles)
         initializeEditingExtensions();
    } else {
        console.error("Cytoscape core failed to initialize. Cannot proceed.");
        setStatusMessage('config-status', "Error: Graph library failed to load.", "error");
        showSpinner(false); // Hide spinner if init fails early
        return; // Stop initialization if core fails
    }

    // 3. Initialize UI Button Listeners etc.
    initializeUI();

    // 4. Load Default Config and Saved List
    setStatusMessage('config-status', "Loading default graph...", "loading");
    showSpinner(true);
    try {
        await loadDefaultConfig(); // Fetch default structure into global var
        if (defaultGraphStructure) {
            loadGraphData(defaultGraphStructure, true); // Load default into cy
            addDefaultToDropdown(); // Add option to select
            selectConfigInDropdown(defaultGraphStructure.id); // Select it by default
            setStatusMessage('config-status', "Default config loaded.", "success");
        } else {
             // Handle case where default couldn't be fetched
             setStatusMessage('config-status', "Could not load default config. Select or save a configuration.", "error");
             updateInputControls([]);
             document.getElementById('current-config-name').textContent = "None";
        }
    } catch (error) {
        console.error("Error loading default config:", error);
        setStatusMessage('config-status', `Failed to load default config: ${error.message}. Check server status.`, "error");
        updateInputControls([]);
        document.getElementById('current-config-name').textContent = "None";
    }

    setStatusMessage('config-status', "Loading saved configurations...", "loading");
    try {
        await loadConfigList(); // Fetch saved configs and add to dropdown
        // Update final status message after both load attempts
        const finalStatus = currentConfig ? `Config '${currentConfig.name}' loaded.` : "Ready. Select or save a config.";
        // Check if status wasn't already set to an error by default load
        if (!document.getElementById('config-status').classList.contains('error')) {
             setStatusMessage('config-status', finalStatus, "success");
        }
    } catch (error) {
        console.error("Error loading saved configs:", error);
        // Don't overwrite previous error message if default load failed
        if (!document.getElementById('config-status').classList.contains('error')) {
            setStatusMessage('config-status', `Failed to load saved configs: ${error.message}. Check server status.`, "error");
        }
    } finally {
        showSpinner(false); // Hide spinner after all initial loading
    }
});

// --- Cytoscape Core Initialization ---
function initializeCytoscape() {
    if (cy) { console.warn("Cytoscape already initialized."); return; }
    console.log("Initializing Cytoscape core...");
    try {
        cy = cytoscape({
            container: document.getElementById('cy'),
            elements: [], // Start empty
            style: [ // Define node/edge appearance
                { selector: 'node', style: { 'background-color': '#ccc', 'label': nodeLabelFunc, 'width': 120, 'height': 120, 'shape': 'ellipse', 'text-valign': 'center', 'text-halign': 'center', 'font-size': '10px', 'font-weight': '100', 'text-wrap': 'wrap', 'text-max-width': 110, 'text-outline-color': '#fff', 'text-outline-width': 1, 'color': '#333', 'transition-property': 'background-color, color', 'transition-duration': '0.5s' } },
                { selector: 'node[nodeType="input"]', style: { 'shape': 'rectangle', 'width': 130, 'height': 70 } },
                { selector: 'node[nodeType="hidden"]', style: { 'shape': 'ellipse' } },
                { selector: 'edge', style: { 'width': 2, 'line-color': '#666', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#666', 'curve-style': 'bezier' } },
                // Style for indicating edge source node (used by removed logic, can keep or remove)
                { selector: '.edge-source-active', style: { 'border-width': 3, 'border-color': '#00ff00' } },
                 // Style for nodes being grabbed by edge handles
                 { selector: '.eh-grabbed', style: { 'border-width': 3, 'border-color': '#007bff' } }
            ],
             // Default layout, run explicitly later
             layout: { name: 'preset' } // Start with preset layout initially
        });
        console.log("Cytoscape core initialized successfully.");
    } catch(error) {
        console.error("Failed to initialize Cytoscape Core:", error);
        cy = null; // Ensure cy is null if init fails
        alert("Critical Error: Failed to initialize the graph visualization library. Please refresh the page.");
    }
}

// --- Editing Extensions Initialization ---
function initializeEditingExtensions() {
    console.log("Initializing editing extensions...");
    if (!cy) { console.error("Cannot init extensions: Cytoscape not ready."); return; }

    // --- Initialize Edge Handles (Primary edge drawing method) ---
    try {
        if (typeof cy.edgehandles === 'function') {
            edgeHandlesInstance = cy.edgehandles({
                snap: true, // Snap to nodes for easier connection
                handleNodes: 'node', // Show handle on all nodes
                handleSize: 10, // Size of the handle
                handleColor: '#007bff', // Color of the handle
                handlePosition: 'middle top', // Where the handle appears
                preview: true, // Show a preview line while dragging
                hoverDelay: 150, // Delay before handle appears on hover
                edgeType: (sourceNode, targetNode) => 'flat', // Type of edge (usually flat)
                loopAllowed: (node) => false, // Prevent self-loops
                nodeLoopOffset: -50, // Offset for loop edges if allowed
                nodeParams: (sourceNode, targetNode) => ({}), // Extra parameters for nodes (unused here)
                edgeParams: (sourceNode, targetNode, i) => ({ // Data for the new edge
                     data: { source: sourceNode.id(), target: targetNode.id() }
                 }),
                 ghostEdgeParams: () => ({}), // Params for the ghost edge during preview
                 // CSS classes for styling edge handles states
                 handleClass: 'eh-handle',
                 hoverClass: 'eh-hover',
                 snapClass: 'eh-snap',
                 sourceNodeClass: 'eh-source',
                 targetNodeClass: 'eh-target',
                 ghostEdgeClass: 'eh-ghost-edge',
                 previewClass: 'eh-preview',
                 grabbedNodeClass: 'eh-grabbed', // Class added to node when handle is grabbed

                // --- IMPORTANT: complete callback ---
                complete: ( sourceNode, targetNode, addedEntities ) => {
                    console.log("Edge added via handles:", addedEntities);
                    // Validate for cycles *after* the edge has been added by the extension
                    const currentEdges = cy.edges().map(e => ({ source: e.source().id(), target: e.target().id() }));
                    const currentGraph = { nodes: cy.nodes().map(n => n.data()), edges: currentEdges };
                    const isOk = simulateIsDag(currentGraph);

                    if (!isOk) {
                        alert(`Cannot add edge ${sourceNode.id()} -> ${targetNode.id()}. This would create a cycle.`);
                        // Remove the invalid edge
                        if (addedEntities) {
                             cy.remove(addedEntities);
                             console.log("Removed edge due to cycle creation.");
                        }
                        setStatusMessage('config-status', `Edge aborted (creates cycle).`, "error");
                    } else {
                         // Edge is valid and already added by the extension
                         // No need to call cy.add() again
                         markConfigUnsaved(); // Mark as unsaved
                         setStatusMessage('config-status', `Edge ${sourceNode.id()} -> ${targetNode.id()} added.`, "success");
                         // Optional: Re-run layout gently after adding edge
                         // runLayout(); // Can cause jumping, maybe only run layout before save?
                    }
                }
            });
            console.log("Edgehandles initialized.");
        } else {
             console.warn("cytoscape-edgehandles not available or not registered. Edge drawing disabled.");
             // Update placeholder text
             const editorPlaceholder = document.querySelector('.graph-editor p');
             if (editorPlaceholder) editorPlaceholder.textContent += " Edge drawing unavailable.";
        }
    } catch (e) { console.error("Error initializing Edgehandles:", e); }

    // --- Initialize Context Menus ---
    try {
        // Ensure dependencies are loaded
        if (typeof cy.contextMenus === 'function' && typeof tippy === 'function') {
            contextMenuInstance = cy.contextMenus({
                // Menu Items Definition
                menuItems: [
                    // Canvas Right Click
                    { id: 'add-input-core', title: 'Add Input Node Here', selector: '', coreAsWell: true, onClickFunction: (evt) => addNodeFromMenu('input', evt.position || evt.cyPosition) },
                    { id: 'add-hidden-core', title: 'Add Hidden Node Here', selector: '', coreAsWell: true, onClickFunction: (evt) => addNodeFromMenu('hidden', evt.position || evt.cyPosition), hasTrailingDivider: true },
                    // Node Right Click
                    { id: 'convert-type', title: 'Convert Type', selector: 'node', onClickFunction: (evt) => convertNodeType(evt.target || evt.cyTarget) },
                    { id: 'delete-node', title: 'Delete Node', selector: 'node', hasTrailingDivider: true, onClickFunction: (evt) => deleteElement(evt.target || evt.cyTarget) },
                    // Edge Right Click
                    { id: 'delete-edge', title: 'Delete Edge', selector: 'edge', onClickFunction: (evt) => deleteElement(evt.target || evt.cyTarget) }
                ],
                // CSS Classes for styling
                menuItemClasses: [ 'ctx-menu-item' ],
                contextMenuClasses: [ 'ctx-menu' ]
            });
            console.log("Context Menus initialized.");
        } else {
            console.warn("cytoscape-context-menus or dependencies not available or not registered. Right-click menus disabled.");
            const editorPlaceholder = document.querySelector('.graph-editor p');
             if (editorPlaceholder) editorPlaceholder.textContent = "Right-click editing unavailable: Menu library failed to load.";
        }
    } catch (e) { console.error("Error initializing Context Menus:", e); }
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

console.log("script.js loaded and executed.");
