// --- public/script.js ---

// Global variables
let cy;
let currentConfig = null; // Represents the currently loaded/active config object
let defaultGraphStructure = null; // To store the structure fetched from backend
let sessionLog = [];

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initialize Cytoscape Instance
    initializeCytoscape(); // Creates the 'cy' instance

    // 2. Initialize UI Listeners (can happen early)
    initializeUI();

    // 3. Fetch and Load Default Config
    setStatusMessage('config-status', "Loading default graph...", "loading");
    try {
        await loadDefaultConfig(); // Fetches and stores default structure in global var
        if (defaultGraphStructure) {
            // Load the default graph into Cytoscape immediately
            loadGraphData(defaultGraphStructure, true); // 'true' marks it as default loading
            setStatusMessage('config-status', "Default config loaded.", "success");
             // Add default to dropdown *after* it's loaded
             addDefaultToDropdown();
        } else {
            // Handle case where default couldn't be fetched
             setStatusMessage('config-status', "Could not load default config. Select or save a configuration.", "error");
              updateInputControls([]); // Ensure inputs are cleared if default fails
              document.getElementById('current-config-name').textContent = "None";
        }
    } catch (error) {
        console.error("Initialization error (loading default):", error);
        setStatusMessage('config-status', `Failed to load default config: ${error.message}`, "error");
         updateInputControls([]);
         document.getElementById('current-config-name').textContent = "None";
    }

    // 4. Fetch and Populate Saved Configs List (after default is handled)
     setStatusMessage('config-status', "Loading saved configurations...", "loading"); // Update status
    try {
        await loadConfigList(); // Fetches saved configs and adds them to dropdown
         // Final status message depends on whether default loaded and saved configs exist
         if (currentConfig && currentConfig.id === 'default-config-001'){
              setStatusMessage('config-status', "Default config loaded. Load or save other configs.", "success");
         } else if (currentConfig) {
             setStatusMessage('config-status', `Config '${currentConfig.name}' loaded.`, "success");
         } else if (document.getElementById('load-config-select').options.length > 1) { // Check if saved configs were added
             setStatusMessage('config-status', "Ready. Select a saved configuration.", "success");
         } else {
              setStatusMessage('config-status', "Ready. Save your first configuration.", "success");
         }

    } catch (error) {
         console.error("Initialization error (loading saved list):", error);
        // Keep previous status message (likely about default config state)
    }
});

function initializeCytoscape() {
    // ... (Cytoscape initialization code remains the same) ...
    if (cy) { console.warn("Cytoscape already initialized."); return; }
    console.log("Initializing Cytoscape...");
    cy = cytoscape({
        container: document.getElementById('cy'),
        elements: [],
        style: [
            { selector: 'node', style: { 'background-color': '#ccc', 'label': nodeLabelFunc, 'width': 120, 'height': 120, 'shape': 'ellipse', 'text-valign': 'center', 'text-halign': 'center', 'font-size': '10px', 'font-weight': '150', 'text-wrap': 'wrap', 'text-max-width': 110, 'text-outline-color': '#fff', 'text-outline-width': 1, 'color': '#333', 'transition-property': 'background-color, color', 'transition-duration': '0.5s' } },
            { selector: 'node[nodeType="input"]', style: { 'shape': 'rectangle', 'width': 130, 'height': 70 } },
            { selector: 'node[nodeType="hidden"]', style: { 'shape': 'ellipse' } },
            { selector: 'edge', style: { 'width': 2, 'line-color': '#666', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#666', 'curve-style': 'bezier' } }
        ],
        layout: { name: 'cola', animate: true, nodeSpacing: 50, edgeLength: 180, padding: 30 }
    });
    console.log("Cytoscape initialized.");
}

function nodeLabelFunc(node) {
    // ... (Label function remains the same) ...
    const id = node.data('id');
    const fullName = node.data('fullName') || id;
    const currentLabel = node.data('currentProbLabel');
    return `${id}: ${fullName}\n${currentLabel || '(N/A)'}`;
}

function initializeUI() {
    // ... (UI listeners remain the same) ...
    console.log("Initializing UI event listeners...");
    document.getElementById('save-config-button').addEventListener('click', saveConfiguration);
    document.getElementById('load-config-button').addEventListener('click', loadSelectedConfiguration);
    document.getElementById('delete-config-button').addEventListener('click', deleteSelectedConfiguration);
    document.getElementById('update-button').addEventListener('click', fetchAndUpdateLLM);
    document.getElementById('gradient-toggle').addEventListener('change', () => updateNodeProbabilities(null));
    document.getElementById('download-session-log-button').addEventListener('click', downloadSessionLog);
    document.getElementById('download-all-log-button').addEventListener('click', downloadAllLogs);
    console.log("UI Listeners attached.");
}

// --- Configuration Management ---

async function loadDefaultConfig() {
    // Fetches default config structure and stores it globally
    console.log("Fetching default configuration structure...");
    try {
        const response = await fetch('/api/configs/default'); // Fetch from the dedicated endpoint
        if (!response.ok) throw new Error(`HTTP error ${response.status} fetching default config`);
        defaultGraphStructure = await response.json(); // Store globally
        console.log("Default config structure stored.", defaultGraphStructure);
    } catch (error) {
        console.error('Error fetching default configuration structure:', error);
        defaultGraphStructure = null; // Ensure it's null if fetch fails
        throw error; // Re-throw to be caught by the caller
    }
}

function addDefaultToDropdown() {
    // Adds the default config option to the dropdown if structure is available
    const select = document.getElementById('load-config-select');
    if (!defaultGraphStructure) return; // Don't add if fetch failed

    // Check if default option already exists to prevent duplicates
    let defaultOptionExists = false;
    for(let i=0; i < select.options.length; i++){
        if (select.options[i].value === 'default-config-001') {
            defaultOptionExists = true;
            break;
        }
    }

    if (!defaultOptionExists) {
        const option = document.createElement('option');
        option.value = defaultGraphStructure.id; // Use the actual ID
        option.textContent = `${defaultGraphStructure.name} (Default)`;
        // Insert after the placeholder "-- Select a config --"
        if (select.options.length > 0) {
            select.insertBefore(option, select.options[1]); // Insert at index 1
        } else {
            select.appendChild(option); // Should not happen if placeholder exists
        }
        console.log("Added Default config to dropdown.");
    }
}

function loadGraphData(configData, isDefault = false) {
    // Loads graph structure into Cytoscape and updates UI state
     if (!cy) { console.error("Cannot load graph, Cytoscape not ready."); return; }
     if (!configData || !configData.graph_structure) { console.error("Invalid config data provided to loadGraphData."); return; }

     console.log(`Loading graph data for: ${configData.name}`);
     const graphElements = configData.graph_structure.nodes.map(n => ({ data: { id: n.id, fullName: n.fullName, nodeType: n.nodeType } }))
         .concat(configData.graph_structure.edges.map(e => ({ data: { source: e.source, target: e.target } })));

     cy.elements().remove();
     cy.add(graphElements);
     runLayout();

     currentConfig = { ...configData }; // Update global state - make a copy
     document.getElementById('config-name').value = isDefault ? '' : currentConfig.name; // Don't prefill name for default
     document.getElementById('current-config-name').textContent = currentConfig.name;
     selectConfigInDropdown(currentConfig.id); // Select the loaded config in dropdown
     updateInputControls(currentConfig.graph_structure.nodes);
     updateNodeProbabilities({}); // Reset probabilities display
     clearSessionLog();
     clearLLMOutputs();
     // Status message is usually set by the calling function (loadSelectedConfiguration or initial load)
}


async function saveConfiguration() {
    // ... (Save logic remains the same) ...
    const configNameInput = document.getElementById('config-name');
    const configName = configNameInput.value.trim();
    if (!configName) { setStatusMessage('config-status', "Please enter a name.", "error"); return; }
    if (!cy) { setStatusMessage('config-status', "Graph not initialized.", "error"); return; }

    setStatusMessage('config-status', "Saving...", "loading");
    enableUI(false);

    const currentGraphStructure = {
        nodes: cy.nodes().map(node => ({ id: node.id(), fullName: node.data('fullName'), nodeType: node.data('nodeType') })),
        edges: cy.edges().map(edge => ({ source: edge.source().id(), target: edge.target().id() }))
    };

    try {
        const response = await fetch('/api/configs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config_name: configName, graph_structure: currentGraphStructure })
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);

        // Update currentConfig ONLY if saving was successful
        currentConfig = { id: result.config_id, name: result.config_name, graph_structure: currentGraphStructure };
        document.getElementById('current-config-name').textContent = currentConfig.name;
        configNameInput.value = '';
        setStatusMessage('config-status', `Saved as '${result.config_name}'.`, "success");
        await loadConfigList(); // Refresh dropdown (will include the new one)
        selectConfigInDropdown(result.config_id); // Select the newly saved one
        clearSessionLog();

    } catch (error) {
        console.error('Error saving configuration:', error);
        setStatusMessage('config-status', `Save failed: ${error.message}`, "error");
        // Do NOT update currentConfig if save failed
    } finally {
        enableUI(true);
    }
}

async function loadConfigList() {
    // Fetches SAVED configs and populates dropdown (excluding default)
    console.log("Updating saved configuration list...");
    // Don't show loading message here, it's part of initial load or follows save/delete
    try {
        const response = await fetch('/api/configs'); // Fetches only saved configs
        if (!response.ok) throw new Error(`HTTP error ${response.status} fetching list`);
        const configs = await response.json();

        const select = document.getElementById('load-config-select');
        const currentSelection = select.value;

        // Clear existing options *except* placeholder and default
        const optionsToRemove = [];
        for (let i = 0; i < select.options.length; i++) {
            const optValue = select.options[i].value;
            if (optValue !== "" && optValue !== "default-config-001") {
                 optionsToRemove.push(select.options[i]);
            }
        }
        optionsToRemove.forEach(opt => select.removeChild(opt));

        // Add fetched configs
        configs.forEach(config => {
            const option = document.createElement('option');
            option.value = config.id;
            option.textContent = config.name;
            select.appendChild(option);
        });

        // Try to restore selection, fallback to current if possible, then default, then placeholder
        select.value = currentSelection;
         if (select.value !== currentSelection) { // If selection wasn't found (e.g., deleted)
             if (currentConfig) select.value = currentConfig.id; // Try current config
             else if (defaultGraphStructure) select.value = defaultGraphStructure.id; // Try default
             else select.selectedIndex = 0; // Fallback to placeholder
         }
        console.log("Saved config list updated.");

    } catch (error) {
        console.error('Error loading configuration list:', error);
        setStatusMessage('config-status', `Failed to update saved list: ${error.message}`, "error");
        // Don't clear existing list on error, just show message
    }
}

async function loadSelectedConfiguration() {
    // Loads the config selected in the dropdown
    const select = document.getElementById('load-config-select');
    const configId = select.value;
    if (!configId) { setStatusMessage('config-status', "Select a config.", "error"); return; }
    if (!cy) { setStatusMessage('config-status', "Graph not ready.", "error"); return; }

    // Handle loading default config via dropdown selection
    if (configId === 'default-config-001') {
        if (defaultGraphStructure) {
            loadGraphData(defaultGraphStructure, true);
            setStatusMessage('config-status', "Default config loaded.", "success");
        } else {
            setStatusMessage('config-status', "Default config data unavailable.", "error");
            await loadDefaultConfig(); // Attempt to refetch it
             if(defaultGraphStructure) loadGraphData(defaultGraphStructure, true);
        }
        return;
    }

    // Handle loading a saved config
    setStatusMessage('config-status', `Loading ${select.options[select.selectedIndex].text}...`, "loading");
    enableUI(false);

    try {
        const response = await fetch(`/api/configs/${configId}`);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` }));
            throw new Error(errorData.detail || `HTTP error ${response.status}`);
        }
        const configData = await response.json();
        loadGraphData(configData, false); // Load fetched data (not default)
        setStatusMessage('config-status', `Config '${configData.name}' loaded.`, "success");

    } catch (error) {
        console.error('Error loading selected configuration:', error);
        setStatusMessage('config-status', `Load failed: ${error.message}`, "error");
        // Attempt to reload default if load fails? Or just show error. Showing error is safer.
         // if(defaultGraphStructure) loadGraphData(defaultGraphStructure, true);
    } finally {
        enableUI(true);
    }
}


async function deleteSelectedConfiguration() {
    // ... (Delete logic remains the same) ...
     const select = document.getElementById('load-config-select');
    const configId = select.value;
    const configName = select.options[select.selectedIndex].text;

    if (!configId || configId === 'default-config-001') {
        setStatusMessage('config-status', "Select a saved config to delete.", "error");
        return;
    }
    if (!confirm(`Delete "${configName}"?\nThis also deletes logs.`)) return;

    setStatusMessage('config-status', `Deleting ${configName}...`, "loading");
    enableUI(false);
    try {
        const response = await fetch(`/api/configs/${configId}`, { method: 'DELETE' });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);
        setStatusMessage('config-status', `Deleted '${configName}'.`, "success");
        // Load default if the deleted one was active
        if (currentConfig && currentConfig.id === configId) {
             if(defaultGraphStructure) {
                 loadGraphData(defaultGraphStructure, true);
                 setStatusMessage('config-status', `Deleted '${configName}'. Default config loaded.`, "success");
             } else {
                 cy.elements().remove(); currentConfig = null; updateInputControls([]); clearSessionLog(); clearLLMOutputs(); document.getElementById('current-config-name').textContent = "None";
                 setStatusMessage('config-status', `Deleted '${configName}'. Load another config.`, "success");
             }
        }
        await loadConfigList(); // Refresh dropdown (will remove the deleted one)
    } catch (error) {
        console.error('Error deleting configuration:', error);
        setStatusMessage('config-status', `Delete failed: ${error.message}`, "error");
    } finally {
        enableUI(true);
    }
}

function selectConfigInDropdown(configId) {
    // Helper to select an option by value
    const select = document.getElementById('load-config-select');
    select.value = configId;
}

// --- UI Enable/Disable ---
function enableUI(enable) {
    // ... (Remains the same) ...
    const buttons = document.querySelectorAll('button');
    const inputs = document.querySelectorAll('input, select');
    buttons.forEach(btn => btn.disabled = !enable);
    inputs.forEach(inp => inp.disabled = !enable);
     document.getElementById('gradient-toggle').disabled = false;
    document.body.style.cursor = enable ? 'default' : 'wait';
}

// --- Layout ---
function runLayout() {
    // ... (Remains the same - cola/dagre fallback) ...
    if (!cy) return;
    let layoutName = 'cola';
    try { if (typeof cy.layout({ name: 'cola' }).run !== 'function') throw new Error(); }
    catch(e) { console.warn("Cola failed, trying dagre..."); layoutName = 'dagre';
          try { if (typeof cy.layout({ name: 'dagre' }).run !== 'function') throw new Error(); }
          catch(e) { console.error("Layouts failed!"); layoutName = 'grid'; } }
    const options = layoutName === 'cola' ? { name: 'cola', animate: true, nodeSpacing: 50, edgeLength: 180, padding: 30 }
                    : layoutName === 'dagre' ? { name: 'dagre', rankDir: 'TB', spacingFactor: 1.2, animate: true, padding: 30 }
                    : { name: 'grid' };
    cy.layout(options).run();
}

// --- Dynamic Input Controls ---
function updateInputControls(nodes) {
    // ... (Remains the same) ...
    const container = document.getElementById('input-controls-container');
    container.innerHTML = '';
    const inputNodes = nodes.filter(n => n.nodeType === 'input');
    if (inputNodes.length === 0) { container.innerHTML = '<p>No input nodes defined.</p>'; return; }
    inputNodes.forEach(node => {
        const div = document.createElement('div');
        const label = document.createElement('label'); label.htmlFor = `input-${node.id}`; label.textContent = `${node.id} (${node.fullName}):`;
        const input = document.createElement('input'); input.type = 'number'; input.id = `input-${node.id}`; input.name = node.id; input.min = "0"; input.max = "1"; input.step = "0.01"; input.value = "0.5";
        div.appendChild(label); div.appendChild(input); container.appendChild(div);
    });
}

// --- Prediction ---
async function fetchAndUpdateLLM() {
    // ... (Remains the same - gathers inputs, calls backend, handles response) ...
     if (!currentConfig || !currentConfig.graph_structure || currentConfig.graph_structure.nodes.length === 0) { alert("Load a config first."); return; }
    if (!cy) { alert("Graph not ready."); return; }

    setStatusMessage('predict-status', "Gathering inputs...", "loading");
    const inputData = { input_values: {} };
    const inputNodes = currentConfig.graph_structure.nodes.filter(n => n.nodeType === 'input');
    let hasInvalidInput = false;
    inputNodes.forEach(node => {
        const inputElement = document.getElementById(`input-${node.id}`);
        const inputContainer = inputElement ? inputElement.parentElement : null; // For styling parent
        if (inputElement) {
            const value = parseFloat(inputElement.value);
            if (isNaN(value) || value < 0 || value > 1) {
                 setStatusMessage('predict-status', `Invalid input for ${node.id}. Use 0-1.`, "error");
                 if(inputContainer) inputContainer.classList.add('invalid-input'); // Highlight invalid input
                 hasInvalidInput = true;
             } else {
                 if(inputContainer) inputContainer.classList.remove('invalid-input'); // Reset style
                 inputData.input_values[node.id] = value;
             }
        } else { inputData.input_values[node.id] = 0.5; }
    });
    if(hasInvalidInput) { enableUI(true); return; } // Stop if input is invalid, re-enable UI

    const payload = {
        input_values: inputData.input_values,
        graph_structure: currentConfig.graph_structure,
        config_id: currentConfig.id,
        config_name: currentConfig.name
    };

    setStatusMessage('predict-status', "Running prediction...", "loading");
    enableUI(false);
    clearLLMOutputs();

    try {
        const response = await fetch('/api/predict_openai_bn_single_call', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);
        updateNodeProbabilities(result.probabilities);
        displayLLMReasoning(result.llm_reasoning);
        displayLLMContext(result.llm_context);
        setStatusMessage('predict-status', "Prediction complete.", "success");
        logPrediction(inputData.input_values, result.probabilities); // Log successful prediction
    } catch (error) {
        console.error('Error fetching LLM predictions:', error);
        setStatusMessage('predict-status', `Prediction failed: ${error.message}`, "error");
        clearLLMOutputs();
    } finally {
        enableUI(true);
    }
}

// --- Node Appearance Update ---
function updateNodeProbabilities(probabilities) {
    // ... (Remains the same - updates node colors/labels) ...
     if (!cy) return;
    const useGradient = document.getElementById('gradient-toggle').checked;
    cy.nodes().forEach(node => {
        const nodeId = node.id(); const nodeType = node.data('nodeType'); let probState1 = null;
        if (probabilities && probabilities[nodeId] && probabilities[nodeId]["1"] !== undefined) { probState1 = probabilities[nodeId]["1"]; node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`); }
        else if (nodeType === 'input') { const inputElement = document.getElementById(`input-${nodeId}`); const currentVal = inputElement ? (parseFloat(inputElement.value) || 0.5) : 0.5; probState1 = currentVal; node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`); }
        else { node.data('currentProbLabel', '(N/A)'); }
        let baseBgColor = nodeType === 'input' ? '#add8e6' : '#f0e68c'; let textColor = '#333'; let finalBgColor = baseBgColor;
        if (probState1 !== null) { if (useGradient) { finalBgColor = `rgb(${Math.round(255 * (1 - probState1))}, ${Math.round(255 * probState1)}, 0)`; textColor = '#333'; } else { finalBgColor = '#4B0082'; textColor = '#FFFFFF'; } }
        else if (!useGradient && nodeType !== 'input') { finalBgColor = '#4B0082'; textColor = '#FFFFFF'; }
        node.style({ 'background-color': finalBgColor, 'color': textColor });
    });
     cy.style().update();
}

// --- LLM Output Display ---
function displayLLMReasoning(reasoningText) {
    // ... (Remains the same) ...
     const reasoningDiv = document.getElementById('llm-reasoning-content');
    reasoningDiv.textContent = reasoningText || "No reasoning provided by the LLM.";
}
function displayLLMContext(context) {
    // ... (Remains the same - populates details section) ...
     if (!context) return;
    const inputDiv = document.getElementById('input-context'); const structureDiv = document.getElementById('structure-context'); const descriptionsDiv = document.getElementById('node-descriptions-context');
    let iHtml='<h3>Input:</h3><ul>';(context.input_states||[]).forEach(i=>{iHtml+=`<li><b>${i.node}</b> (${i.description||'N/A'}): ${i.state} (p=${i.value.toFixed(2)})</li>`;});iHtml+='</ul>';inputDiv.innerHTML=iHtml||'<p>N/A</p>';
    let dHtml='<h3>Nodes:</h3><ul>';Object.entries(context.node_descriptions||{}).forEach(([n,d])=>{dHtml+=`<li><b>${n}</b>: ${d}</li>`;});dHtml+='</ul>';descriptionsDiv.innerHTML=dHtml||'<p>N/A</p>';
    let sHtml='<h3>Deps:</h3><pre>';Object.entries(context.node_dependencies||{}).forEach(([n,p])=>{sHtml+=`${n}<-${p.join(',')||'(In)'}\n`;});sHtml+='</pre>';structureDiv.innerHTML=sHtml||'<p>N/A</p>';
}
function clearLLMOutputs() {
    // ... (Remains the same) ...
     document.getElementById('llm-reasoning-content').textContent = 'Run prediction to see LLM reasoning.';
     document.getElementById('input-context').innerHTML = '<h3>Input States Provided:</h3><p>Run prediction to see context.</p>';
     document.getElementById('structure-context').innerHTML = '<h3>Node Dependencies (DAG):</h3><p>Run prediction to see context.</p>';
     document.getElementById('node-descriptions-context').innerHTML = '<h3>Node Descriptions:</h3><p>Run prediction to see context.</p>';
     // Also clear prediction status
     setStatusMessage('predict-status', "", "");
}

// --- Logging ---
function logPrediction(inputs, probabilities) {
    // ... (Session log logic remains the same) ...
    const timestamp = new Date().toISOString();
    const logEntry = { timestamp: timestamp, configId: currentConfig.id || "unsaved", configName: currentConfig.name, inputs: { ...inputs }, probabilities: {} };
    for (const nodeId in probabilities) { logEntry.probabilities[nodeId] = probabilities[nodeId]["1"]; }
    for (const inputId in inputs) { if (!(inputId in logEntry.probabilities)) { logEntry.probabilities[inputId] = inputs[inputId]; }}
    sessionLog.push(logEntry);
    document.getElementById('log-count').textContent = `Log entries this session: ${sessionLog.length}`;
    console.log("Prediction added to session log.");
}
function clearSessionLog() {
    // ... (Remains the same) ...
    sessionLog = [];
    document.getElementById('log-count').textContent = `Log entries this session: 0`;
    console.log("Session log cleared.");
}
function downloadSessionLog() {
    // ... (Remains the same) ...
     if (sessionLog.length === 0) { alert("No data logged this session."); return; }
    console.log("Generating session log CSV...");
    const allNodeIds = new Set(); sessionLog.forEach(e => { Object.keys(e.probabilities).forEach(n => allNodeIds.add(n)); });
    const sortedNodeIds = Array.from(allNodeIds).sort();
    const csvData = sessionLog.map(entry => { const row = { Timestamp: entry.timestamp, ConfigID: entry.configId, ConfigName: entry.configName }; sortedNodeIds.forEach(nodeId => { row[nodeId] = entry.probabilities[nodeId] !== undefined ? entry.probabilities[nodeId].toFixed(4) : ''; }); return row; });
    const csvHeaders = ["Timestamp", "ConfigID", "ConfigName", ...sortedNodeIds];
    const csvString = Papa.unparse({ fields: csvHeaders, data: csvData });
    triggerCsvDownload(csvString, `session_log_${(currentConfig?.name || 'unsaved').replace(/[^a-z0-9]/gi, '_')}`);
}
async function downloadAllLogs() {
    // ... (Remains the same - calls backend endpoint) ...
     if (!currentConfig || !currentConfig.id || currentConfig.id === 'default-config-001') { alert("Load a saved config to download its historical logs."); return; }
    const configId = currentConfig.id; const configName = currentConfig.name;
    setStatusMessage('predict-status', `Downloading all logs for ${configName}...`, "loading"); enableUI(false);
    try {
        const response = await fetch(`/api/download_log/${configId}`);
        if (!response.ok) { if (response.status === 404) { throw new Error(`No historical logs found for config '${configName}'.`); } else { const errorData = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` })); throw new Error(errorData.detail || `HTTP error ${response.status}`); } }
        const blob = await response.blob(); triggerCsvDownload(blob, `all_logs_${configName.replace(/[^a-z0-9]/gi, '_')}`); setStatusMessage('predict-status', `Downloaded all logs for ${configName}.`, "success");
    } catch (error) { console.error("Error downloading all logs:", error); setStatusMessage('predict-status', `Failed to download logs: ${error.message}`, "error"); }
    finally { enableUI(true); }
}
function triggerCsvDownload(csvDataOrBlob, baseFilename) {
    // ... (Remains the same) ...
     const blob = (csvDataOrBlob instanceof Blob) ? csvDataOrBlob : new Blob([csvDataOrBlob], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a"); const url = URL.createObjectURL(blob); link.setAttribute("href", url);
    const timestampStr = new Date().toISOString().replace(/[:.]/g, '-'); link.setAttribute("download", `${baseFilename}_${timestampStr}.csv`);
    link.style.visibility = 'hidden'; document.body.appendChild(link); link.click(); document.body.removeChild(link); URL.revokeObjectURL(url);
    console.log("CSV download triggered for:", baseFilename);
}

// --- Utility ---
function setStatusMessage(elementId, message, type) {
    // ... (Remains the same) ...
     const statusElement = document.getElementById(elementId);
    if (statusElement) { statusElement.textContent = message; statusElement.className = `status-message ${type}`; }
}

console.log("Initial script execution finished. Waiting for DOMContentLoaded.");
