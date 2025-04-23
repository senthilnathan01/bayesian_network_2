// --- public/script.js ---

// Global variables
let cy;
let currentConfig = null; // Start with no config loaded
let sessionLog = [];
let defaultGraphStructure = null; // To store the default structure fetched from backend

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    initializeCytoscape(); // Initialize cy first
    initializeUI();
    setStatusMessage('config-status', "Loading initial data...", "loading");
    try {
        await Promise.all([
            loadDefaultConfig(), // Fetch default config structure
            loadConfigList()     // Fetch saved configurations
        ]);
        // Load default graph *after* fetching its structure
        if (defaultGraphStructure && !currentConfig) {
            loadGraphData(defaultGraphStructure, true); // Load default if nothing else loaded
            setStatusMessage('config-status', "Default config loaded.", "success");
        } else if (!currentConfig) {
             setStatusMessage('config-status', "Ready. Select or save a configuration.", "success");
        }
    } catch (error) {
        console.error("Initialization error:", error);
        setStatusMessage('config-status', `Initialization failed: ${error.message}`, "error");
    }
});

function initializeCytoscape() {
    if (cy) { console.warn("Cytoscape already initialized."); return; }
    console.log("Initializing Cytoscape...");
    cy = cytoscape({
        container: document.getElementById('cy'),
        elements: [],
        style: [ // Base styles (unchanged from previous)
            { selector: 'node', style: { 'background-color': '#ccc', 'label': nodeLabelFunc, 'width': 120, 'height': 120, 'shape': 'ellipse', 'text-valign': 'center', 'text-halign': 'center', 'font-size': '10px', 'font-weight': '150', 'text-wrap': 'wrap', 'text-max-width': 110, 'text-outline-color': '#fff', 'text-outline-width': 1, 'color': '#333', 'transition-property': 'background-color, color', 'transition-duration': '0.5s' } },
            { selector: 'node[nodeType="input"]', style: { 'shape': 'rectangle', 'width': 130, 'height': 70 } },
            { selector: 'node[nodeType="hidden"]', style: { 'shape': 'ellipse' } }, // Hidden nodes are ellipses
            { selector: 'edge', style: { 'width': 2, 'line-color': '#666', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#666', 'curve-style': 'bezier' } }
        ],
        layout: { name: 'cola', animate: true, nodeSpacing: 50, edgeLength: 180, padding: 30 }
    });
    console.log("Cytoscape initialized.");
}

// Separate label function for clarity
function nodeLabelFunc(node) {
    const id = node.data('id');
    const fullName = node.data('fullName') || id;
    const currentLabel = node.data('currentProbLabel');
    return `${id}: ${fullName}\n${currentLabel || '(N/A)'}`;
}


function initializeUI() {
    console.log("Initializing UI event listeners...");
    // Config Buttons
    document.getElementById('save-config-button').addEventListener('click', saveConfiguration);
    document.getElementById('load-config-button').addEventListener('click', loadSelectedConfiguration);
    document.getElementById('delete-config-button').addEventListener('click', deleteSelectedConfiguration); // Added delete listener

    // Prediction Buttons
    document.getElementById('update-button').addEventListener('click', fetchAndUpdateLLM);
    document.getElementById('gradient-toggle').addEventListener('change', () => updateNodeProbabilities(null));

    // Logging Buttons
    document.getElementById('download-session-log-button').addEventListener('click', downloadSessionLog);
    document.getElementById('download-all-log-button').addEventListener('click', downloadAllLogs); // Added all logs listener
    console.log("UI Listeners attached.");
}

// --- Configuration Management ---

async function loadDefaultConfig() {
    console.log("Fetching default configuration...");
    try {
        const response = await fetch('/api/configs/default');
        if (!response.ok) throw new Error(`HTTP error ${response.status}`);
        defaultGraphStructure = await response.json();
        console.log("Default config structure loaded.", defaultGraphStructure);

        // Add default config to the dropdown if not already present
        const select = document.getElementById('load-config-select');
        let defaultOptionExists = false;
        for(let i=0; i < select.options.length; i++){
            if (select.options[i].value === 'default-config-001') {
                defaultOptionExists = true;
                break;
            }
        }
        if (!defaultOptionExists && defaultGraphStructure) {
            const option = document.createElement('option');
            option.value = defaultGraphStructure.id; // Use the ID from the structure
            option.textContent = `${defaultGraphStructure.name} (Default)`;
            // Add it after the placeholder "-- Select a config --"
             if (select.options.length > 0) {
                 select.insertBefore(option, select.options[1]);
             } else {
                 select.appendChild(option);
             }
        }

    } catch (error) {
        console.error('Error fetching default configuration:', error);
        setStatusMessage('config-status', `Error loading default config: ${error.message}`, "error");
        // Allow app to continue without default if fetch fails
        defaultGraphStructure = null;
    }
}

function loadGraphData(configData, isDefault = false) {
     if (!cy) { console.error("Cannot load graph, Cytoscape not ready."); return; }
     if (!configData || !configData.graph_structure) { console.error("Invalid config data provided to loadGraphData."); return; }

     console.log(`Loading graph data for: ${configData.name}`);
     const graphElements = configData.graph_structure.nodes.map(n => ({ data: { id: n.id, fullName: n.fullName, nodeType: n.nodeType } }))
         .concat(configData.graph_structure.edges.map(e => ({ data: { source: e.source, target: e.target } })));

     cy.elements().remove();
     cy.add(graphElements);
     runLayout();

     currentConfig = configData; // Update global state
     document.getElementById('config-name').value = isDefault ? '' : currentConfig.name; // Don't prefill name for default
     document.getElementById('current-config-name').textContent = currentConfig.name;
     updateInputControls(currentConfig.graph_structure.nodes);
     updateNodeProbabilities({}); // Reset probabilities display
     clearSessionLog(); // Loading resets the session log
     clearLLMOutputs(); // Clear reasoning and context displays
     setStatusMessage('config-status', `Config '${currentConfig.name}' loaded.`, "success");
}


async function saveConfiguration() {
    const configNameInput = document.getElementById('config-name');
    const configName = configNameInput.value.trim();
    if (!configName) { setStatusMessage('config-status', "Please enter a name.", "error"); return; }
    if (!cy) { setStatusMessage('config-status', "Graph not initialized.", "error"); return; }

    setStatusMessage('config-status', "Saving...", "loading");
    enableUI(false); // Disable UI during save

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

        currentConfig = { id: result.config_id, name: result.config_name, graph_structure: currentGraphStructure };
        document.getElementById('current-config-name').textContent = currentConfig.name;
        configNameInput.value = ''; // Clear input after save
        setStatusMessage('config-status', `Saved as '${result.config_name}'.`, "success");
        await loadConfigList(); // Refresh dropdown with new config
        selectConfigInDropdown(result.config_id); // Select the newly saved config
        clearSessionLog();

    } catch (error) {
        console.error('Error saving configuration:', error);
        setStatusMessage('config-status', `Save failed: ${error.message}`, "error");
    } finally {
        enableUI(true); // Re-enable UI
    }
}

async function loadConfigList() {
    console.log("Updating configuration list...");
    setStatusMessage('config-status', "Updating config list...", "loading"); // Use config status
    try {
        const response = await fetch('/api/configs');
        if (!response.ok) throw new Error(`HTTP error ${response.status}`);
        const configs = await response.json();

        const select = document.getElementById('load-config-select');
        const currentSelection = select.value; // Preserve selection if possible
        // Clear existing options except placeholder and default
        while (select.options.length > 1 && select.options[1].value !== 'default-config-001') {
            select.remove(1); // Remove starting from index 1
        }
         while (select.options.length > 2) { // Remove any after default if it exists
            select.remove(2);
        }


        // Add fetched configs
        configs.forEach(config => {
            // Don't re-add default if it came from backend list
            if (config.id !== 'default-config-001') {
                const option = document.createElement('option');
                option.value = config.id;
                option.textContent = config.name;
                select.appendChild(option);
            }
        });
        select.value = currentSelection; // Try to restore selection
         if (select.selectedIndex === -1) select.selectedIndex = 0; // Fallback to placeholder
        setStatusMessage('config-status', "Config list updated.", "success");
    } catch (error) {
        console.error('Error loading configuration list:', error);
        setStatusMessage('config-status', `Failed to update config list: ${error.message}`, "error");
    }
}

async function loadSelectedConfiguration() {
    const select = document.getElementById('load-config-select');
    const configId = select.value;
    if (!configId) { setStatusMessage('config-status', "Select a config.", "error"); return; }
    if (!cy) { setStatusMessage('config-status', "Graph not ready.", "error"); return; }

    // Handle loading default config
    if (configId === 'default-config-001') {
        if (defaultGraphStructure) {
            loadGraphData(defaultGraphStructure, true);
            setStatusMessage('config-status', "Default config loaded.", "success");
        } else {
            setStatusMessage('config-status', "Default config data not available.", "error");
        }
        return;
    }

    setStatusMessage('config-status', `Loading ${select.options[select.selectedIndex].text}...`, "loading");
    enableUI(false);

    try {
        const response = await fetch(`/api/configs/${configId}`);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` }));
            throw new Error(errorData.detail || `HTTP error ${response.status}`);
        }
        const configData = await response.json();
        loadGraphData(configData); // Load the fetched data
    } catch (error) {
        console.error('Error loading selected configuration:', error);
        setStatusMessage('config-status', `Load failed: ${error.message}`, "error");
    } finally {
        enableUI(true);
    }
}


async function deleteSelectedConfiguration() {
    const select = document.getElementById('load-config-select');
    const configId = select.value;
    const configName = select.options[select.selectedIndex].text;

    if (!configId || configId === 'default-config-001') {
        setStatusMessage('config-status', "Select a saved config to delete.", "error");
        return;
    }

    if (!confirm(`Are you sure you want to delete configuration "${configName}"?\nThis action cannot be undone and will also delete associated logs.`)) {
        return;
    }

    setStatusMessage('config-status', `Deleting ${configName}...`, "loading");
    enableUI(false);

    try {
        const response = await fetch(`/api/configs/${configId}`, { method: 'DELETE' });
        const result = await response.json(); // Expecting JSON response on success/error
        if (!response.ok) {
            throw new Error(result.detail || `HTTP error ${response.status}`);
        }

        setStatusMessage('config-status', `Deleted '${configName}'.`, "success");
        // If the deleted config was loaded, load the default
        if (currentConfig && currentConfig.id === configId) {
             if(defaultGraphStructure) loadGraphData(defaultGraphStructure, true);
             else { // If somehow default isn't available, clear graph
                 cy.elements().remove();
                 currentConfig = null;
                 updateInputControls([]);
                 clearSessionLog();
                 clearLLMOutputs();
                 document.getElementById('current-config-name').textContent = "None";
             }
        }
        await loadConfigList(); // Refresh dropdown

    } catch (error) {
        console.error('Error deleting configuration:', error);
        setStatusMessage('config-status', `Delete failed: ${error.message}`, "error");
    } finally {
        enableUI(true);
    }
}

function selectConfigInDropdown(configId) {
    const select = document.getElementById('load-config-select');
    select.value = configId;
}

// --- UI Enable/Disable ---
function enableUI(enable) {
    const buttons = document.querySelectorAll('button');
    const inputs = document.querySelectorAll('input, select');
    buttons.forEach(btn => btn.disabled = !enable);
    inputs.forEach(inp => inp.disabled = !enable);
     // Keep gradient toggle always enabled
     document.getElementById('gradient-toggle').disabled = false;
    document.body.style.cursor = enable ? 'default' : 'wait';
}

// --- Layout ---
function runLayout() {
    // (Unchanged - uses cola with dagre fallback)
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
    // (Unchanged from previous)
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
    if (!currentConfig || !currentConfig.graph_structure || currentConfig.graph_structure.nodes.length === 0) { alert("Load a config first."); return; }
    if (!cy) { alert("Graph not ready."); return; }

    setStatusMessage('predict-status', "Gathering inputs...", "loading");
    const inputData = { input_values: {} };
    const inputNodes = currentConfig.graph_structure.nodes.filter(n => n.nodeType === 'input');
    let hasInvalidInput = false;
    inputNodes.forEach(node => {
        const inputElement = document.getElementById(`input-${node.id}`);
        if (inputElement) {
            const value = parseFloat(inputElement.value);
            if (isNaN(value) || value < 0 || value > 1) {
                 setStatusMessage('predict-status', `Invalid input for ${node.id}. Use 0-1.`, "error");
                 inputElement.style.border = '2px solid red'; // Highlight invalid input
                 hasInvalidInput = true;
             } else {
                 inputElement.style.border = ''; // Reset border
                 inputData.input_values[node.id] = value;
             }
        } else { inputData.input_values[node.id] = 0.5; } // Default if missing
    });
    if(hasInvalidInput) return; // Stop if input is invalid

    const payload = {
        input_values: inputData.input_values,
        graph_structure: currentConfig.graph_structure,
        config_id: currentConfig.id, // Send config ID for logging
        config_name: currentConfig.name // Send config name for logging
    };

    setStatusMessage('predict-status', "Running prediction...", "loading");
    enableUI(false);
    clearLLMOutputs();

    try {
        const response = await fetch('/api/predict_openai_bn_single_call', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json(); // Always expect JSON, even for errors now
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);

        updateNodeProbabilities(result.probabilities);
        displayLLMReasoning(result.llm_reasoning); // Display reasoning
        displayLLMContext(result.llm_context); // Display context in details section
        setStatusMessage('predict-status', "Prediction complete.", "success");

        // Log after successful update and display
        logPrediction(inputData.input_values, result.probabilities);

    } catch (error) {
        console.error('Error fetching LLM predictions:', error);
        setStatusMessage('predict-status', `Prediction failed: ${error.message}`, "error");
        clearLLMOutputs(); // Clear reasoning on error too
    } finally {
        enableUI(true);
    }
}

// --- Node Appearance Update ---
function updateNodeProbabilities(probabilities) {
    // (Logic largely unchanged, uses nodeType now, handles inputs slightly differently)
    if (!cy) return;
    const useGradient = document.getElementById('gradient-toggle').checked;

    cy.nodes().forEach(node => {
        const nodeId = node.id();
        const nodeType = node.data('nodeType');
        let probState1 = null;

        if (probabilities && probabilities[nodeId] && probabilities[nodeId]["1"] !== undefined) {
            probState1 = probabilities[nodeId]["1"];
            node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`);
        } else if (nodeType === 'input') {
            const inputElement = document.getElementById(`input-${nodeId}`);
            const currentVal = inputElement ? (parseFloat(inputElement.value) || 0.5) : 0.5;
            probState1 = currentVal; // Reflect input value directly
            node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`);
        } else {
            node.data('currentProbLabel', '(N/A)');
        }

        // Determine base colors (inputs blue, hidden khaki)
        let baseBgColor = nodeType === 'input' ? '#add8e6' : '#f0e68c';
        let textColor = '#333';

        // Apply special colors if probability exists
        let finalBgColor = baseBgColor;
        if (probState1 !== null) {
            if (useGradient) {
                finalBgColor = `rgb(${Math.round(255 * (1 - probState1))}, ${Math.round(255 * probState1)}, 0)`;
                textColor = '#333';
            } else {
                finalBgColor = '#4B0082'; textColor = '#FFFFFF';
            }
        } else if (!useGradient && nodeType !== 'input') { // Apply clean color to hidden nodes even w/o prob if clean mode on
             finalBgColor = '#4B0082'; textColor = '#FFFFFF';
        }

        node.style({ 'background-color': finalBgColor, 'color': textColor });
    });
    // Re-render labels after data change (important!)
     cy.style().update();
}


// --- LLM Output Display ---

function displayLLMReasoning(reasoningText) {
    const reasoningDiv = document.getElementById('llm-reasoning-content');
    reasoningDiv.textContent = reasoningText || "No reasoning provided by the LLM.";
}

function displayLLMContext(context) {
    // (Unchanged - populates the <details> section)
    if (!context) return;
    const inputDiv = document.getElementById('input-context');
    const structureDiv = document.getElementById('structure-context');
    const descriptionsDiv = document.getElementById('node-descriptions-context');
    let inputHtml = '<h3>Input States Provided:</h3><ul>'; (context.input_states || []).forEach(i => { inputHtml += `<li><strong>${i.node}</strong> (${i.description || 'N/A'}): ${i.state} (prob ${i.value.toFixed(2)})</li>`; }); inputHtml += '</ul>'; inputDiv.innerHTML = inputHtml || '<p>N/A</p>';
    let descHtml = '<h3>Node Descriptions:</h3><ul>'; Object.entries(context.node_descriptions || {}).forEach(([n, d]) => { descHtml += `<li><strong>${n}</strong>: ${d}</li>`; }); descHtml += '</ul>'; descriptionsDiv.innerHTML = descHtml || '<p>N/A</p>';
    let structureHtml = '<h3>Node Dependencies (DAG):</h3><pre>'; Object.entries(context.node_dependencies || {}).forEach(([n, p]) => { structureHtml += `${n} <- ${p.join(', ') || '(Input)'}\n`; }); structureHtml += '</pre>'; structureDiv.innerHTML = structureHtml || '<p>N/A</p>';
}

function clearLLMOutputs() {
     document.getElementById('llm-reasoning-content').textContent = 'Run prediction to see LLM reasoning.';
     document.getElementById('input-context').innerHTML = '<h3>Input States Provided:</h3><p>Run prediction to see context.</p>';
     document.getElementById('structure-context').innerHTML = '<h3>Node Dependencies (DAG):</h3><p>Run prediction to see context.</p>';
     document.getElementById('node-descriptions-context').innerHTML = '<h3>Node Descriptions:</h3><p>Run prediction to see context.</p>';
      setStatusMessage('predict-status', "", ""); // Clear predict status
}

// --- Logging ---

function logPrediction(inputs, probabilities) {
    // Log data sent to backend /api/log endpoint after successful prediction
    // Frontend only keeps session log for download
    const timestamp = new Date().toISOString();
    const logEntry = {
        timestamp: timestamp,
        configId: currentConfig.id || "unsaved",
        configName: currentConfig.name,
        inputs: { ...inputs },
        probabilities: {} // Store P(1)
    };
    for (const nodeId in probabilities) { logEntry.probabilities[nodeId] = probabilities[nodeId]["1"]; }
    for (const inputId in inputs) { if (!(inputId in logEntry.probabilities)) { logEntry.probabilities[inputId] = inputs[inputId]; }} // Also add inputs

    sessionLog.push(logEntry);
    document.getElementById('log-count').textContent = `Log entries this session: ${sessionLog.length}`;
    console.log("Prediction added to session log.");
     // No need to call backend /api/log here, it's handled internally by predict endpoint now
}

function clearSessionLog() {
    sessionLog = [];
    document.getElementById('log-count').textContent = `Log entries this session: 0`;
    console.log("Session log cleared.");
}

function downloadSessionLog() {
    // (Unchanged from previous - uses PapaParse)
    if (sessionLog.length === 0) { alert("No data logged this session."); return; }
    console.log("Generating session log CSV...");
    const allNodeIds = new Set(); sessionLog.forEach(e => { Object.keys(e.probabilities).forEach(n => allNodeIds.add(n)); });
    const sortedNodeIds = Array.from(allNodeIds).sort();
    const csvData = sessionLog.map(entry => {
        const row = { Timestamp: entry.timestamp, ConfigID: entry.configId, ConfigName: entry.configName };
        sortedNodeIds.forEach(nodeId => { row[nodeId] = entry.probabilities[nodeId] !== undefined ? entry.probabilities[nodeId].toFixed(4) : ''; });
        return row;
    });
    const csvHeaders = ["Timestamp", "ConfigID", "ConfigName", ...sortedNodeIds];
    const csvString = Papa.unparse({ fields: csvHeaders, data: csvData });
    triggerCsvDownload(csvString, `session_log_${(currentConfig?.name || 'unsaved').replace(/[^a-z0-9]/gi, '_')}`);
}

async function downloadAllLogs() {
     if (!currentConfig || !currentConfig.id || currentConfig.id === 'default-config-001') {
        alert("Please load a saved configuration to download its historical logs.");
        return;
    }
    const configId = currentConfig.id;
    const configName = currentConfig.name;
    console.log(`Requesting all logs for config: ${configName} (${configId})`);
    setStatusMessage('predict-status', `Downloading all logs for ${configName}...`, "loading"); // Use predict status temporarily
    enableUI(false);

    try {
        const response = await fetch(`/api/download_log/${configId}`);
        if (!response.ok) {
             if (response.status === 404) {
                 throw new Error(`No historical logs found for config '${configName}'.`);
             } else {
                const errorData = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` }));
                throw new Error(errorData.detail || `HTTP error ${response.status}`);
             }
        }

        // Handle the file stream
        const blob = await response.blob();
        triggerCsvDownload(blob, `all_logs_${configName.replace(/[^a-z0-9]/gi, '_')}`);
        setStatusMessage('predict-status', `Downloaded all logs for ${configName}.`, "success");

    } catch (error) {
        console.error("Error downloading all logs:", error);
        setStatusMessage('predict-status', `Failed to download logs: ${error.message}`, "error");
    } finally {
        enableUI(true);
    }
}

function triggerCsvDownload(csvDataOrBlob, baseFilename) {
    const blob = (csvDataOrBlob instanceof Blob) ? csvDataOrBlob : new Blob([csvDataOrBlob], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    const timestampStr = new Date().toISOString().replace(/[:.]/g, '-');
    link.setAttribute("download", `${baseFilename}_${timestampStr}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url); // Clean up
    console.log("CSV download triggered for:", baseFilename);
}

// --- Utility ---
function setStatusMessage(elementId, message, type) { // type: 'success', 'error', 'loading', or ''
    const statusElement = document.getElementById(elementId);
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = `status-message ${type}`; // Use classes for styling
    }
}

console.log("Initial script execution finished. Waiting for DOMContentLoaded.");
