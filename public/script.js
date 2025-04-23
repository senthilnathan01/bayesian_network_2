// public/script.js
let cy;
let currentConfig = null;
let defaultGraphStructure = null;
let sessionLog = [];

document.addEventListener('DOMContentLoaded', async () => {
    initializeCytoscape();
    initializeUI();

    setStatusMessage('config-status', "Loading default graph...", "loading");
    showSpinner(true);
    try {
        await loadDefaultConfig();
        if (defaultGraphStructure) {
            loadGraphData(defaultGraphStructure, true);
            addDefaultToDropdown();
            selectConfigInDropdown(defaultGraphStructure.id);
            setStatusMessage('config-status', "Default config loaded.", "success");
        } else {
            setStatusMessage('config-status', "Could not load default config.", "error");
            updateInputControls([]);
            document.getElementById('current-config-name').textContent = "None";
        }
    } catch (error) {
        console.error("Error loading default config:", error);
        setStatusMessage('config-status', `Failed to load default: ${error.message}`, "error");
        updateInputControls([]);
        document.getElementById('current-config-name').textContent = "None";
    }

    setStatusMessage('config-status', "Loading saved configurations...", "loading");
    try {
        await loadConfigList();
        setStatusMessage('config-status', currentConfig ? `Config '${currentConfig.name}' loaded.` : "Ready. Select or save a config.", "success");
    } catch (error) {
        console.error("Error loading saved configs:", error);
        setStatusMessage('config-status', `Failed to load saved configs: ${error.message}`, "error");
    } finally {
        showSpinner(false);
    }
});

function initializeCytoscape() {
    if (cy) {
        console.warn("Cytoscape already initialized.");
        return;
    }
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

    // Initialize context menus for graph editing, with fallback
    if (typeof cy.contextMenus === 'function') {
        cy.contextMenus({
            menuItems: [
                {
                    id: 'add-node',
                    content: 'Add Node',
                    selector: '*',
                    coreAsWell: true,
                    onClickFunction: (event) => {
                        const pos = event.position || event.cyPosition;
                        const nodeId = prompt("Enter node ID:");
                        if (!nodeId || cy.getElementById(nodeId).length > 0) {
                            alert("Invalid or duplicate node ID.");
                            return;
                        }
                        const fullName = prompt("Enter node name:") || nodeId;
                        const nodeType = prompt("Enter node type (input/hidden):") || "hidden";
                        if (nodeType !== "input" && nodeType !== "hidden") {
                            alert("Node type must be 'input' or 'hidden'.");
                            return;
                        }
                        cy.add({
                            group: 'nodes',
                            data: { id: nodeId, fullName, nodeType },
                            position: pos
                        });
                        runLayout();
                        updateInputControls(cy.nodes().map(n => n.data()));
                    }
                },
                {
                    id: 'remove-node',
                    content: 'Remove Node',
                    selector: 'node',
                    onClickFunction: (event) => {
                        const node = event.target;
                        if (confirm(`Remove node ${node.id()}?`)) {
                            node.remove();
                            runLayout();
                            updateInputControls(cy.nodes().map(n => n.data()));
                        }
                    }
                },
                {
                    id: 'add-edge',
                    content: 'Add Edge',
                    selector: 'node',
                    onClickFunction: (event) => {
                        const sourceNode = event.target;
                        const targetId = prompt(`Enter target node ID for edge from ${sourceNode.id()}:`);
                        if (!targetId || !cy.getElementById(targetId).length) {
                            alert("Invalid target node ID.");
                            return;
                        }
                        cy.add({
                            group: 'edges',
                            data: { source: sourceNode.id(), target: targetId }
                        });
                        runLayout();
                    }
                },
                {
                    id: 'remove-edge',
                    content: 'Remove Edge',
                    selector: 'edge',
                    onClickFunction: (event) => {
                        const edge = event.target;
                        if (confirm(`Remove edge ${edge.source().id()} -> ${edge.target().id()}?`)) {
                            edge.remove();
                            runLayout();
                        }
                    }
                }
            ]
        });
        console.log("Cytoscape context menus initialized.");
    } else {
        console.warn("Cytoscape context menus extension not available. Graph editing disabled.");
        const graphEditor = document.querySelector('.graph-editor p');
        if (graphEditor) {
            graphEditor.textContent = "Graph editing unavailable due to missing context menus extension. Load a saved configuration or use the default graph.";
            graphEditor.style.color = '#dc3545';
        }
    }
}

function nodeLabelFunc(node) {
    const id = node.data('id');
    const fullName = node.data('fullName') || id;
    const currentLabel = node.data('currentProbLabel');
    return `${id}: ${fullName}\n${currentLabel || '(N/A)'}`;
}

function initializeUI() {
    console.log("Initializing UI event listeners...");
    document.getElementById('save-config-button').addEventListener('click', saveConfiguration);
    document.getElementById('load-config-button').addEventListener('click', loadSelectedConfiguration);
    document.getElementById('set-default-button').addEventListener('click', setDefaultConfiguration);
    document.getElementById('delete-config-button').addEventListener('click', deleteSelectedConfiguration);
    document.getElementById('update-button').addEventListener('click', fetchAndUpdateLLM);
    document.getElementById('gradient-toggle').addEventListener('change', () => updateNodeProbabilities(null));
    document.getElementById('download-session-log-button').addEventListener('click', downloadSessionLog);
    document.getElementById('download-all-log-button').addEventListener('click', downloadAllLogs);
    console.log("UI Listeners attached.");
}

async function loadDefaultConfig() {
    console.log("Fetching default configuration...");
    await retryFetch(async () => {
        const response = await fetch('/api/configs/default');
        if (!response.ok) throw new Error(`HTTP error ${response.status}`);
        defaultGraphStructure = await response.json();
        console.log("Default config loaded.", defaultGraphStructure);
    }, 3);
}

function addDefaultToDropdown() {
    const select = document.getElementById('load-config-select');
    if (!defaultGraphStructure) return;
    if (Array.from(select.options).some(opt => opt.value === 'default-config-001')) return;
    const option = document.createElement('option');
    option.value = defaultGraphStructure.id;
    option.textContent = `${defaultGraphStructure.name} (Default)`;
    select.insertBefore(option, select.options[1] || null);
    console.log("Added default config to dropdown.");
}

function loadGraphData(configData, isDefault = false) {
    if (!cy) {
        console.error("Cytoscape not ready.");
        return;
    }
    if (!configData || !configData.graph_structure) {
        console.error("Invalid config data.");
        return;
    }
    console.log(`Loading graph: ${configData.name}`);
    const graphElements = configData.graph_structure.nodes.map(n => ({
        data: { id: n.id, fullName: n.fullName, nodeType: n.nodeType }
    })).concat(configData.graph_structure.edges.map(e => ({
        data: { source: e.source, target: e.target }
    })));
    cy.elements().remove();
    cy.add(graphElements);
    runLayout();
    currentConfig = { ...configData };
    document.getElementById('config-name').value = isDefault ? '' : currentConfig.name;
    document.getElementById('current-config-name').textContent = currentConfig.name;
    selectConfigInDropdown(currentConfig.id);
    updateInputControls(currentConfig.graph_structure.nodes);
    updateNodeProbabilities({});
    clearSessionLog();
    clearLLMOutputs();
}

async function saveConfiguration() {
    const configNameInput = document.getElementById('config-name');
    const configName = configNameInput.value.trim();
    if (!configName) {
        setStatusMessage('config-status', "Please enter a name.", "error");
        return;
    }
    if (!cy) {
        setStatusMessage('config-status', "Graph not initialized.", "error");
        return;
    }
    setStatusMessage('config-status', "Saving...", "loading");
    showSpinner(true);
    enableUI(false);
    const currentGraphStructure = {
        nodes: cy.nodes().map(node => ({
            id: node.id(),
            fullName: node.data('fullName'),
            nodeType: node.data('nodeType')
        })),
        edges: cy.edges().map(edge => ({
            source: edge.source().id(),
            target: edge.target().id()
        }))
    };
    await retryFetch(async () => {
        const response = await fetch('/api/configs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config_name: configName, graph_structure: currentGraphStructure })
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);
        currentConfig = { id: result.config_id, name: result.config_name, graph_structure: currentGraphStructure };
        document.getElementById('current-config-name').textContent = currentConfig.name;
        configNameInput.value = '';
        setStatusMessage('config-status', `Saved as '${result.config_name}'.`, "success");
        await loadConfigList();
        selectConfigInDropdown(result.config_id);
        clearSessionLog();
    }, 3, () => setStatusMessage('config-status', "Save failed. Retrying...", "error")).finally(() => {
        enableUI(true);
        showSpinner(false);
    });
}

async function loadConfigList() {
    console.log("Updating saved config list...");
    await retryFetch(async () => {
        const response = await fetch('/api/configs');
        if (!response.ok) throw new Error(`HTTP error ${response.status}`);
        const configs = await response.json();
        const select = document.getElementById('load-config-select');
        const currentSelection = select.value;
        Array.from(select.options).filter(opt => opt.value && opt.value !== 'default-config-001').forEach(opt => select.removeChild(opt));
        configs.forEach(config => {
            const option = document.createElement('option');
            option.value = config.id;
            option.textContent = config.name;
            select.appendChild(option);
        });
        select.value = currentSelection || (currentConfig ? currentConfig.id : (defaultGraphStructure ? defaultGraphStructure.id : ""));
        console.log("Saved config list updated.");
    }, 3, () => setStatusMessage('config-status', "Failed to update list. Retrying...", "error"));
}

async function loadSelectedConfiguration() {
    const select = document.getElementById('load-config-select');
    const configId = select.value;
    if (!configId) {
        setStatusMessage('config-status', "Select a config.", "error");
        return;
    }
    if (!cy) {
        setStatusMessage('config-status', "Graph not ready.", "error");
        return;
    }
    setStatusMessage('config-status', `Loading ${select.options[select.selectedIndex].text}...`, "loading");
    showSpinner(true);
    enableUI(false);
    await retryFetch(async () => {
        if (configId === 'default-config-001' && defaultGraphStructure) {
            loadGraphData(defaultGraphStructure, true);
            setStatusMessage('config-status', "Default config loaded.", "success");
            return;
        }
        const response = await fetch(`/api/configs/${configId}`);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` }));
            throw new Error(errorData.detail || `HTTP error ${response.status}`);
        }
        const configData = await response.json();
        loadGraphData(configData, false);
        setStatusMessage('config-status', `Config '${configData.name}' loaded.`, "success");
    }, 3, () => setStatusMessage('config-status', "Load failed. Retrying...", "error")).finally(() => {
        enableUI(true);
        showSpinner(false);
    });
}

async function setDefaultConfiguration() {
    const select = document.getElementById('load-config-select');
    const configId = select.value;
    const configName = select.options[select.selectedIndex].text;
    if (!configId || configId === 'default-config-001') {
        setStatusMessage('config-status', "Select a saved config to set as default.", "error");
        return;
    }
    if (!confirm(`Set "${configName}" as the default configuration?`)) return;
    setStatusMessage('config-status', `Setting ${configName} as default...`, "loading");
    showSpinner(true);
    enableUI(false);
    await retryFetch(async () => {
        const response = await fetch('/api/configs/set_default', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(configId)
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);
        setStatusMessage('config-status', `Set '${configName}' as default.`, "success");
        await loadDefaultConfig();
        loadGraphData(defaultGraphStructure, true);
    }, 3, () => setStatusMessage('config-status', "Failed to set default. Retrying...", "error")).finally(() => {
        enableUI(true);
        showSpinner(false);
    });
}

async function deleteSelectedConfiguration() {
    const select = document.getElementById('load-config-select');
    const configId = select.value;
    const configName = select.options[select.selectedIndex].text;
    if (!configId || configId === 'default-config-001') {
        setStatusMessage('config-status', "Select a saved config to delete.", "error");
        return;
    }
    if (!confirm(`Delete "${configName}" and its logs?`)) return;
    setStatusMessage('config-status', `Deleting ${configName}...`, "loading");
    showSpinner(true);
    enableUI(false);
    await retryFetch(async () => {
        const response = await fetch(`/api/configs/${configId}`, { method: 'DELETE' });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);
        if (currentConfig && currentConfig.id === configId) {
            if (defaultGraphStructure) {
                loadGraphData(defaultGraphStructure, true);
                setStatusMessage('config-status', `Deleted '${configName}'. Default config loaded.`, "success");
            } else {
                cy.elements().remove();
                currentConfig = null;
                updateInputControls([]);
                clearSessionLog();
                clearLLMOutputs();
                document.getElementById('current-config-name').textContent = "None";
                setStatusMessage('config-status', `Deleted '${configName}'. Load another config.`, "success");
            }
        } else {
            setStatusMessage('config-status', `Deleted '${configName}'.`, "success");
        }
        await loadConfigList();
    }, 3, () => setStatusMessage('config-status', "Delete failed. Retrying...", "error")).finally(() => {
        enableUI(true);
        showSpinner(false);
    });
}

function selectConfigInDropdown(configId) {
    const select = document.getElementById('load-config-select');
    select.value = configId;
}

function enableUI(enable) {
    const buttons = document.querySelectorAll('button');
    const inputs = document.querySelectorAll('input, select');
    buttons.forEach(btn => btn.disabled = !enable);
    inputs.forEach(inp => inp.disabled = !enable);
    document.getElementById('gradient-toggle').disabled = false;
    document.body.style.cursor = enable ? 'default' : 'wait';
}

function showSpinner(show) {
    document.getElementById('loading-spinner').classList.toggle('hidden', !show);
}

function runLayout() {
    if (!cy) return;
    let layoutName = 'cola';
    try {
        if (typeof cy.layout({ name: 'cola' }).run !== 'function') throw new Error();
    } catch (e) {
        console.warn("Cola failed, trying dagre...");
        layoutName = 'dagre';
        try {
            if (typeof cy.layout({ name: 'dagre' }).run !== 'function') throw new Error();
        } catch (e) {
            console.error("Layouts failed!");
            layoutName = 'grid';
        }
    }
    const options = layoutName === 'cola' ? { name: 'cola', animate: true, nodeSpacing: 50, edgeLength: 180, padding: 30 }
                  : layoutName === 'dagre' ? { name: 'dagre', rankDir: 'TB', spacingFactor: 1.2, animate: true, padding: 30 }
                  : { name: 'grid' };
    cy.layout(options).run();
}

function updateInputControls(nodes) {
    const container = document.getElementById('input-controls-container');
    container.innerHTML = '';
    const inputNodes = nodes.filter(n => n.nodeType === 'input');
    if (inputNodes.length === 0) {
        container.innerHTML = '<p>No input nodes defined.</p>';
        return;
    }
    inputNodes.forEach(node => {
        const div = document.createElement('div');
        const label = document.createElement('label');
        label.htmlFor = `input-${node.id}`;
        label.textContent = `${node.id} (${node.fullName}):`;
        const input = document.createElement('input');
        input.type = 'number';
        input.id = `input-${node.id}`;
        input.name = node.id;
        input.min = "0";
        input.max = "1";
        input.step = "0.01";
        input.value = "0.5";
        div.appendChild(label);
        div.appendChild(input);
        container.appendChild(div);
    });
}

async function fetchAndUpdateLLM() {
    if (!currentConfig || !currentConfig.graph_structure || currentConfig.graph_structure.nodes.length === 0) {
        alert("Load a config first.");
        return;
    }
    if (!cy) {
        alert("Graph not ready.");
        return;
    }
    setStatusMessage('predict-status', "Gathering inputs...", "loading");
    showSpinner(true);
    const inputData = { input_values: {} };
    const inputNodes = currentConfig.graph_structure.nodes.filter(n => n.nodeType === 'input');
    let hasInvalidInput = false;
    inputNodes.forEach(node => {
        const inputElement = document.getElementById(`input-${node.id}`);
        const inputContainer = inputElement ? inputElement.parentElement : null;
        if (inputElement) {
            const value = parseFloat(inputElement.value);
            if (isNaN(value) || value < 0 || value > 1) {
                setStatusMessage('predict-status', `Invalid input for ${node.id}. Use 0-1.`, "error");
                if (inputContainer) inputContainer.classList.add('invalid-input');
                hasInvalidInput = true;
            } else {
                if (inputContainer) inputContainer.classList.remove('invalid-input');
                inputData.input_values[node.id] = value;
            }
        } else {
            inputData.input_values[node.id] = 0.5;
        }
    });
    if (hasInvalidInput) {
        enableUI(true);
        showSpinner(false);
        return;
    }
    const payload = {
        input_values: inputData.input_values,
        graph_structure: currentConfig.graph_structure,
        config_id: currentConfig.id,
        config_name: currentConfig.name
    };
    setStatusMessage('predict-status', "Running prediction...", "loading");
    enableUI(false);
    clearLLMOutputs();
    await retryFetch(async () => {
        const response = await fetch('/api/predict_openai_bn_single_call', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);
        updateNodeProbabilities(result.probabilities);
        displayLLMReasoning(result.llm_reasoning);
        displayLLMContext(result.llm_context);
        setStatusMessage('predict-status', "Prediction complete.", "success");
        logPrediction(inputData.input_values, result.probabilities);
    }, 3, () => setStatusMessage('predict-status', "Prediction failed. Retrying...", "error")).finally(() => {
        enableUI(true);
        showSpinner(false);
    });
}

function updateNodeProbabilities(probabilities) {
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
            probState1 = currentVal;
            node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`);
        } else {
            node.data('currentProbLabel', '(N/A)');
        }
        let baseBgColor = nodeType === 'input' ? '#add8e6' : '#f0e68c';
        let textColor = '#333';
        let finalBgColor = baseBgColor;
        if (probState1 !== null) {
            if (useGradient) {
                finalBgColor = `rgb(${Math.round(255 * (1 - probState1))}, ${Math.round(255 * probState1)}, 0)`;
                textColor = '#333';
            } else {
                finalBgColor = '#4B0082';
                textColor = '#FFFFFF';
            }
        } else if (!useGradient && nodeType !== 'input') {
            finalBgColor = '#4B0082';
            textColor = '#FFFFFF';
        }
        node.style({ 'background-color': finalBgColor, 'color': textColor });
    });
    cy.style().update();
}

function displayLLMReasoning(reasoningText) {
    const reasoningDiv = document.getElementById('llm-reasoning-content');
    reasoningDiv.textContent = reasoningText || "No reasoning provided by the LLM.";
}

function displayLLMContext(context) {
    if (!context) return;
    const inputDiv = document.getElementById('input-context');
    const structureDiv = document.getElementById('structure-context');
    const descriptionsDiv = document.getElementById('node-descriptions-context');

    let iHtml = '<ul>';
    (context.input_states || []).forEach(state => {
        iHtml += `<li>${state.node} (${state.description}): ${state.state} (P(1)=${state.value.toFixed(2)})</li>`;
    });
    iHtml += '</ul>';
    inputDiv.innerHTML = context.input_states && context.input_states.length > 0 ? iHtml : '<p>No input states provided.</p>';

    let sHtml = '<ul>';
    Object.entries(context.node_dependencies || {}).forEach(([node, parents]) => {
        sHtml += `<li>${node}: ${parents.length > 0 ? parents.join(', ') : 'None'}</li>`;
    });
    sHtml += '</ul>';
    structureDiv.innerHTML = context.node_dependencies ? sHtml : '<p>No dependencies provided.</p>';

    let dHtml = '<ul>';
    Object.entries(context.node_descriptions || {}).forEach(([node, desc]) => {
        dHtml += `<li>${node}: ${desc}</li>`;
    });
    dHtml += '</ul>';
    descriptionsDiv.innerHTML = context.node_descriptions ? dHtml : '<p>No descriptions provided.</p>';
}

function setStatusMessage(elementId, message, statusType) {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.classList.remove('success', 'error', 'loading');
    if (statusType) {
        element.classList.add(statusType);
    }
}

async function retryFetch(fetchFn, maxRetries, onRetry) {
    let lastError;
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            await fetchFn();
            return;
        } catch (error) {
            lastError = error;
            console.warn(`Attempt ${attempt} failed: ${error.message}`);
            if (attempt < maxRetries && onRetry) {
                onRetry();
                await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
            }
        }
    }
    throw lastError;
}

function logPrediction(inputs, probabilities) {
    const logEntry = {
        timestamp: new Date().toISOString(),
        configId: currentConfig ? currentConfig.id : "unknown",
        configName: currentConfig ? currentConfig.name : "Unknown",
        inputs: { ...inputs },
        probabilities: { ...probabilities }
    };
    sessionLog.push(logEntry);
    document.getElementById('log-count').textContent = `Log entries this session: ${sessionLog.length}`;
}

function clearSessionLog() {
    sessionLog = [];
    document.getElementById('log-count').textContent = `Log entries this session: 0`;
}

function clearLLMOutputs() {
    document.getElementById('llm-reasoning-content').textContent = "Run prediction to see LLM reasonings.";
    document.getElementById('input-context').innerHTML = "<p>Run prediction to see context.</p>";
    document.getElementById('structure-context').innerHTML = "<p>Run prediction to see context.</p>";
    document.getElementById('node-descriptions-context').innerHTML = "<p>Run prediction to see context.</p>";
}

function downloadSessionLog() {
    if (sessionLog.length === 0) {
        alert("No session logs to download.");
        return;
    }
    const headers = ['Timestamp', 'ConfigID', 'ConfigName', 'NodeID', 'ProbabilityP1'];
    const rows = [];
    sessionLog.forEach(log => {
        Object.entries(log.probabilities).forEach(([nodeId, probs]) => {
            rows.push([log.timestamp, log.configId, log.configName, nodeId, probs["1"].toFixed(4)]);
        });
    });
    const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `session_log_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

async function downloadAllLogs() {
    if (!currentConfig || !currentConfig.id || currentConfig.id === "unknown" || currentConfig.id === "default-config-001") {
        alert("Select a saved config to download logs.");
        return;
    }
    setStatusMessage('predict-status', "Downloading logs...", "loading");
    showSpinner(true);
    enableUI(false);
    await retryFetch(async () => {
        const response = await fetch(`/api/download_log/${currentConfig.id}`);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` }));
            throw new Error(errorData.detail || `HTTP error ${response.status}`);
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `log_${currentConfig.id.replace('bn_config:', '')}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        setStatusMessage('predict-status', "Log downloaded.", "success");
    }, 3, () => setStatusMessage('predict-status', "Download failed. Retrying...", "error")).finally(() => {
        enableUI(true);
        showSpinner(false);
    });
}
