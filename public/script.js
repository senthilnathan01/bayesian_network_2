// --- public/script.js ---

// Global variables
let cy; // Cytoscape instance
let currentConfig = { // Holds the currently loaded configuration
    id: null,
    name: "Unsaved Configuration",
    graph_structure: { nodes: [], edges: [] }
};
let sessionLog = []; // Array to hold log entries for the current session

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    initializeCytoscape();
    initializeUI();
    loadConfigList(); // Fetch saved configurations on load
});

function initializeCytoscape() {
    console.log("Initializing Cytoscape...");
    cy = cytoscape({
        container: document.getElementById('cy'),
        elements: [], // Start empty, load config later
        style: [ // Base styles, colors updated dynamically
            {
                selector: 'node',
                style: {
                    'background-color': '#ccc', // Default, overridden by type/probability
                    'label': function(node) {
                        const id = node.data('id');
                        const fullName = node.data('fullName') || id;
                        const currentLabel = node.data('currentProbLabel');
                        return `${id}: ${fullName}\n${currentLabel || '(N/A)'}`;
                    },
                    'width': 120, 'height': 120,
                    'shape': 'ellipse', // Default shape
                    'text-valign': 'center', 'text-halign': 'center',
                    'font-size': '10px', 'font-weight': '150',
                    'text-wrap': 'wrap', 'text-max-width': 110,
                    'text-outline-color': '#fff', 'text-outline-width': 1,
                    'color': '#333', // Default text color
                    'transition-property': 'background-color, color',
                    'transition-duration': '0.5s'
                }
            },
            { // Style for INPUT nodes
                selector: 'node[nodeType="input"]',
                style: { 'shape': 'rectangle', 'width': 130, 'height': 70 }
            },
             { // Style for HIDDEN nodes (and formerly OUTPUT nodes)
                 selector: 'node[nodeType="hidden"]',
                 style: { 'shape': 'ellipse' } // Keep default shape
             },
            { // Edge style
                selector: 'edge',
                style: {
                    'width': 2, 'line-color': '#666',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': '#666',
                    'curve-style': 'bezier'
                }
            }
        ],
        layout: { name: 'cola', // Default layout
             animate: true, nodeSpacing: 50, edgeLength: 180, padding: 30
         }
    });
     console.log("Cytoscape initialized.");

    // Placeholder: Add event listeners for graph editing here later
    // cy.on('tap', 'node', (evt) => { /* ... */ });
}

function initializeUI() {
    console.log("Initializing UI event listeners...");
    document.getElementById('save-config-button').addEventListener('click', saveConfiguration);
    document.getElementById('load-config-button').addEventListener('click', loadSelectedConfiguration);
    document.getElementById('update-button').addEventListener('click', fetchAndUpdateLLM);
    document.getElementById('gradient-toggle').addEventListener('change', () => updateNodeProbabilities(null)); // Redraw with current probs
    document.getElementById('download-session-log-button').addEventListener('click', downloadSessionLog);
    console.log("UI Listeners attached.");
}

// --- Configuration Management ---

async function saveConfiguration() {
    const configNameInput = document.getElementById('config-name');
    const configName = configNameInput.value.trim();
    if (!configName) {
        setStatusMessage("Please enter a name for the configuration.", true);
        return;
    }
    if (!cy) {
         setStatusMessage("Cytoscape graph not initialized.", true);
         return;
     }

    setStatusMessage("Saving configuration...", false);
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

        if (!response.ok) {
            throw new Error(result.detail || `HTTP error ${response.status}`);
        }

        setStatusMessage(`Configuration '${result.config_name}' saved successfully (ID: ${result.config_id}).`, false);
        currentConfig = { // Update current config state
             id: result.config_id,
             name: result.config_name,
             graph_structure: currentGraphStructure
        };
        document.getElementById('current-config-name').textContent = currentConfig.name;
        configNameInput.value = ''; // Clear input
        await loadConfigList(); // Refresh dropdown
        clearSessionLog(); // Saving creates a new context

    } catch (error) {
        console.error('Error saving configuration:', error);
        setStatusMessage(`Error saving configuration: ${error.message}`, true);
    }
}

async function loadConfigList() {
    console.log("Loading configuration list...");
    setStatusMessage("Loading config list...", false);
    try {
        const response = await fetch('/api/configs');
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        const configs = await response.json();
        const select = document.getElementById('load-config-select');
        select.innerHTML = '<option value="">-- Select a config --</option>'; // Clear existing options
        configs.forEach(config => {
            const option = document.createElement('option');
            option.value = config.id;
            option.textContent = config.name;
            select.appendChild(option);
        });
         setStatusMessage("Config list loaded.", false);
         console.log("Config list updated.");
    } catch (error) {
        console.error('Error loading configuration list:', error);
        setStatusMessage(`Error loading config list: ${error.message}`, true);
    }
}

async function loadSelectedConfiguration() {
    const select = document.getElementById('load-config-select');
    const configId = select.value;
    if (!configId) {
        setStatusMessage("Please select a configuration to load.", true);
        return;
    }
    if (!cy) {
         setStatusMessage("Cytoscape graph not initialized.", true);
         return;
     }

    setStatusMessage(`Loading configuration ${configId}...`, false);
    try {
        const response = await fetch(`/api/configs/${configId}`);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` }));
            throw new Error(errorData.detail || `HTTP error ${response.status}`);
        }
        const configData = await response.json();

        // Update Cytoscape
        const graphElements = configData.graph_structure.nodes.map(n => ({ data: { id: n.id, fullName: n.fullName, nodeType: n.nodeType } }))
            .concat(configData.graph_structure.edges.map(e => ({ data: { source: e.source, target: e.target } })));

        cy.elements().remove(); // Clear existing graph
        cy.add(graphElements); // Add new elements
        runLayout(); // Apply layout

        currentConfig = configData; // Update global state
        document.getElementById('config-name').value = currentConfig.name; // Set name input for potential re-save
         document.getElementById('current-config-name').textContent = currentConfig.name;
        updateInputControls(currentConfig.graph_structure.nodes); // Update input fields
        updateNodeProbabilities({}); // Reset probabilities on graph
        clearSessionLog(); // Loading resets the session

        setStatusMessage(`Configuration '${currentConfig.name}' loaded successfully.`, false);
        console.log(`Loaded config ${configId}`);

    } catch (error) {
        console.error('Error loading selected configuration:', error);
        setStatusMessage(`Error loading configuration: ${error.message}`, true);
    }
}

function runLayout() {
     if (!cy) return;
     let layoutName = 'cola'; // Default
     try {
         // Check if layout is registered (basic check)
         if (typeof cy.layout({ name: 'cola' }).run !== 'function') throw new Error("Cola not registered");
     } catch(e) {
         console.warn("Cola layout failed or not registered, trying dagre...");
         layoutName = 'dagre';
          try {
              if (typeof cy.layout({ name: 'dagre' }).run !== 'function') throw new Error("Dagre not registered");
         } catch(e) {
             console.error("Neither Cola nor Dagre layout seem available!");
             layoutName = 'grid'; // Absolute fallback
         }
     }

    console.log(`Running layout: ${layoutName}`);
     const layoutOptions = layoutName === 'cola'
         ? { name: 'cola', animate: true, nodeSpacing: 50, edgeLength: 180, padding: 30 }
         : layoutName === 'dagre'
             ? { name: 'dagre', rankDir: 'TB', spacingFactor: 1.2, animate: true, padding: 30 }
             : { name: 'grid' }; // Fallback

    cy.layout(layoutOptions).run();
}


function setStatusMessage(message, isError = false) {
    const statusElement = document.getElementById('config-status');
    statusElement.textContent = message;
    statusElement.className = isError ? 'error' : '';
}

// --- Dynamic Input Controls ---

function updateInputControls(nodes) {
    const container = document.getElementById('input-controls-container');
    container.innerHTML = ''; // Clear existing
    const inputNodes = nodes.filter(n => n.nodeType === 'input');

    if (inputNodes.length === 0) {
        container.innerHTML = '<p>No input nodes defined in this configuration.</p>';
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
        input.value = "0.5"; // Default value

        div.appendChild(label);
        div.appendChild(input);
        container.appendChild(div);
    });
}

// --- Prediction ---

async function fetchAndUpdateLLM() {
    if (!currentConfig || !currentConfig.graph_structure || currentConfig.graph_structure.nodes.length === 0) {
        alert("Please load or define a graph configuration first.");
        return;
    }
     if (!cy) {
         alert("Cytoscape not initialized.");
         return;
     }

    // Gather input values dynamically
    const inputData = { input_values: {} };
    const inputNodes = currentConfig.graph_structure.nodes.filter(n => n.nodeType === 'input');
    let missingInput = false;
    inputNodes.forEach(node => {
        const inputElement = document.getElementById(`input-${node.id}`);
        if (inputElement) {
            const value = parseFloat(inputElement.value);
            if (isNaN(value)) {
                alert(`Invalid input for ${node.id}. Please enter a number between 0 and 1.`);
                missingInput = true;
            }
            inputData.input_values[node.id] = isNaN(value) ? 0.5 : Math.max(0, Math.min(1, value)); // Clamp and default NaN
        } else {
            console.warn(`Input element for ${node.id} not found. Using default 0.5`);
             inputData.input_values[node.id] = 0.5; // Default if element is missing
        }
    });

    if (missingInput) return;

    const payload = {
        input_values: inputData.input_values,
        graph_structure: currentConfig.graph_structure // Send the current graph structure
    };

    const updateButton = document.getElementById('update-button');
    updateButton.disabled = true;
    updateButton.textContent = 'Updating...';
    document.body.style.cursor = 'wait';
    clearLLMContextDisplay(); // Clear previous context

    try {
        const response = await fetch('/api/predict_openai_bn_single_call', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
             let errorDetail = `HTTP error! status: ${response.status}`;
             try { const errorData = await response.json(); errorDetail += `, detail: ${errorData.detail || JSON.stringify(errorData)}`; }
             catch (e) { errorDetail += `, response text: ${await response.text()}`; }
             throw new Error(errorDetail);
        }

        const result = await response.json();
        updateNodeProbabilities(result.probabilities);
        displayLLMContext(result.llm_context);
        logPrediction(inputData.input_values, result.probabilities); // Log the result

    } catch (error) {
        console.error('Error fetching LLM predictions:', error);
        alert(`Error fetching predictions:\n${error.message}`);
        clearLLMContextDisplayOnError();
    } finally {
        updateButton.disabled = false;
        updateButton.textContent = 'Update Probabilities';
        document.body.style.cursor = 'default';
    }
}

// --- Node Appearance Update ---

function updateNodeProbabilities(probabilities) {
     if (!cy) return;
    const useGradient = document.getElementById('gradient-toggle').checked;

    cy.nodes().forEach(node => {
        const nodeId = node.id();
        const nodeType = node.data('nodeType'); // Get type ('input' or 'hidden')
        let probState1 = null;

        // Check if probability data exists for this node
        if (probabilities && probabilities[nodeId] && probabilities[nodeId]["1"] !== undefined) {
             probState1 = probabilities[nodeId]["1"];
            node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`);
        } else if (nodeType === 'input') {
            // For input nodes, display their set value if probabilities object is null (initial load/redraw)
            const inputElement = document.getElementById(`input-${nodeId}`);
            if (inputElement) {
                 probState1 = parseFloat(inputElement.value) || 0.5;
                  node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`);
            } else {
                 node.data('currentProbLabel', '(N/A)');
            }
        }
         else {
            // Hidden node with no probability yet
            node.data('currentProbLabel', '(N/A)');
        }

        // Determine base color by type
        let baseBgColor = '#ccc'; // Default grey
        let textColor = '#333';
        if (nodeType === 'input') {
            baseBgColor = '#add8e6'; // Light blue for inputs
        } else if (nodeType === 'hidden') {
            // Style change: hidden nodes now look like this (was pink)
             baseBgColor = '#f0e68c'; // Khaki for hidden/intermediate
        }
         // No specific output color anymore

        // Apply coloring based on toggle and probability
        let finalBgColor = baseBgColor;
        if (probState1 !== null) { // Only apply special colors if probability exists
            if (useGradient) {
                // Low prob -> red, high prob -> green gradient
                finalBgColor = `rgb(${Math.round(255 * (1 - probState1))}, ${Math.round(255 * probState1)}, 0)`;
                textColor = '#333'; // Ensure readability on gradient
            } else {
                // Clean coloring: dark purple background, white text
                finalBgColor = '#4B0082'; // Dark purple
                textColor = '#FFFFFF'; // White text
            }
        } else {
            // If no probability, revert to base color (unless clean mode is on)
            if (!useGradient) {
                 finalBgColor = '#4B0082'; // Dark purple
                 textColor = '#FFFFFF'; // White text
            }
        }

        node.style({
            'background-color': finalBgColor,
            'color': textColor
        });
         node.trigger('style'); // Force label redraw
    });
}

// --- LLM Context Display ---

function displayLLMContext(context) {
    if (!context) return;
    const inputDiv = document.getElementById('input-context');
    const structureDiv = document.getElementById('structure-context');
    const rulesDiv = document.getElementById('rules-context');
    const descriptionsDiv = document.getElementById('node-descriptions-context');

    // Input States
    let inputHtml = '<h3>Input States Provided:</h3><ul>';
    (context.input_states || []).forEach(input => {
        inputHtml += `<li><strong>${input.node}</strong> (${input.description || 'N/A'}): ${input.state} (prob approx ${input.value.toFixed(2)})</li>`;
    });
    inputHtml += '</ul>';
    inputDiv.innerHTML = inputHtml || '<p>No input states provided to LLM.</p>';

    // Node Descriptions
    let descHtml = '<h3>Node Descriptions:</h3><ul>';
    Object.entries(context.node_descriptions || {}).forEach(([node, desc]) => {
        descHtml += `<li><strong>${node}</strong>: ${desc}</li>`;
    });
    descHtml += '</ul>';
    descriptionsDiv.innerHTML = descHtml || '<p>No node descriptions provided to LLM.</p>';

    // Dependencies
    let structureHtml = '<h3>Node Dependencies (DAG):</h3><pre>';
    Object.entries(context.node_dependencies || {}).forEach(([node, parents]) => {
        structureHtml += `${node} <- ${parents.join(', ') || '(Input Node)'}\n`;
    });
    structureHtml += '</pre>';
    structureDiv.innerHTML = structureHtml || '<p>No dependencies provided to LLM.</p>';

    // Qualitative Rules
    let rulesHtml = '<h3>Qualitative Rules Provided:</h3><ul>';
    (context.qualitative_rules || []).forEach(rule => {
        rulesHtml += `<li><strong>${rule.node}</strong>: ${rule.qualitative || 'Generic influence.'}</li>`;
    });
    rulesHtml += '</ul>';
    rulesDiv.innerHTML = rulesHtml || '<p>No specific qualitative rules provided to LLM.</p>';
}

function clearLLMContextDisplay() {
     document.getElementById('input-context').innerHTML = '<h3>Input States Provided:</h3><p>Updating...</p>';
     document.getElementById('structure-context').innerHTML = '<h3>Node Dependencies (DAG):</h3><p>Updating...</p>';
     document.getElementById('rules-context').innerHTML = '<h3>Qualitative Rules Provided:</h3><p>Updating...</p>';
     document.getElementById('node-descriptions-context').innerHTML = '<h3>Node Descriptions:</h3><p>Updating...</p>';
}
function clearLLMContextDisplayOnError() {
     document.getElementById('input-context').innerHTML = '<h3>Input States Provided:</h3><p style="color:red;">Error fetching prediction context.</p>';
     document.getElementById('structure-context').innerHTML = '';
     document.getElementById('rules-context').innerHTML = '';
     document.getElementById('node-descriptions-context').innerHTML = '';
}


// --- Session Logging ---

function logPrediction(inputs, probabilities) {
    const timestamp = new Date().toISOString();
    const logEntry = {
        timestamp: timestamp,
        configId: currentConfig.id || "unsaved",
        configName: currentConfig.name,
        inputs: { ...inputs }, // Clone input values
        probabilities: {} // Store P(1) for all nodes
    };

    // Extract P(1) for all nodes from the result
    for (const nodeId in probabilities) {
        logEntry.probabilities[nodeId] = probabilities[nodeId]["1"];
    }
     // Add input node values to probabilities as well for complete record
     for (const inputId in inputs) {
         if (!(inputId in logEntry.probabilities)) {
             logEntry.probabilities[inputId] = inputs[inputId];
         }
     }


    sessionLog.push(logEntry);
    document.getElementById('log-count').textContent = `Log entries this session: ${sessionLog.length}`;
    console.log("Prediction logged to session:", logEntry);
}

function clearSessionLog() {
    sessionLog = [];
    document.getElementById('log-count').textContent = `Log entries this session: 0`;
    console.log("Session log cleared.");
}

function downloadSessionLog() {
    if (sessionLog.length === 0) {
        alert("No data logged in this session yet.");
        return;
    }

    console.log("Generating session log CSV...");

    // Determine all unique node IDs across all log entries to create headers
    const allNodeIds = new Set();
    sessionLog.forEach(entry => {
        Object.keys(entry.probabilities).forEach(nodeId => allNodeIds.add(nodeId));
    });
    const sortedNodeIds = Array.from(allNodeIds).sort();

    const csvData = sessionLog.map(entry => {
        const row = {
            Timestamp: entry.timestamp,
            ConfigID: entry.configId,
            ConfigName: entry.configName,
        };
        // Add probability for each node, leave blank if not present in this entry
        sortedNodeIds.forEach(nodeId => {
            row[nodeId] = entry.probabilities[nodeId] !== undefined ? entry.probabilities[nodeId].toFixed(4) : '';
        });
        return row;
    });

    const csvHeaders = ["Timestamp", "ConfigID", "ConfigName", ...sortedNodeIds];

    const csvString = Papa.unparse({
        fields: csvHeaders,
        data: csvData
    });

    // Trigger download
    const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    const safeConfigName = (currentConfig.name || 'unsaved').replace(/[^a-z0-9]/gi, '_').toLowerCase();
    const timestampStr = new Date().toISOString().replace(/[:.]/g, '-');
    link.setAttribute("download", `bn_session_log_${safeConfigName}_${timestampStr}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    console.log("CSV download triggered.");
}


console.log("Initial script execution finished. Waiting for DOMContentLoaded.");
