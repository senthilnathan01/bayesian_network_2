// --- public/script.js ---

// --- Initialize Cytoscape (cy variable) ---
const graphElements = [
    { data: { id: 'A1', fullName: 'Domain Expertise' } },
    { data: { id: 'A2', fullName: 'Web Literacy' } },
    { data: { id: 'A3', fullName: 'Task Familiarity' } },
    { data: { id: 'A4', fullName: 'Goal Clarity' } },
    { data: { id: 'A5', fullName: 'Motivation' } },
    { data: { id: 'UI', fullName: 'UI State (Quality/Clarity)' } },
    { data: { id: 'H', fullName: 'History (Relevant past interactions)' } },
    { data: { id: 'IS1', fullName: 'Confidence' } },
    { data: { id: 'IS2', fullName: 'Cognitive Load' } },
    { data: { id: 'IS3', fullName: 'Task Understanding' } },
    { data: { id: 'IS4', fullName: 'Interaction Fluency' } },
    { data: { id: 'IS5', fullName: 'Relevant Knowledge Activation' } },
    { data: { id: 'O1', fullName: 'Predicted Success Probability' } },
    { data: { id: 'O2', fullName: 'Action Speed/Efficiency' } },
    { data: { id: 'O3', fullName: 'Help Seeking Likelihood' } },
    { data: { source: 'A1', target: 'IS3' } },
    { data: { source: 'A3', target: 'IS3' } },
    { data: { source: 'A4', target: 'IS3' } },
    { data: { source: 'H', target: 'IS3' } },
    { data: { source: 'A2', target: 'IS4' } },
    { data: { source: 'UI', target: 'IS4' } },
    { data: { source: 'H', target: 'IS4' } },
    { data: { source: 'A1', target: 'IS5' } },
    { data: { source: 'A3', target: 'IS5' } },
    { data: { source: 'IS3', target: 'IS5' } },
    { data: { source: 'IS4', target: 'IS2' } },
    { data: { source: 'IS3', target: 'IS2' } },
    { data: { source: 'A5', target: 'IS1' } },
    { data: { source: 'IS2', target: 'IS1' } },
    { data: { source: 'IS3', target: 'IS1' } },
    { data: { source: 'IS4', target: 'IS1' } },
    { data: { source: 'IS5', target: 'IS1' } },
    { data: { source: 'IS1', target: 'O1' } },
    { data: { source: 'IS2', target: 'O1' } },
    { data: { source: 'IS3', target: 'O1' } },
    { data: { source: 'IS5', target: 'O1' } },
    { data: { source: 'IS1', target: 'O2' } },
    { data: { source: 'IS2', target: 'O2' } },
    { data: { source: 'A5', target: 'O2' } },
    { data: { source: 'IS1', target: 'O3' } },
    { data: { source: 'IS2', target: 'O3' } },
    { data: { source: 'IS3', target: 'O3' } }
];

const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: graphElements,
    style: [
        {
            selector: 'node',
            style: {
                'background-color': '#ccc',
                'label': function(node) {
                    const id = node.data('id');
                    const fullName = node.data('fullName') || id;
                    const currentLabel = node.data('currentProbLabel');
                    return `${id}: ${fullName}\n${currentLabel || '(N/A)'}`;
                },
                'width': 120,
                'height': 120,
                'shape': 'ellipse',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '10px',
                'text-wrap': 'wrap',
                'text-max-width': 110,
                'text-outline-color': '#fff',
                'text-outline-width': 1,
                'color': '#333',
                'transition-property': 'background-color, color',
                'transition-duration': '0.5s'
            }
        },
        {
            selector: 'node[id = "A1"], node[id = "A2"], node[id = "A3"], node[id = "A4"], node[id = "A5"], node[id = "UI"], node[id = "H"]',
            style: {
                'background-color': '#add8e6',
                'shape': 'rectangle',
                'width': 130,
                'height': 70
            }
        },
        {
            selector: 'node[id = "O1"], node[id = "O2"], node[id = "O3"]',
            style: {
                'background-color': '#90ee90',
                'shape': 'roundrectangle',
                'width': 130,
                'height': 70
            }
        },
        {
            selector: 'node[id ^= "IS"]',
            style: {
                'background-color': '#ffb6c1',
                'shape': 'ellipse'
            }
        },
        {
            selector: 'edge',
            style: {
                'width': 2,
                'line-color': '#666',
                'target-arrow-shape W': 'triangle',
                'target-arrow-color': '#666',
                'curve-style': 'bezier'
            }
        }
    ],
    layout: {
        name: 'cola',
        animate: true,
        ungrabifyWhileSimulating: false,
        nodeSpacing: 40,
        edgeLength: 150,
        padding: 20,
        randomize: false
    }
});

// --- Function to update node appearance based on probabilities ---
function updateNodeProbabilities(probabilities) {
    const useGradient = document.getElementById('gradient-toggle').checked;

    cy.nodes().forEach(node => {
        const nodeId = node.id();
        if (probabilities[nodeId] && probabilities[nodeId]["1"] !== undefined) {
            const probState1 = probabilities[nodeId]["1"];
            node.data('currentProbLabel', `P(1)=${probState1.toFixed(3)}`);

            if (useGradient) {
                // Reversed gradient: low prob -> red, high prob -> green
                const color = `rgb(${Math.round(255 * (1 - probState1))}, ${Math.round(255 * probState1)}, 0)`;
                node.style({
                    'background-color': color,
                    'color': '#333' // Default text color for gradient mode
                });
            } else {
                // Clean coloring: dark purple background, white text
                node.style({
                    'background-color': '#4B0082', // Dark purple
                    'color': '#FFFFFF' // White text
                });
            }
        } else {
            node.data('currentProbLabel', '(N/A)');
            // Apply default colors based on node type
            let bgColor, textColor = '#333';
            if (['A1', 'A2', 'A3', 'A4', 'A5', 'UI', 'H'].includes(nodeId)) {
                bgColor = '#add8e6';
            } else if (['O1', 'O2', 'O3'].includes(nodeId)) {
                bgColor = '#90ee90';
            } else {
                bgColor = '#ffb6c1';
            }
            if (!useGradient) {
                bgColor = '#4B0082';
                textColor = '#FFFFFF';
            }
            node.style({
                'background-color': bgColor,
                'color': textColor
            });
        }
    });
}

// --- Function to display LLM context ---
function displayLLMContext(context) {
    const inputDiv = document.getElementById('input-context');
    const structureDiv = document.getElementById('structure-context');
    const rulesDiv = document.getElementById('rules-context');
    const descriptionsDiv = document.getElementById('node-descriptions-context');

    let inputHtml = '<h3>Input States Provided:</h3><ul>';
    context.input_states.forEach(input => {
        inputHtml += `<li><strong>${input.node}</strong> (${input.description}): ${input.state} (prob approx ${input.value.toFixed(2)})</li>`;
    });
    inputHtml += '</ul>';
    inputDiv.innerHTML = inputHtml;

    let descHtml = '<h3>Node Descriptions:</h3><ul>';
    Object.entries(context.node_descriptions).forEach(([node, desc]) => {
        descHtml += `<li><strong>${node}</strong>: ${desc}</li>`;
    });
    descHtml += '</ul>';
    descriptionsDiv.innerHTML = descHtml;

    let structureHtml = '<h3>Node Dependencies (DAG):</h3><pre>';
    Object.entries(context.node_dependencies).forEach(([node, parents]) => {
        structureHtml += `${node} <- ${parents.join(', ')}\n`;
    });
    structureHtml += '</pre>';
    structureDiv.innerHTML = structureHtml;

    let rulesHtml = '<h3>Qualitative Rules Provided:</h3><ul>';
    context.qualitative_rules.forEach(rule => {
        rulesHtml += `<li><strong>${rule.node}</strong> (${rule.description}): ${rule.qualitative || 'No specific rule provided.'}</li>`;
    });
    rulesHtml += '</ul>';
    rulesDiv.innerHTML = rulesHtml;
}

// --- Function to gather inputs and fetch predictions from backend ---
async function fetchAndUpdateLLM() {
    const inputData = {
        A1: parseFloat(document.getElementById('input-A1').value) || 0.5,
        A2: parseFloat(document.getElementById('input-A2').value) || 0.5,
        A3: parseFloat(document.getElementById('input-A3').value) || 0.5,
        A4: parseFloat(document.getElementById('input-A4').value) || 0.5,
        A5: parseFloat(document.getElementById('input-A5').value) || 0.5,
        UI: parseFloat(document.getElementById('input-UI').value) || 0.5,
        H: parseFloat(document.getElementById('input-H').value) || 0.5,
    };

    const updateButton = document.getElementById('update-button');
    updateButton.disabled = true;
    updateButton.textContent = 'Updating...';
    document.body.style.cursor = 'wait';

    document.getElementById('input-context').innerHTML = '<h3>Input States Provided:</h3><p>Updating...</p>';
    document.getElementById('structure-context').innerHTML = '<h3>Node Dependencies (DAG):</h3><p>Updating...</p>';
    document.getElementById('rules-context').innerHTML = '<h3>Qualitative Rules Provided:</h3><p>Updating...</p>';
    document.getElementById('node-descriptions-context').innerHTML = '<h3>Node Descriptions:</h3><p>Updating...</p>';

    try {
        const response = await fetch('/api/predict_openai_bn_single_call', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        });

        if (!response.ok) {
            let errorDetail = `HTTP error! status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorDetail += `, detail: ${errorData.detail || JSON.stringify(errorData)}`;
            } catch (e) {
                errorDetail += `, response text: ${await response.text()}`;
            }
            throw new Error(errorDetail);
        }

        const result = await response.json();
        const allNodeProbabilities = result.probabilities;
        const llmContext = result.llm_context;

        updateNodeProbabilities(allNodeProbabilities);
        displayLLMContext(llmContext);

    } catch (error) {
        console.error('Error fetching LLM predictions:', error);
        alert(`Error fetching predictions:\n${error.message}`);
        document.getElementById('input-context').innerHTML = '<h3>Input States Provided:</h3><p style="color:red;">Error loading context.</p>';
        document.getElementById('structure-context').innerHTML = '';
        document.getElementById('rules-context').innerHTML = '';
        document.getElementById('node-descriptions-context').innerHTML = '';
    } finally {
        updateButton.disabled = false;
        updateButton.textContent = 'Update Probabilities';
        document.body.style.cursor = 'default';
    }
}

// --- Add event listeners ---
document.getElementById('update-button').addEventListener('click', fetchAndUpdateLLM);

// Add listener for gradient toggle
document.getElementById('gradient-toggle').addEventListener('change', () => {
    // Re-run updateNodeProbabilities with current probabilities to apply new coloring
    cy.nodes().forEach(node => {
        const nodeId = node.id();
        const prob = node.data('currentProbLabel') ? parseFloat(node.data('currentProbLabel').replace('P(1)=', '')) : null;
        const probabilities = prob ? { [nodeId]: { "1": prob } } : {};
        updateNodeProbabilities(probabilities);
    });
});

// --- Initialize node appearance and layout on page load ---
cy.ready(function() {
    console.log("Cytoscape instance is ready. Attempting to run initial layout.");
    let layoutSuccess = false;

    if (typeof window.cola !== 'undefined') {
        try {
            cy.layout({
                name: 'cola',
                animate: true,
                ungrabifyWhileSimulating: false,
                nodeSpacing: 40,
                edgeLength: 150,
                padding: 20,
                randomize: false
            }).run();
            console.log("Initial Cola layout run initiated.");
            layoutSuccess = true;
        } catch (colaError) {
            console.error("ERROR running initial Cola layout:", colaError);
            alert("Cola layout failed. Falling back to Dagre layout.");
        }
    } else {
        console.warn("Cola layout not available. Falling back to Dagre layout.");
    }

    if (!layoutSuccess) {
        try {
            cy.layout({
                name: 'dagre',
                rankDir: 'TB',
                spacingFactor: 1.2,
                animate: true,
                padding: 20
            }).run();
            console.log("Fallback Dagre layout run initiated.");
            layoutSuccess = true;
        } catch (dagreError) {
            console.error("ERROR running fallback Dagre layout:", dagreError);
            alert("Fallback Dagre layout also failed. Graph display might be broken.");
        }
    }

    try {
        updateNodeProbabilities({});
        console.log("Initial node probabilities set.");
        displayLLMContext({
            input_states: [{ node: 'Loading...', description: '', value: 0, state: '' }],
            node_dependencies: { 'Loading...': [] },
            qualitative_rules: [{ node: 'Loading...', description: '', qualitative: '' }],
            node_descriptions: { 'Loading...': 'Loading...' }
        });
        console.log("Initial LLM context placeholder set.");
    } catch (initError) {
        console.error("ERROR setting initial node probabilities or context:", initError);
        alert("Error initializing graph data. Check the console for details.");
    }
});

console.log("Initial script execution finished. Waiting for cy.ready().");
